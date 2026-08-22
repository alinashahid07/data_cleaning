import hashlib
import io
import numpy as np
import pandas as pd
import streamlit as st

# fingerprint helper
# files above this row count get dtype downcasting and sampled analysis
LARGE_FILE_THRESHOLD = 50_000
# files above this get a warning banner shown to the user
LARGE_FILE_WARN_THRESHOLD = 50_000
# rows used for sampled analysis functions on large files
ANALYSIS_SAMPLE_SIZE = 20_000

def make_df_key(df: pd.DataFrame) -> str:
    """
    Cheap string fingerprint for a dataframe.
    Used as the cache key instead of passing the full df,
    so st.cache_data never has to hash thousands of cells.
    Shape + dtype signature + hash of first and last 5 rows covers
    virtually all cases where the df has actually changed.
    """
    shape_part = f"{df.shape[0]}x{df.shape[1]}"
    dtype_part = "".join(str(t) for t in df.dtypes)
    sample = pd.concat([df.head(5), df.tail(5)])
    sample_hash = str(pd.util.hash_pandas_object(sample).sum())
    return f"{shape_part}__{sample_hash}__{hash(dtype_part)}"

def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts int64 to int32 and float64 to float32 to cut memory roughly in half.
    Only applied to large files. A warning is shown to the user at load time
    because int32 max is 2 billion and float32 has less decimal precision than float64.
    Columns that would overflow int32 are left as int64.
    """
    for col in df.select_dtypes(include="integer").columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].astype(np.float32)
    return df

def _is_large(df: pd.DataFrame) -> bool:
    return len(df) >= LARGE_FILE_THRESHOLD

def _sample_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a random sample of the df for expensive analysis functions.
    On small files returns the full df. On large files returns ANALYSIS_SAMPLE_SIZE rows.
    The sample is seeded so results are stable across reruns.
    """
    if len(df) <= ANALYSIS_SAMPLE_SIZE:
        return df
    return df.sample(ANALYSIS_SAMPLE_SIZE, random_state=42)

# file loading

@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, filename: str, file_id: str, sheet_name=None):
    """
    Loads a CSV or Excel file from bytes.
    For CSV files above LARGE_FILE_THRESHOLD rows, reads in chunks and concatenates
    so peak memory during load stays low. Dtype downcasting is applied automatically
    to large files to cut memory usage roughly in half.
    A warning is stored in session state so app.py can show it after load.
    """
    ext = filename.split(".")[-1].lower()
    buf = io.BytesIO(file_bytes)
    if ext == "csv":
        CHUNK = 50_000
        chunks = []
        reader = pd.read_csv(
            buf,
            quotechar='"',
            skipinitialspace=True,
            chunksize=CHUNK,
        )
        for chunk in reader:
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]
    else:
        df = pd.read_excel(buf, sheet_name=sheet_name)

    if _is_large(df):
        df = _downcast_numeric(df)
        st.session_state["_large_file_warning"] = (
            f"Large file detected ({len(df):,} rows). "
            "Numeric columns were downcast to 32-bit to save memory. "
            "int32 max is 2.1 billion. float32 has slightly less decimal precision than float64. "
            "This has no effect on cleaning operations."
        )
    else:
        st.session_state.pop("_large_file_warning", None)
    return df

# cached analysis functions, all keyed on df_key (cheap string)
# df itself is passed separately and NOT used as a cache key discriminator
# st.cache_data hashes ALL args, so we pass df last and rely on df_key
# being unique enough to bust the cache correctly

# get stats summary
@st.cache_data(show_spinner=False)
def get_dataframe_stats(df_key, _df):
    df = _df
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": len(df.select_dtypes(include=np.number).columns),
        "categorical_cols": len(df.select_dtypes(exclude=np.number).columns),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024 ** 2,
        "is_large": _is_large(df),
    }

# per column stats for profiler
@st.cache_data(show_spinner=False)
def get_column_profile(df_key, _df):
    df = _df
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        row = {
            "Column": col,
            "Type": str(s.dtype),
            "Non-Null": int(s.notna().sum()),
            "Null": int(s.isna().sum()),
            "Null %": f"{s.isna().mean() * 100:.1f}%",
            "Unique": int(s.nunique()),
        }
        if is_num:
            row.update({
                "Min": round(float(s.min()), 4) if s.notna().any() else None,
                "Max": round(float(s.max()), 4) if s.notna().any() else None,
                "Mean": round(float(s.mean()), 4) if s.notna().any() else None,
                "Median": round(float(s.median()), 4) if s.notna().any() else None,
                "Std": round(float(s.std()), 4) if s.notna().any() else None,
                "Skew": round(float(s.skew()), 4) if s.notna().any() else None,
                "Sample Values": ", ".join(str(v) for v in s.dropna().head(3).tolist()),
            })
        else:
            row.update({
                "Min": "-", "Max": "-", "Mean": "-",
                "Median": "-", "Std": "-", "Skew": "-",
                "Sample Values": ", ".join(
                    f'"{v}"' for v in s.dropna().value_counts().head(3).index.tolist()
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows)

# analyze plus recommend, cached together
@st.cache_data(show_spinner=False)
def get_analysis_and_recommendations(df_key, _df, conversion_threshold):
    """
    On large files this runs on a random sample to stay fast.
    The sample size is large enough that recommendations are still reliable.
    """
    df = _df
    sample = _sample_for_analysis(df)
    is_sampled = len(sample) < len(df)
    issues = {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_cols": max(
            len(df.columns[df.columns.duplicated()].tolist()),
            len(df.columns) - len(df.T.drop_duplicates().T.columns),
        ),
        "whitespace_cols": [],
        "currency_cols": [],
        "percentage_cols": [],
        "unit_cols": [],
        "duration_cols": [],
        "missing_cells": int(df.isna().sum().sum()),
        "missing_cols": [],
        "edge_char_cols": [],
        "is_sampled": is_sampled,
        "sample_size": len(sample),
    }
    currency_pattern = r'[$€£¥₹₽₺₩฿]|(USD|EUR|GBP|JPY|CNY|INR|PKR)'
    for col in sample.select_dtypes(include="object").columns:
        s_str = sample[col].astype(str)
        if s_str.str.strip().ne(s_str).any():
            issues["whitespace_cols"].append(col)
        non_empty = s_str.str.strip().replace("", np.nan).dropna()
        if non_empty.empty:
            continue
        if (
            non_empty.str.contains(r"\d", regex=True)
            & non_empty.str.contains(currency_pattern, case=False, regex=True)
        ).mean() > conversion_threshold:
            issues["currency_cols"].append(col)
        elif non_empty.str.contains("%").mean() > conversion_threshold:
            issues["percentage_cols"].append(col)
        elif non_empty.str.contains(r"\d+\s?(kg|g|cm|mm|km|ml|l|lb)", case=False, regex=True).mean() > conversion_threshold:
            issues["unit_cols"].append(col)
        elif non_empty.str.contains(r"(h|hr|hour|min|minute|sec|second)", case=False, regex=True).mean() > conversion_threshold:
            issues["duration_cols"].append(col)
        if (s_str.str.match(r"^\W") | s_str.str.match(r"\W$")).sum() > 0:
            issues["edge_char_cols"].append(col)
    for col in df.columns:
        p = df[col].isna().mean()
        if p > 0:
            issues["missing_cols"].append((col, p))

    recs = []
    if issues["duplicate_rows"] > 0:
        recs.append(("", f"Found {issues['duplicate_rows']} duplicate rows", "Removing duplicates prevents skewed analysis.", "drop_duplicates"))
    if issues["duplicate_cols"] > 0:
        recs.append(("", f"Found {issues['duplicate_cols']} duplicate columns", "These columns are wasting memory.", "drop_dup_cols"))
    if issues["whitespace_cols"]:
        recs.append(("", f"{len(issues['whitespace_cols'])} columns have extra whitespace", f"Columns: {', '.join(issues['whitespace_cols'][:3])}", "strip_whitespace"))
    if issues["currency_cols"]:
        recs.append(("", f"{len(issues['currency_cols'])} columns look like currency but are not numeric", f"Columns: {', '.join(issues['currency_cols'][:3])}", "convert_currency"))
    if issues["percentage_cols"]:
        recs.append(("", f"{len(issues['percentage_cols'])} columns contain percentages as text", f"Columns: {', '.join(issues['percentage_cols'][:3])}", "convert_percentage"))
    if issues["unit_cols"]:
        recs.append(("", f"{len(issues['unit_cols'])} columns have measurement units mixed in", f"Columns: {', '.join(issues['unit_cols'][:3])}", "convert_units"))
    if issues["duration_cols"]:
        recs.append(("", f"{len(issues['duration_cols'])} columns contain time durations", f"Columns: {', '.join(issues['duration_cols'][:3])}", "convert_duration"))
    if issues["missing_cells"] > 0:
        top = sorted(issues["missing_cols"], key=lambda x: x[1], reverse=True)[:3]
        recs.append(("", f"{issues['missing_cells']} missing values found", f"Worst: {', '.join(f'{c} ({p*100:.0f}%)' for c, p in top)}", "handle_missing"))
    if issues["edge_char_cols"]:
        recs.append(("", f"{len(issues['edge_char_cols'])} columns have unwanted edge characters", f"Columns: {', '.join(issues['edge_char_cols'][:3])}", "clean_edges"))
    return issues, recs

# histogram plus kde data for a numeric column
@st.cache_data(show_spinner=False)
def get_histogram_data(df_key, series, n_bins=30):
    """
    On large series uses a sample for the KDE calculation which is the slow part.
    Histogram counts still use the full series for accuracy.
    """
    clean = series.dropna()
    if clean.empty:
        return {}
    counts, edges = np.histogram(clean, bins=n_bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    kde_source = clean.sample(min(10_000, len(clean)), random_state=42) if len(clean) > 10_000 else clean
    n = len(kde_source)
    std = kde_source.std()
    if std == 0:
        kde_x = bin_centers.tolist()
        kde_y = [0.0] * len(bin_centers)
    else:
        bw = 1.06 * std * (n ** -0.2)
        kde_x = np.linspace(edges[0], edges[-1], 200)
        diffs = (kde_x[:, None] - kde_source.values[None, :]) / bw
        kde_y = np.exp(-0.5 * diffs ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
        kde_x = kde_x.tolist()
        kde_y = kde_y.tolist()
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    return {
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "kde_x": kde_x,
        "kde_y": kde_y,
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "lower_fence": q1 - 1.5 * iqr,
        "upper_fence": q3 + 1.5 * iqr,
        "n_outliers": int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()),
    }

# value counts for a categorical column
@st.cache_data(show_spinner=False)
def get_bar_data(df_key, series, top_n=20):
    clean = series.dropna()
    if clean.empty:
        return {}
    counts = clean.value_counts().head(top_n)
    total = len(clean)
    return {
        "labels": [str(v) for v in counts.index.tolist()],
        "counts": counts.tolist(),
        "pct": [round(c / total * 100, 1) for c in counts.tolist()],
        "total": total,
        "unique": int(series.nunique()),
        "shown": len(counts),
    }

# row sampled null matrix, capped so the chart stays readable
@st.cache_data(show_spinner=False)
def get_missing_heatmap_data(df_key, _df):
    """
    Caps the sample at 300 rows for rendering. On large files this is sampled
    from the full df so the heatmap is still representative.
    """
    df = _df
    max_rows = 300
    sample = df.sample(max_rows, random_state=0) if len(df) > max_rows else df
    cols_with_nulls = [c for c in sample.columns if sample[c].isna().any()]
    if not cols_with_nulls:
        return {}
    matrix = sample[cols_with_nulls].isna().astype(int)
    return {
        "columns": cols_with_nulls,
        "matrix": matrix.values.tolist(),
        "null_pct": {c: round(df[c].isna().mean() * 100, 1) for c in cols_with_nulls},
        "n_rows_shown": len(sample),
        "total_rows": len(df),
    }

# correlation matrix for all numeric columns, melted into long form
@st.cache_data(show_spinner=False)
def get_correlation_data(df_key, _df, method="pearson"):
    """
    On large files computes correlation on a sample to stay fast.
    Correlation on 20k rows is statistically equivalent to the full dataset.
    """
    df = _df
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return {}
    sample = _sample_for_analysis(df)
    corr = sample[num_cols].corr(method=method).round(3)
    rows = []
    for col_a in num_cols:
        for col_b in num_cols:
            val = corr.loc[col_a, col_b]
            rows.append({
                "col_a": col_a,
                "col_b": col_b,
                "value": round(float(val), 3),
                "abs_value": round(abs(float(val)), 3),
            })
    return {
        "rows": rows,
        "col_order": num_cols,
        "n_cols": len(num_cols),
        "is_sampled": len(sample) < len(df),
        "sample_size": len(sample),
    }

# scans every column and suggests a better type with a confidence score
@st.cache_data(show_spinner=False)
def get_type_suggestions(df_key, _df):
    """Only columns where the current type is wrong or suboptimal get a suggestion.
    On large files runs on a sample. Type patterns are consistent enough
    that 20k rows gives the same suggestions as the full dataset."""
    df = _df
    sample = _sample_for_analysis(df)
    email_pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
    currency_pattern = r'[$€£¥₹₽₺₩฿]|(USD|EUR|GBP|JPY|CNY|INR|PKR|Rs\.?)'
    bool_values = {"true", "false", "yes", "no", "1", "0", "y", "n"}
    suggestions = []

    for col in sample.columns:
        s = sample[col]
        current_type = str(s.dtype)
        non_null = s.dropna()
        n = len(non_null)
        if n == 0:
            continue

        # already numeric, check if it could be boolean
        if pd.api.types.is_numeric_dtype(s):
            unique_vals = set(s.dropna().unique())
            if unique_vals <= {0, 1, 0.0, 1.0}:
                suggestions.append({
                    "column": col,
                    "current_type": current_type,
                    "suggested_action": "convert_to_boolean",
                    "suggested_label": "Boolean",
                    "reason": "Only contains 0 and 1 values",
                    "confidence": 99,
                    "sample": ", ".join(str(v) for v in list(unique_vals)[:4]),
                })
            continue

        str_series = non_null.astype(str).str.strip()

        # email detection
        email_match = str_series.str.match(email_pattern).mean()
        if email_match >= 0.7:
            suggestions.append({
                "column": col,
                "current_type": current_type,
                "suggested_action": "validate_email",
                "suggested_label": "Email",
                "reason": f"{email_match*100:.0f}% of values look like email addresses",
                "confidence": round(email_match * 100),
                "sample": ", ".join(str_series.head(3).tolist()),
            })
            continue

        # boolean detection
        lower = str_series.str.lower()
        bool_match = lower.isin(bool_values).mean()
        if bool_match >= 0.7:
            suggestions.append({
                "column": col,
                "current_type": current_type,
                "suggested_action": "convert_to_boolean",
                "suggested_label": "Boolean",
                "reason": f"{bool_match*100:.0f}% of values are true/false/yes/no",
                "confidence": round(bool_match * 100),
                "sample": ", ".join(str_series.head(3).tolist()),
            })
            continue

        # currency detection
        currency_like = (
            str_series.str.contains(r"\d", regex=True)
            & str_series.str.contains(currency_pattern, case=False, regex=True)
        )
        if currency_like.mean() >= 0.6:
            suggestions.append({
                "column": col,
                "current_type": current_type,
                "suggested_action": "convert_currency",
                "suggested_label": "Numeric (currency)",
                "reason": f"{currency_like.mean()*100:.0f}% of values look like currency amounts",
                "confidence": round(currency_like.mean() * 100),
                "sample": ", ".join(str_series.head(3).tolist()),
            })
            continue

        # percentage detection
        pct_match = str_series.str.contains("%").mean()
        if pct_match >= 0.6:
            suggestions.append({
                "column": col,
                "current_type": current_type,
                "suggested_action": "convert_percentage",
                "suggested_label": "Numeric (percentage)",
                "reason": f"{pct_match*100:.0f}% of values contain a % symbol",
                "confidence": round(pct_match * 100),
                "sample": ", ".join(str_series.head(3).tolist()),
            })
            continue

        # unit measurement detection
        unit_match = str_series.str.contains(
            r"\d+\s?(kg|g|mg|cm|mm|km|ml|l|lb|lbs|oz|kgs)", case=False, regex=True
        ).mean()
        if unit_match >= 0.6:
            suggestions.append({
                "column": col,
                "current_type": current_type,
                "suggested_action": "convert_units",
                "suggested_label": "Numeric (measurement)",
                "reason": f"{unit_match*100:.0f}% of values contain a measurement unit",
                "confidence": round(unit_match * 100),
                "sample": ", ".join(str_series.head(3).tolist()),
            })
            continue

        # duration detection
        duration_match = str_series.str.contains(
            r"(h|hr|hour|min|minute|sec|second)", case=False, regex=True
        ).mean()
        if duration_match >= 0.6:
            suggestions.append({
                "column": col,
                "current_type": current_type,
                "suggested_action": "convert_duration",
                "suggested_label": "Numeric (duration in seconds)",
                "reason": f"{duration_match*100:.0f}% of values look like time durations",
                "confidence": round(duration_match * 100),
                "sample": ", ".join(str_series.head(3).tolist()),
            })
            continue

        # datetime detection
        date_sample = str_series.head(20)
        parsed = pd.to_datetime(date_sample, errors="coerce")
        date_match = parsed.notna().mean()
        if date_match >= 0.6:
            suggestions.append({
                "column": col,
                "current_type": current_type,
                "suggested_action": "convert_datetime",
                "suggested_label": "Datetime",
                "reason": f"{date_match*100:.0f}% of sampled values parse as dates",
                "confidence": round(date_match * 100),
                "sample": ", ".join(str_series.head(3).tolist()),
            })
            continue

        # plain numeric string detection
        cleaned = str_series.str.replace(r"[^\d.\-]", "", regex=True).replace("", np.nan)
        numeric_match = pd.to_numeric(cleaned, errors="coerce").notna().mean()
        if numeric_match >= 0.7:
            suggestions.append({
                "column": col,
                "current_type": current_type,
                "suggested_action": "convert_numeric",
                "suggested_label": "Numeric",
                "reason": f"{numeric_match*100:.0f}% of values are numeric strings",
                "confidence": round(numeric_match * 100),
                "sample": ", ".join(str_series.head(3).tolist()),
            })
            continue

        # low cardinality category detection
        unique_ratio = s.nunique() / n
        if unique_ratio <= 0.05 and s.nunique() <= 50:
            suggestions.append({
                "column": col,
                "current_type": current_type,
                "suggested_action": "convert_category",
                "suggested_label": "Category",
                "reason": f"Only {s.nunique()} unique values across {n} rows ({unique_ratio*100:.1f}% unique), storing as category saves memory",
                "confidence": round((1 - unique_ratio) * 100),
                "sample": ", ".join(str(v) for v in s.value_counts().head(3).index.tolist()),
            })

    return sorted(suggestions, key=lambda x: x["confidence"], reverse=True)

# scores data quality from 0 to 100 across five weighted dimensions
@st.cache_data(show_spinner=False)
def get_quality_score(df_key, _df):
    """Each dimension contributes 20 points. Cache is keyed on df_key so the score
    still updates live as the user cleans, without hashing the full dataframe.
    On large files uses a sample for the expensive per-column passes.
    Row-level metrics like duplicates and missing counts still use the full df."""
    df = _df
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return {"total": 0, "breakdown": {}}

    n_rows = len(df)
    n_cols = df.shape[1]
    sample = _sample_for_analysis(df)

    # completeness, deducted proportionally to the fraction of missing cells
    missing_frac = df.isna().sum().sum() / total_cells
    completeness_score = round(20 * (1 - missing_frac))

    # uniqueness, deducted proportionally to the fraction of duplicate rows
    dup_frac = df.duplicated().sum() / max(n_rows, 1)
    uniqueness_score = round(20 * (1 - dup_frac))

    # type consistency, flags object columns that mostly hold numeric text
    inconsistent_cols = 0
    for col in sample.select_dtypes(include="object").columns:
        s = sample[col].dropna().astype(str).str.strip()
        if len(s) == 0:
            continue
        numeric_frac = pd.to_numeric(
            s.str.replace(r"[^\d.\-]", "", regex=True).replace("", np.nan),
            errors="coerce"
        ).notna().mean()
        if numeric_frac > 0.7:
            inconsistent_cols += 1
    type_score = round(max(0, 20 - (inconsistent_cols / max(n_cols, 1)) * 20))

    # outlier cleanliness, uses 3x iqr fences to only flag extreme outliers
    num_cols = sample.select_dtypes(include=np.number).columns
    outlier_fracs = []
    for col in num_cols:
        s = sample[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_frac = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean()
        outlier_fracs.append(outlier_frac)
    avg_outlier_frac = float(np.mean(outlier_fracs)) if outlier_fracs else 0.0
    outlier_score = round(max(0, 20 * (1 - avg_outlier_frac * 5)))

    # validity, checks for placeholder text like none, na, unknown, null
    invalid_pattern = r"^(none|na|n/a|null|unknown|\?|nan|-|)$"
    invalid_fracs = []
    for col in sample.select_dtypes(include="object").columns:
        s = sample[col].dropna().astype(str).str.strip().str.lower()
        if len(s) == 0:
            continue
        invalid_frac = s.str.match(invalid_pattern, na=False).mean()
        invalid_fracs.append(invalid_frac)
    avg_invalid_frac = float(np.mean(invalid_fracs)) if invalid_fracs else 0.0
    validity_score = round(max(0, 20 * (1 - avg_invalid_frac * 3)))

    total = min(100, completeness_score + uniqueness_score + type_score + outlier_score + validity_score)

    def grade(s):
        if s >= 18:
            return "good"
        if s >= 12:
            return "fair"
        return "poor"

    return {
        "total": total,
        "breakdown": {
            "Completeness": {
                "score": completeness_score,
                "max": 20,
                "grade": grade(completeness_score),
                "detail": f"{int(missing_frac * total_cells)} missing cells ({missing_frac*100:.1f}%)",
            },
            "Uniqueness": {
                "score": uniqueness_score,
                "max": 20,
                "grade": grade(uniqueness_score),
                "detail": f"{int(dup_frac * n_rows)} duplicate rows ({dup_frac*100:.1f}%)",
            },
            "Type Consistency": {
                "score": type_score,
                "max": 20,
                "grade": grade(type_score),
                "detail": f"{inconsistent_cols} column(s) storing numbers as text",
            },
            "Outlier Cleanliness": {
                "score": outlier_score,
                "max": 20,
                "grade": grade(outlier_score),
                "detail": f"{avg_outlier_frac*100:.1f}% of numeric values are extreme outliers",
            },
            "Validity": {
                "score": validity_score,
                "max": 20,
                "grade": grade(validity_score),
                "detail": f"{avg_invalid_frac*100:.1f}% of text values are placeholder or invalid strings",
            },
        },
    }
