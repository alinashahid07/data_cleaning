import re
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

# knn and mice become too slow and memory hungry above this row count
# above this threshold we fall back to fast mean or median imputation
IMPUTER_ROW_LIMIT = 50_000

# parse a duration string into seconds
def _convert_duration_val(v):
    t = 0
    for n, u in re.findall(r"(\d+\.?\d*)\s*(h(?:ou?r)?|m(?:in)?|s(?:ec)?)", str(v).lower()):
        n = float(n)
        if u.startswith("h"):
            t += n * 3600
        elif u.startswith("m"):
            t += n * 60
        elif u.startswith("s"):
            t += n
    return t if t > 0 else np.nan

# convert messy numeric columns
def smart_column_cleaner(df, conversion_threshold=0.6, inplace=False):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return None if inplace else df.copy()
    df_clean = df if inplace else df.copy()
    currency_pattern = r'[$€£¥₹₽₺₩฿]|(USD|EUR|GBP|JPY|CNY|INR|PKR|AUD|CAD)'
    for col in df_clean.select_dtypes(include="object").columns:
        series = df_clean[col].astype(str).str.strip()
        non_empty = series.replace("", np.nan).dropna()
        if non_empty.empty:
            continue

        # currency detection
        currency_like = (
            non_empty.str.contains(r"\d", regex=True)
            & non_empty.str.contains(currency_pattern, case=False, regex=True)
        )
        if currency_like.mean() > conversion_threshold:
            cleaned = (
                non_empty.str.replace(r"[^\d.,\-()]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(r"\((.+?)\)", r"-\1", regex=True)
                .str.extract(r"([-]?\d[\d\.,]*)", expand=False)
                .str.replace(",", "", regex=False)
            )
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().mean() > conversion_threshold:
                df_clean[col] = converted.reindex(df_clean.index)
                continue

        # percentage detection
        if non_empty.str.contains("%").mean() > conversion_threshold:
            cleaned = non_empty.str.replace("%", "", regex=False).str.replace(r"[^\d.\-]", "", regex=True)
            converted = pd.to_numeric(cleaned, errors="coerce") / 100
            if converted.notna().mean() > conversion_threshold:
                df_clean[col] = converted.reindex(df_clean.index)
                continue

        # unit detection
        if non_empty.str.contains(r"\d+\s?(kg|g|mg|cm|mm|km|ml|l|lb|oz)", case=False, regex=True).mean() > conversion_threshold:
            cleaned = non_empty.str.extract(r"([-]?\d+\.?\d*)", expand=False)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().mean() > conversion_threshold:
                df_clean[col] = converted.reindex(df_clean.index)
                continue

        # generic numeric detection
        cleaned = non_empty.str.replace(r"[^\d.\-]", "", regex=True)
        converted = pd.to_numeric(cleaned, errors="coerce")
        if converted.notna().mean() > conversion_threshold:
            df_clean[col] = converted.reindex(df_clean.index)
    return None if inplace else df_clean

# fill missing values
def _fast_numeric_impute(df_clean, num_cols):
    """
    Mean imputation used when the file is too large for KNN or MICE.
    Much faster and uses almost no extra memory. Less accurate than KNN
    but the only practical option for 100k plus row files in a browser app.
    """
    for col in num_cols:
        if df_clean[col].isna().any():
            fill_val = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(fill_val)
    return df_clean

def missing_value_handler(df, threshold=0.3, inplace=False, numeric_strategy="auto"):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return None if inplace else df.copy()
    df_clean = df.copy() if not inplace else df
    n_rows = len(df_clean)
    is_large = n_rows >= IMPUTER_ROW_LIMIT
    if numeric_strategy == "auto":
        if is_large:
            numeric_strategy = "fast"
        elif df_clean.shape[1] > 50 or n_rows > 5000:
            numeric_strategy = "mice"
        else:
            numeric_strategy = "knn"

    # convert missing indicators
    df_clean = df_clean.replace(["?", "NA", "unknown", "n/a", "NaN", "null", -999, 999, 9999, ""], np.nan)

    # drop mostly empty columns
    cols_to_drop = df_clean.columns[df_clean.isna().mean() > threshold]
    if len(cols_to_drop):
        df_clean = df_clean.drop(columns=cols_to_drop)

    num_cols = df_clean.select_dtypes(include=np.number).columns
    cat_cols = df_clean.select_dtypes(exclude=np.number).columns

    # numeric imputation
    if not num_cols.empty and df_clean[num_cols].isna().any().any():
        if numeric_strategy == "fast":
            # mean imputation, runs in linear time, safe for 500k rows
            df_clean = _fast_numeric_impute(df_clean, num_cols)
        elif numeric_strategy == "knn":
            n_neighbors = max(1, min(5, max(1, n_rows // 1000), n_rows))
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
        else:
            # mice needs at least 2 numeric columns to model relationships
            if len(num_cols) >= 2:
                imputer = IterativeImputer(max_iter=10, random_state=42)
            else:
                imputer = KNNImputer(n_neighbors=max(1, min(3, n_rows)))
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])

    # categorical imputation
    for col in cat_cols:
        if df_clean[col].isna().any():
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Missing"
            df_clean[col] = df_clean[col].fillna(mode_val)

    return None if inplace else df_clean
