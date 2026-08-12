import numpy as np
import pandas as pd

# split a column on a delimiter
def split_column(df, col, delimiter, new_col_names, keep_original=False):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not delimiter:
        raise ValueError("Delimiter cannot be empty")
    if not new_col_names:
        raise ValueError("At least one output column name is required")
    tmp = df.copy()
    n_parts = len(new_col_names)
    split_result = tmp[col].astype(str).str.split(delimiter, expand=True)
    for i in range(n_parts):
        col_name = new_col_names[i].strip() or f"{col}_part{i + 1}"
        tmp[col_name] = split_result[i] if i < split_result.shape[1] else np.nan
    if not keep_original:
        tmp = tmp.drop(columns=[col])
    return tmp

# merge columns into one
def merge_columns(df, cols, new_col_name, separator, keep_originals=False):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if len(cols) < 2:
        raise ValueError("At least two columns are required for merging")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")
    if not new_col_name or not new_col_name.strip():
        raise ValueError("New column name cannot be empty")
    tmp = df.copy()
    merged = tmp[cols[0]].astype(str).str.strip()
    for c in cols[1:]:
        merged = merged + separator + tmp[c].astype(str).str.strip()
    tmp[new_col_name.strip()] = merged
    if not keep_originals:
        tmp = tmp.drop(columns=[c for c in cols if c != new_col_name.strip()])
    return tmp

# rename columns from a mapping
def rename_columns(df, rename_map):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    # skip blank entries and names that already match
    clean_map = {
        old: new.strip()
        for old, new in rename_map.items()
        if new.strip() and new.strip() != old and old in df.columns
    }
    if not clean_map:
        return df.copy()
    duplicates = [n for n in clean_map.values() if n in df.columns and n not in clean_map]
    if duplicates:
        raise ValueError(f"New names conflict with existing columns: {', '.join(duplicates)}")
    return df.rename(columns=clean_map)

# apply a batch of type guesser suggestions
def apply_type_suggestions(df, selected_suggestions):
    """selected_suggestions is a filtered list where the user ticked which ones to apply.
    Each entry needs column and suggested_action keys."""
    from .advanced import _convert_duration_val
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    tmp = df.copy()
    applied = []
    for suggestion in selected_suggestions:
        col = suggestion["column"]
        action = suggestion["suggested_action"]
        if col not in tmp.columns:
            continue
        s = tmp[col].astype(str).str.strip()
        non_empty = s.replace("", np.nan).dropna()
        try:
            if action == "convert_currency":
                cleaned = (
                    non_empty
                    .str.replace(r"[^\d.,\-()]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.replace(r"\((.+?)\)", r"-\1", regex=True)
                    .str.extract(r"([-]?\d[\d\.,]*)", expand=False)
                    .str.replace(",", "", regex=False)
                )
                tmp[col] = pd.to_numeric(cleaned, errors="coerce").reindex(tmp.index)
                applied.append(col)
            elif action == "convert_percentage":
                cleaned = non_empty.str.replace("%", "", regex=False).str.replace(r"[^\d.\-]", "", regex=True)
                tmp[col] = (pd.to_numeric(cleaned, errors="coerce") / 100).reindex(tmp.index)
                applied.append(col)
            elif action == "convert_units":
                cleaned = non_empty.str.extract(r"([-]?\d+\.?\d*)", expand=False)
                tmp[col] = pd.to_numeric(cleaned, errors="coerce").reindex(tmp.index)
                applied.append(col)
            elif action == "convert_duration":
                tmp[col] = non_empty.apply(_convert_duration_val).reindex(tmp.index)
                applied.append(col)
            elif action == "convert_datetime":
                tmp[col] = pd.to_datetime(tmp[col], errors="coerce")
                applied.append(col)
            elif action == "convert_numeric":
                cleaned = non_empty.str.replace(r"[^\d.\-]", "", regex=True)
                tmp[col] = pd.to_numeric(cleaned, errors="coerce").reindex(tmp.index)
                applied.append(col)
            elif action == "convert_to_boolean":
                tmp[col] = tmp[col].astype(str).str.lower().map(
                    {"true": True, "1": True, "yes": True, "y": True,
                     "false": False, "0": False, "no": False, "n": False}
                )
                applied.append(col)
            elif action == "convert_category":
                tmp[col] = tmp[col].astype("category")
                applied.append(col)
            elif action == "validate_email":
                pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
                tmp[f"{col}_valid_email"] = tmp[col].astype(str).str.strip().str.match(pattern)
                applied.append(col)
        except Exception:
            continue
    return tmp, applied
