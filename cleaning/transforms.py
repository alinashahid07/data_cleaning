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
