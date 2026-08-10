import numpy as np
import pandas as pd

# validate input
def validate_df(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    # empty df is allowed, operations just return it unchanged

# remove duplicate rows
def remove_duplicate_rows(df):
    validate_df(df)
    return df.drop_duplicates()

# remove duplicate columns
def remove_duplicate_columns(df):
    validate_df(df)
    # drop columns with duplicate names first
    df = df.loc[:, ~df.columns.duplicated()]
    # drop columns with identical content using transpose dedup
    return df.T.drop_duplicates().T

# strip whitespace
def strip_whitespace(df):
    validate_df(df)
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# trim edge characters
def clean_string_edges(df, threshold=0.7, inplace=False):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return None if inplace else df.copy()
    df_clean = df if inplace else df.copy()
    for col in df_clean.select_dtypes(include=["object"]).columns:
        col_series = df_clean[col].astype(str)
        leading = col_series.str.extract(r"^([^\w\s])")[0].dropna()
        trailing = col_series.str.extract(r"([^\w\s])$")[0].dropna()
        keep_leading = (leading.value_counts(normalize=True).iloc[0] > threshold if len(leading) > 0 else False)
        keep_trailing = (trailing.value_counts(normalize=True).iloc[0] > threshold if len(trailing) > 0 else False)
        if not keep_leading:
            df_clean[col] = col_series.str.replace(r"^\W+", "", regex=True)
        if not keep_trailing:
            df_clean[col] = df_clean[col].astype(str).str.replace(r"\W+$", "", regex=True)
    return None if inplace else df_clean

# find and replace text in a column
def find_and_replace(df, col, find, replace, use_regex=False):
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    # only touch non null values so existing nans are not turned into the string nan
    mask = df_clean[col].notna()
    df_clean.loc[mask, col] = df_clean.loc[mask, col].astype(str).str.replace(find, replace, regex=use_regex)
    return df_clean
