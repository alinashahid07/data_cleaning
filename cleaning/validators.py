import re
import numpy as np
import pandas as pd
from .basic import validate_df

# check email format
def validate_email_col(df, col, action="flag"):
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
    is_valid = df_clean[col].astype(str).str.strip().str.match(pattern)
    if action == "flag":
        df_clean[f"{col}_valid_email"] = is_valid
    elif action == "remove":
        df_clean = df_clean[is_valid].reset_index(drop=True)
    return df_clean

# standardize phone numbers
def validate_phone_col(df, col):
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()

    def standardize(val):
        digits = re.sub(r"\D", "", str(val))
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits.startswith("1"):
            return f"+{digits}"
        elif len(digits) >= 7:
            return f"+{digits}"
        return np.nan

    df_clean[col] = df_clean[col].apply(standardize)
    return df_clean

# standardize date formats
def validate_date_col(df, col, output_format="%Y-%m-%d"):
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    fmts = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y",
        "%m-%d-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
        "%d/%m/%y", "%m/%d/%y", "%Y.%m.%d", "%d.%m.%Y",
    ]

    def parse(val):
        if pd.isna(val) or str(val).strip() == "":
            return pd.NaT
        s = str(val).strip()
        for fmt in fmts:
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue
        try:
            return pd.to_datetime(s)
        except Exception:
            return pd.NaT

    parsed = df_clean[col].apply(parse)
    df_clean[col] = parsed.dt.strftime(output_format).where(parsed.notna(), other=None)
    return df_clean

# cap or drop outliers
def cap_outliers(df, col, method="iqr", action="cap", threshold=1.5):
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")
    df_clean = df.copy()
    s = df_clean[col].dropna()
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lower = q1 - threshold * (q3 - q1)
        upper = q3 + threshold * (q3 - q1)
    else:
        lower = s.mean() - threshold * s.std()
        upper = s.mean() + threshold * s.std()
    if action == "cap":
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    else:
        df_clean = df_clean[df_clean[col].isna() | df_clean[col].between(lower, upper)].reset_index(drop=True)
    return df_clean

# flag or drop out of range values
def validate_range(df, col, min_val, max_val, action="flag"):
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")
    df_clean = df.copy()
    in_range = df_clean[col].between(min_val, max_val, inclusive="both") | df_clean[col].isna()
    if action == "flag":
        df_clean[f"{col}_in_range"] = in_range
    else:
        df_clean = df_clean[in_range].reset_index(drop=True)
    return df_clean
