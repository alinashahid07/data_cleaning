# imports
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from typing import Optional, Dict, List, Tuple
import io
import openpyxl

# page setup
st.set_page_config(
    page_title="Data Cleaning Pipeline",
    page_icon="*",
    layout="wide"
)

# custom css
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 500;
    }

    /* navbar style tab bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #0e1117;
        padding: 0 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 18px 32px;
        color: #aaaaaa;
        border-radius: 0;
        border: none;
        background: transparent;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background: rgba(255,255,255,0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        background: transparent;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #1f77b4;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: transparent;
    }
    .stTabs {
        margin-top: -1rem;
    }
    </style>
""", unsafe_allow_html=True)

# validate input
def validate_df(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    # empty df is allowed, operations just return it unchanged

# remove duplicate rows
def remove_duplicate_rows(df: pd.DataFrame):
    validate_df(df)
    return df.drop_duplicates()

# remove duplicate columns
def remove_duplicate_columns(df: pd.DataFrame):
    validate_df(df)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.T.drop_duplicates().T

# strip whitespace
def strip_whitespace(df: pd.DataFrame):
    validate_df(df)
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# remove chosen character
def remove_character(df: pd.DataFrame, colname: str, ch_to_remove: str):
    validate_df(df)
    if colname in df.columns:
        df[colname] = df[colname].astype(str).str.replace(ch_to_remove, "", regex=False)
    return df

# trim edge characters
def clean_string_edges(
    df: pd.DataFrame,
    threshold: float = 0.7,
    inplace: bool = False,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Intelligently trims edge characters when conditions are met."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        if verbose:
            st.warning("Warning: Empty DataFrame received")
        return None if inplace else df.copy()

    try:
        df_clean = df if inplace else df.copy()
        cleaned_cols = []

        for col in df_clean.select_dtypes(include=['object']).columns:
            col_series = df_clean[col].astype(str)

            leading_chars = col_series.str.extract(r'^([^\w\s])')[0].dropna()
            trailing_chars = col_series.str.extract(r'([^\w\s])$')[0].dropna()

            keep_leading = (leading_chars.value_counts(normalize=True).iloc[0] > threshold
                           if len(leading_chars) > 0 else False)
            keep_trailing = (trailing_chars.value_counts(normalize=True).iloc[0] > threshold
                             if len(trailing_chars) > 0 else False)

            if not keep_leading:
                if keep_leading is False:
                    df_clean[col] = col_series.str.replace(r'^\W+', '', regex=True)
                    cleaned_cols.append(col)

            if not keep_trailing:
                if keep_trailing is False:
                    df_clean[col] = col_series.str.replace(r'\W+$', '', regex=True)
                    if col not in cleaned_cols:
                        cleaned_cols.append(col)

        if verbose and cleaned_cols:
            st.success(f"Cleaned string edges in {len(cleaned_cols)} columns: {', '.join(cleaned_cols[:5])}")

        return None if inplace else df_clean
    except Exception as e:
        if verbose:
            st.error(f"Error during string cleaning: {str(e)}")
        raise

# convert messy numeric columns
def smart_column_cleaner(
    df: pd.DataFrame,
    conversion_threshold: float = 0.6,
    inplace: bool = False,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Ultimate smart column cleaner for numeric formats."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        if verbose:
            st.warning("Empty DataFrame received")
        return None if inplace else df.copy()

    try:
        df_clean = df if inplace else df.copy()

        # currency patterns
        currency_symbols = r'[$€£¥₹₽₺₩฿₡₦₲₴₵₸₳₻₼₽₾₿]'
        currency_codes = r'(USD|EUR|GBP|JPY|CNY|INR|RUB|AUD|CAD|PKR|BDT|LKR|NPR|SGD|HKD|AED|CHF)'
        currency_text = r'(dollars?|euros?|pounds?|rupees?|yuan|yen|rubles?|pesos?|riyal|ringgit|baht|dinar|lei|krona|forint|złoty)'
        currency_pattern = f'{currency_symbols}|{currency_codes}|{currency_text}'

        conversions = []

        for col in df_clean.select_dtypes(include='object').columns:
            series = df_clean[col].astype(str).str.strip()
            non_empty = series.replace('', np.nan).dropna()

            if non_empty.empty:
                continue

            # currency detection
            currency_like = non_empty.str.contains(r'\d', regex=True) & non_empty.str.contains(currency_pattern, case=False, regex=True)
            if currency_like.mean() > conversion_threshold:
                cleaned = (
                    non_empty.str.replace(r'[^\d.,\-()]', ' ', regex=True)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.replace(r'\((.+?)\)', r'-\1', regex=True)
                    .str.extract(r'([-]?\d[\d\.,]*)', expand=False)
                    .str.replace(',', '', regex=False)
                )

                converted = pd.to_numeric(cleaned, errors='coerce')
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted.reindex(df_clean.index)
                    conversions.append(f"{col} (currency)")
                    continue

            # percentage detection
            if non_empty.str.contains('%').mean() > conversion_threshold:
                cleaned = non_empty.str.replace('%', '', regex=False)
                cleaned = cleaned.str.replace(r'[^\d.\-]', '', regex=True)
                converted = pd.to_numeric(cleaned, errors='coerce') / 100
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted.reindex(df_clean.index)
                    conversions.append(f"{col} (percentage)")
                    continue

            # unit detection
            unit_pattern = r'\d+\s?(kg|g|mg|cm|mm|m|km|ml|l|lb|oz|gal|pt|°C|°F|kWh|cal|ha|ac|sqft|m²|km²)'
            if non_empty.str.contains(unit_pattern, case=False, regex=True).mean() > conversion_threshold:
                cleaned = non_empty.str.extract(r'([-]?\d+\.?\d*)', expand=False)
                converted = pd.to_numeric(cleaned, errors='coerce')
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted.reindex(df_clean.index)
                    conversions.append(f"{col} (unit)")
                    continue

            # duration detection
            duration_pattern = r'(h|hr|hour|min|minute|sec|second|s|m)'
            if non_empty.str.contains(duration_pattern, case=False, regex=True).mean() > conversion_threshold:
                def convert_duration(val):
                    val = str(val).lower()
                    total_seconds = 0
                    parts = re.findall(r'(\d+\.?\d*)\s*(h(?:ou?r)?|m(?:in)?|s(?:ec)?)', val)
                    for num, unit in parts:
                        num = float(num)
                        if unit.startswith('h'):
                            total_seconds += num * 3600
                        elif unit.startswith('m'):
                            total_seconds += num * 60
                        elif unit.startswith('s'):
                            total_seconds += num
                    return total_seconds if total_seconds > 0 else np.nan

                converted = non_empty.apply(convert_duration)
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted.reindex(df_clean.index)
                    conversions.append(f"{col} (duration to seconds)")
                    continue

            # generic numeric detection
            cleaned = non_empty.str.replace(r'[^\d.\-]', '', regex=True)
            converted = pd.to_numeric(cleaned, errors='coerce')
            if converted.notna().mean() > conversion_threshold:
                df_clean[col] = converted.reindex(df_clean.index)
                conversions.append(f"{col} (numeric)")

        if verbose and conversions:
            st.success(f"Converted {len(conversions)} columns:")
            for conv in conversions:
                st.write(f"  - {conv}")

        return None if inplace else df_clean

    except Exception as e:
        if verbose:
            st.error(f"Error during smart cleaning: {str(e)}")
        raise

# fill missing values
def missing_value_handler(
    df: pd.DataFrame,
    threshold: float = 0.3,
    inplace: bool = False,
    numeric_strategy: str = 'auto',
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Enhanced missing value handler with KNN/MICE imputation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    if df.empty:
        if verbose:
            st.warning("Warning: Empty DataFrame received")
        return None if inplace else df.copy()

    try:
        if not inplace:
            df_clean = df.copy()
        else:
            df_clean = df

        # switch to mice for large data
        if numeric_strategy == 'auto':
            if df_clean.shape[1] > 50 or len(df_clean) > 5000:
                numeric_strategy = 'mice'
                if verbose:
                    st.info("Auto-switched to MICE (dataset exceeds 50 columns or 5,000 rows)")

        # convert missing indicators
        missing_indicators = ['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', -999, 999, 9999, '']
        df_clean = df_clean.replace(missing_indicators, np.nan)

        # drop mostly empty columns
        missing_percent = df_clean.isna().mean()
        cols_to_drop = missing_percent[missing_percent > threshold].index
        if len(cols_to_drop) > 0:
            df_clean = df_clean.drop(columns=cols_to_drop)
            if verbose:
                st.warning(f"Dropped {len(cols_to_drop)} columns with more than {threshold*100}% missing values")

        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns

        # numeric imputation
        if not numeric_cols.empty and df_clean[numeric_cols].isna().any().any():
            if numeric_strategy == 'knn' or (numeric_strategy == 'auto' and len(df_clean) <= 5000 and df_clean.shape[1] <= 50):
                if verbose:
                    st.info("Using KNN imputer for numeric columns")
                n_neighbors = max(1, min(5, max(1, len(df_clean) // 1000), len(df_clean)))
                imputer = KNNImputer(n_neighbors=n_neighbors)
            else:
                # MICE needs at least 2 numeric columns to model relationships between them
                if len(numeric_cols) >= 2:
                    if verbose:
                        st.info("Using MICE imputer for numeric columns")
                    imputer = IterativeImputer(max_iter=10, random_state=42)
                else:
                    if verbose:
                        st.info("Not enough numeric columns for MICE, using KNN instead")
                    imputer = KNNImputer(n_neighbors=max(1, min(3, len(df_clean))))

            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

        # categorical imputation
        for col in cat_cols:
            if df_clean[col].isna().any():
                if df_clean[col].nunique() < 0.5 * len(df_clean):
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Missing'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                else:
                    df_clean[col] = df_clean[col].fillna('Missing')

        if verbose:
            st.success(f"Imputed {len(numeric_cols)} numeric and {len(cat_cols)} categorical columns")

        return None if inplace else df_clean

    except Exception as e:
        if verbose:
            st.error(f"Error during missing value handling: {str(e)}")
        raise

# check email format
def validate_email_col(
    df: pd.DataFrame,
    col: str,
    action: str = 'flag'
) -> pd.DataFrame:
    """Validates email format in a column. Flags or removes invalid entries."""
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")

    df_clean = df.copy()
    pattern = r'^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$'
    is_valid = df_clean[col].astype(str).str.strip().str.match(pattern)

    if action == 'flag':
        df_clean[f'{col}_valid_email'] = is_valid
    elif action == 'remove':
        df_clean = df_clean[is_valid].reset_index(drop=True)

    return df_clean

# standardize phone numbers
def validate_phone_col(
    df: pd.DataFrame,
    col: str,
    output_format: str = '+1XXXXXXXXXX'
) -> pd.DataFrame:
    """Strips all non-digit characters and standardizes phone numbers."""
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")

    df_clean = df.copy()

    def standardize_phone(val):
        digits = re.sub(r'\D', '', str(val))
        if len(digits) == 10:
            return f'+1{digits}'
        elif len(digits) == 11 and digits.startswith('1'):
            return f'+{digits}'
        elif len(digits) >= 7:
            return f'+{digits}'
        return np.nan

    df_clean[col] = df_clean[col].apply(standardize_phone)
    return df_clean

# standardize date formats
def validate_date_col(
    df: pd.DataFrame,
    col: str,
    output_format: str = '%Y-%m-%d'
) -> pd.DataFrame:
    """Detects and parses mixed date formats into one consistent format."""
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")

    df_clean = df.copy()

    formats_to_try = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%m-%d-%Y',
        '%d %b %Y',
        '%d %B %Y',
        '%b %d %Y',
        '%B %d %Y',
        '%b %d, %Y',
        '%B %d, %Y',
        '%d/%m/%y',
        '%m/%d/%y',
        '%d-%m-%y',
        '%Y.%m.%d',
        '%d.%m.%Y',
    ]

    def parse_single(val):
        if pd.isna(val) or str(val).strip() == '':
            return pd.NaT
        val_str = str(val).strip()
        for fmt in formats_to_try:
            try:
                return pd.to_datetime(val_str, format=fmt)
            except (ValueError, TypeError):
                continue
        try:
            return pd.to_datetime(val_str)
        except (ValueError, TypeError):
            return pd.NaT

    parsed = df_clean[col].apply(parse_single)
    df_clean[col] = parsed.dt.strftime(output_format).where(parsed.notna(), other=np.nan)
    return df_clean

# cap or drop outliers
def cap_outliers(
    df: pd.DataFrame,
    col: str,
    method: str = 'iqr',
    action: str = 'cap',
    threshold: float = 1.5
) -> pd.DataFrame:
    """Detects outliers via IQR or Z-score and caps or removes them."""
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")

    df_clean = df.copy()
    series = df_clean[col].dropna()

    if method == 'iqr':
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
    else:
        mean, std = series.mean(), series.std()
        lower = mean - threshold * std
        upper = mean + threshold * std

    if action == 'cap':
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    elif action == 'remove':
        mask = df_clean[col].isna() | ((df_clean[col] >= lower) & (df_clean[col] <= upper))
        df_clean = df_clean[mask].reset_index(drop=True)

    return df_clean

# flag or drop out of range values
def validate_range(
    df: pd.DataFrame,
    col: str,
    min_val: float,
    max_val: float,
    action: str = 'flag'
) -> pd.DataFrame:
    """Flags or removes rows where column value is outside [min_val, max_val]."""
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")

    df_clean = df.copy()
    in_range = df_clean[col].between(min_val, max_val, inclusive='both') | df_clean[col].isna()

    if action == 'flag':
        df_clean[f'{col}_in_range'] = in_range
    elif action == 'remove':
        df_clean = df_clean[in_range].reset_index(drop=True)

    return df_clean

# find and replace text in a column
def find_and_replace(df: pd.DataFrame, col: str, find: str, replace: str, use_regex: bool = False) -> pd.DataFrame:
    """Find and replace values in a column, with optional regex support."""
    validate_df(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    # only touch non-null values so existing NaNs are not turned into the string nan
    mask = df_clean[col].notna()
    df_clean.loc[mask, col] = df_clean.loc[mask, col].astype(str).str.replace(find, replace, regex=use_regex)
    return df_clean

# snapshot before an operation
def snapshot():
    """Takes a snapshot of current_df. Pass the result to commit_history() once the operation succeeds."""
    return st.session_state.current_df.copy()

# commit a snapshot after success
def commit_history(label, snap):
    """Commits a previously taken snapshot to history. Only call this after the operation succeeds."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if len(st.session_state.history) >= 20:
        st.session_state.history.pop(0)
    st.session_state.history.append({'label': label, 'df': snap})
    # bumps a counter used as a cheap cache key for the excel export
    st.session_state['history_len'] = st.session_state.get('history_len', 0) + 1

# restore previous state
def undo_last():
    """Restore the previous df state from history."""
    if st.session_state.get('history'):
        last = st.session_state.history.pop()
        st.session_state.current_df = last['df']
        st.session_state['history_len'] = st.session_state.get('history_len', 0) + 1
        return last['label']
    return None

# turn history into a runnable script
def build_pipeline_script(history):
    """Builds a standalone python script that replays the recorded cleaning steps."""
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "import re",
        "from sklearn.impute import KNNImputer",
        "",
        "# load your file here",
        "df = pd.read_csv('your_file.csv')",
        "",
    ]

    for step in history:
        label = step['label']
        lines.append(f"# {label}")

        if label == "Strip Whitespace":
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")

        elif label == "Drop Duplicate Rows":
            lines.append("df = df.drop_duplicates().reset_index(drop=True)")

        elif label == "Drop Duplicate Columns":
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")
            lines.append("df = df.T.drop_duplicates().T")

        elif label == "Clean String Edges":
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    df[col] = df[col].astype(str).str.replace(r'^\\W+', '', regex=True).str.replace(r'\\W+$', '', regex=True)")

        elif label == "Smart Column Cleaner":
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    cleaned = df[col].str.replace(r'[^\\d.\\-]', '', regex=True)")
            lines.append("    converted = pd.to_numeric(cleaned, errors='coerce')")
            lines.append("    if converted.notna().mean() > 0.6:")
            lines.append("        df[col] = converted")

        elif label == "Handle Missing Values":
            lines.append("df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan, inplace=True)")
            lines.append("num_cols = df.select_dtypes(include=np.number).columns")
            lines.append("if not num_cols.empty and df[num_cols].isna().any().any():")
            lines.append("    df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])")
            lines.append("for col in df.select_dtypes(exclude=np.number).columns:")
            lines.append("    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Missing')")

        elif label == "Full Pipeline" or label == "Auto-Fix All":
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")
            lines.append("df = df.drop_duplicates().reset_index(drop=True)")
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")
            lines.append("df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan, inplace=True)")
            lines.append("num_cols = df.select_dtypes(include=np.number).columns")
            lines.append("if not num_cols.empty and df[num_cols].isna().any().any():")
            lines.append("    df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])")

        elif label.startswith("Fix: strip_whitespace"):
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")

        elif label.startswith("Fix: convert_currency"):
            lines.append("# currency conversion, update column names as needed")
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    cleaned = df[col].str.replace(r'[^\\d.,\\-()]', ' ', regex=True).str.extract(r'([-]?\\d[\\d\\.,]*)', expand=False).str.replace(',', '', regex=False)")
            lines.append("    converted = pd.to_numeric(cleaned, errors='coerce')")
            lines.append("    if converted.notna().mean() > 0.6: df[col] = converted")

        elif label.startswith("Fix: convert_percentage"):
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    if df[col].str.contains('%').mean() > 0.6:")
            lines.append("        df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False), errors='coerce') / 100")

        elif label.startswith("Fix: convert_units"):
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    cleaned = df[col].str.extract(r'([-]?\\d+\\.?\\d*)', expand=False)")
            lines.append("    converted = pd.to_numeric(cleaned, errors='coerce')")
            lines.append("    if converted.notna().mean() > 0.6: df[col] = converted")

        elif label.startswith("Fix: handle_missing"):
            lines.append("df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan, inplace=True)")
            lines.append("num_cols = df.select_dtypes(include=np.number).columns")
            lines.append("if not num_cols.empty:")
            lines.append("    df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])")

        elif label.startswith("Fix: clean_edges"):
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    df[col] = df[col].astype(str).str.replace(r'^\\W+', '', regex=True).str.replace(r'\\W+$', '', regex=True)")

        elif label.startswith("Fix: drop_duplicates"):
            lines.append("df = df.drop_duplicates().reset_index(drop=True)")

        elif label.startswith("Fix: drop_dup_cols"):
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")

        elif label.startswith("Find & Replace in"):
            col = label.replace("Find & Replace in ", "").strip()
            lines.append(f"df['{col}'] = df['{col}'].astype(str).str.replace('FIND', 'REPLACE', regex=False)  # update FIND and REPLACE")

        elif label.startswith("Type Override:"):
            parts = label.replace("Type Override: ", "").split(" -> ")
            if len(parts) == 2:
                col, dtype = parts[0].strip(), parts[1].strip()
                if "int" in dtype:
                    lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce').astype('Int64')")
                elif "float" in dtype:
                    lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')")
                elif "datetime" in dtype:
                    lines.append(f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
                elif "bool" in dtype:
                    lines.append(f"df['{col}'] = df['{col}'].astype(str).str.lower().map({{'true':True,'1':True,'yes':True,'false':False,'0':False,'no':False}})")
                elif "category" in dtype:
                    lines.append(f"df['{col}'] = df['{col}'].astype('category')")
                else:
                    lines.append(f"df['{col}'] = df['{col}'].astype(str)")

        elif label == "Validate Email":
            lines.append("pattern = r'^[\\w\\.\\+\\-]+@[\\w\\-]+\\.[a-zA-Z]{2,}$'")
            lines.append("# to flag: df['email_valid'] = df['email_col'].astype(str).str.match(pattern)")
            lines.append("# to remove: df = df[df['email_col'].astype(str).str.match(pattern)]")

        elif label == "Standardize Phone":
            lines.append("def standardize_phone(val):")
            lines.append("    digits = re.sub(r'\\D', '', str(val))")
            lines.append("    if len(digits) == 10: return f'+1{digits}'")
            lines.append("    elif len(digits) >= 7: return f'+{digits}'")
            lines.append("    return np.nan")
            lines.append("# df['phone_col'] = df['phone_col'].apply(standardize_phone)  # update col name")

        elif label == "Standardize Dates":
            lines.append("# df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce').dt.strftime('%Y-%m-%d')  # update col name and format")

        elif label == "Cap Outliers":
            lines.append("# iqr outlier capping, update col name")
            lines.append("# col = 'your_column'")
            lines.append("# q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)")
            lines.append("# df[col] = df[col].clip(lower=q1 - 1.5*(q3-q1), upper=q3 + 1.5*(q3-q1))")

        elif label == "Validate Range":
            lines.append("# range validation, update col name and bounds")
            lines.append("# df = df[df['your_column'].between(0, 100)]")

        else:
            lines.append("# manual step, no code generated")

        lines.append("")

    lines += ["print('Pipeline complete. Shape:', df.shape)"]
    return "\n".join(lines)

# scan dataframe for issues
def analyze_data_issues(df: pd.DataFrame, conversion_threshold: float = 0.6) -> Dict[str, any]:
    """Analyzes the dataframe and identifies potential issues."""
    issues = {
        'duplicate_rows': 0,
        'duplicate_cols': 0,
        'whitespace_cols': [],
        'currency_cols': [],
        'percentage_cols': [],
        'unit_cols': [],
        'duration_cols': [],
        'missing_cells': 0,
        'missing_cols': [],
        'edge_char_cols': []
    }

    issues['duplicate_rows'] = df.duplicated().sum()
    issues['duplicate_cols'] = len(df.columns) - len(df.T.drop_duplicates().T.columns)

    for col in df.select_dtypes(include='object').columns:
        if df[col].astype(str).str.strip().ne(df[col].astype(str)).any():
            issues['whitespace_cols'].append(col)

    currency_symbols = r'[$€£¥₹₽₺₩฿₡₦₲₴₵₸₳₻₼₽₾₿]'
    currency_codes = r'(USD|EUR|GBP|JPY|CNY|INR|RUB|AUD|CAD|PKR)'
    currency_pattern = f'{currency_symbols}|{currency_codes}'

    for col in df.select_dtypes(include='object').columns:
        series = df[col].astype(str).str.strip()
        non_empty = series.replace('', np.nan).dropna()

        if non_empty.empty:
            continue

        currency_like = non_empty.str.contains(r'\d', regex=True) & non_empty.str.contains(currency_pattern, case=False, regex=True)
        if currency_like.mean() > conversion_threshold:
            issues['currency_cols'].append(col)
            continue

        if non_empty.str.contains('%').mean() > conversion_threshold:
            issues['percentage_cols'].append(col)
            continue

        unit_pattern = r'\d+\s?(kg|g|mg|cm|mm|m|km|ml|l|lb|oz)'
        if non_empty.str.contains(unit_pattern, case=False, regex=True).mean() > conversion_threshold:
            issues['unit_cols'].append(col)
            continue

        duration_pattern = r'(h|hr|hour|min|minute|sec|second)'
        if non_empty.str.contains(duration_pattern, case=False, regex=True).mean() > conversion_threshold:
            issues['duration_cols'].append(col)

    issues['missing_cells'] = df.isna().sum().sum()
    for col in df.columns:
        missing_pct = df[col].isna().mean()
        if missing_pct > 0:
            issues['missing_cols'].append((col, missing_pct))

    for col in df.select_dtypes(include='object').columns:
        col_series = df[col].astype(str)
        has_edge_chars = col_series.str.match(r'^\W') | col_series.str.match(r'\W$')
        if has_edge_chars.sum() > 0:
            issues['edge_char_cols'].append(col)

    return issues

# turn issues into suggestions
def generate_recommendations(issues: Dict[str, any]) -> List[Tuple[str, str, str, str]]:
    """Generates actionable recommendations based on detected issues.
    Returns list of (icon, title, description, action_key) tuples."""
    recommendations = []

    if issues['duplicate_rows'] > 0:
        recommendations.append((
            "",
            f"Found {issues['duplicate_rows']} duplicate rows",
            "Removing duplicates will reduce your dataset size and prevent skewed analysis.",
            "drop_duplicates"
        ))

    if issues['duplicate_cols'] > 0:
        recommendations.append((
            "",
            f"Found {issues['duplicate_cols']} duplicate columns",
            "These columns are wasting memory and processing power.",
            "drop_dup_cols"
        ))

    if len(issues['whitespace_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['whitespace_cols'])} columns have extra whitespace",
            f"Columns: {', '.join(issues['whitespace_cols'][:3])}{'...' if len(issues['whitespace_cols']) > 3 else ''}",
            "strip_whitespace"
        ))

    if len(issues['currency_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['currency_cols'])} columns look like currency but aren't numeric",
            f"Columns: {', '.join(issues['currency_cols'][:3])}{'...' if len(issues['currency_cols']) > 3 else ''}. Convert them to do calculations.",
            "convert_currency"
        ))

    if len(issues['percentage_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['percentage_cols'])} columns contain percentages as text",
            f"Columns: {', '.join(issues['percentage_cols'][:3])}{'...' if len(issues['percentage_cols']) > 3 else ''}. Should be decimals for math.",
            "convert_percentage"
        ))

    if len(issues['unit_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['unit_cols'])} columns have measurement units mixed with numbers",
            f"Columns: {', '.join(issues['unit_cols'][:3])}{'...' if len(issues['unit_cols']) > 3 else ''}",
            "convert_units"
        ))

    if len(issues['duration_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['duration_cols'])} columns contain time durations",
            f"Columns: {', '.join(issues['duration_cols'][:3])}{'...' if len(issues['duration_cols']) > 3 else ''}. Convert to seconds for consistency.",
            "convert_duration"
        ))

    if issues['missing_cells'] > 0:
        top_missing = sorted(issues['missing_cols'], key=lambda x: x[1], reverse=True)[:3]
        col_details = ', '.join([f"{col} ({pct*100:.0f}%)" for col, pct in top_missing])
        recommendations.append((
            "",
            f"{issues['missing_cells']} missing values found",
            f"Worst affected: {col_details}. Use ML imputation to fill them intelligently.",
            "handle_missing"
        ))

    if len(issues['edge_char_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['edge_char_cols'])} columns have unwanted edge characters",
            f"Columns: {', '.join(issues['edge_char_cols'][:3])}{'...' if len(issues['edge_char_cols']) > 3 else ''}",
            "clean_edges"
        ))

    return recommendations

# read uploaded file, cached
@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, filename: str, file_id: str, sheet_name=None) -> pd.DataFrame:
    """Caches file parsing so it does not re-read on every interaction."""
    ext = filename.split('.')[-1].lower()
    buf = io.BytesIO(file_bytes)
    if ext == 'csv':
        return pd.read_csv(buf, quotechar='"', skipinitialspace=True)
    else:
        return pd.read_excel(buf, sheet_name=sheet_name)

# get stats summary
@st.cache_data(show_spinner=False)
def get_dataframe_stats(df: pd.DataFrame) -> Dict:
    """Returns key statistics about the DataFrame, cached until df changes."""
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_cells': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_cols': len(df.select_dtypes(include=np.number).columns),
        'categorical_cols': len(df.select_dtypes(exclude=np.number).columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }

# per column stats for profiler
@st.cache_data(show_spinner=False)
def get_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column stats for the profiling panel."""
    rows = []
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        row = {
            'Column': col,
            'Type': str(series.dtype),
            'Non-Null': int(series.notna().sum()),
            'Null': int(series.isna().sum()),
            'Null %': f"{series.isna().mean()*100:.1f}%",
            'Unique': int(series.nunique()),
        }
        if is_numeric:
            row['Min'] = round(float(series.min()), 4) if series.notna().any() else None
            row['Max'] = round(float(series.max()), 4) if series.notna().any() else None
            row['Mean'] = round(float(series.mean()), 4) if series.notna().any() else None
            row['Median'] = round(float(series.median()), 4) if series.notna().any() else None
            row['Std'] = round(float(series.std()), 4) if series.notna().any() else None
            row['Skew'] = round(float(series.skew()), 4) if series.notna().any() else None
            row['Sample Values'] = ', '.join(str(v) for v in series.dropna().head(3).tolist())
        else:
            row['Min'] = row['Max'] = row['Mean'] = row['Median'] = row['Std'] = row['Skew'] = '-'
            top = series.dropna().value_counts().head(3)
            row['Sample Values'] = ', '.join(f'"{v}"' for v in top.index.tolist())
        rows.append(row)
    return pd.DataFrame(rows)
@st.cache_data(show_spinner=False)
def get_analysis_and_recommendations(
    df: pd.DataFrame, conversion_threshold: float
) -> Tuple[Dict, List]:
    """Single cached call that replaces separate per-render analysis calls."""
    issues = analyze_data_issues(df, conversion_threshold)
    recommendations = generate_recommendations(issues)
    return issues, recommendations

# sidebar settings
with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Mode", ["Simple", "Advanced"], horizontal=True, key="mode_radio",
        help="Simple uses sensible defaults so you can start cleaning right away. Advanced lets you tune the thresholds manually."
    )

    if mode == "Simple":
        st.caption("Using default settings. Switch to Advanced to customize.")
        missing_threshold = 0.30
        numeric_strategy = "auto"
        conversion_threshold = 0.60
    else:
        st.subheader("Missing Value Handler")
        missing_threshold = st.slider(
            "Drop columns with missing % >",
            0, 100,
            value=st.session_state.get("missing_threshold_val", 30),
            help="Columns where more than this percent of values are missing get dropped entirely. Raise it to keep sparser columns, lower it to be stricter."
        ) / 100
        st.session_state["missing_threshold_val"] = int(missing_threshold * 100)

        numeric_strategy = st.selectbox(
            "Numeric Imputation Strategy",
            ['auto', 'knn', 'mice'],
            key="numeric_strategy_select",
            help="auto picks KNN for small files and MICE for large ones. KNN fills gaps using nearby similar rows, fast and good for most cases. MICE models each column iteratively, more accurate but slower, best for large datasets with lots of missing data."
        )

        st.subheader("Smart Cleaner")
        conversion_threshold = st.slider(
            "Conversion Threshold %",
            0, 100,
            value=st.session_state.get("conversion_threshold_val", 60),
            help="How many values in a column must match a pattern before the whole column gets converted. At 60%, a column converts if 60% of its values look like currency. Lower to convert more aggressively, raise to be more conservative."
        ) / 100
        st.session_state["conversion_threshold_val"] = int(conversion_threshold * 100)

        st.divider()

        if st.button("Reset All", key="reset_all_btn", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != "mode_radio":
                    del st.session_state[key]
            st.session_state["missing_threshold_val"] = 30
            st.session_state["conversion_threshold_val"] = 60
            st.cache_data.clear()
            st.rerun()

# tabs sit at the top, navbar style
tab_upload, tab_overview, tab_recommend, tab_clean, tab_validate, tab_profile, tab_history = st.tabs([
    "  Upload  ", "  Overview  ", "  Recommendations  ",
    "  Clean  ", "  Validate  ", "  Profile  ", "  History & Export  ",
])

# upload tab, file uploader lives here
with tab_upload:
    st.subheader("Advanced Data Cleaning Pipeline")
    st.caption("Upload a CSV or Excel file to get started. [Test CSV on GitHub](https://github.com/Aneezakiran07/Data-Pipelining)")
    st.write("")
    uploaded_file = st.file_uploader("", type=['csv', 'xlsx', 'xls'], key="uploader", label_visibility="collapsed")
    if uploaded_file is None:
        st.write("")
        st.subheader("Sample Data Format")
        st.dataframe(pd.DataFrame({
            'name': ['  Alice  ', 'Bob', 'Charlie', 'Alice'],
            'price': ['$100', '$200.50', '€300', '$100'],
            'percentage': ['75%', '80.5%', '99%', '75%'],
            'weight': ['100kg', '150.5 lbs', '?', '100kg'],
            'duration': ['1h30m', '90min', 'NA', '1h30m']
        }), use_container_width=True)
        st.caption("The pipeline can handle currency, percentages, units, and missing values automatically!")
    else:
        st.success(f"{uploaded_file.name} is loaded. Navigate to any tab to start cleaning.")

# read upload from session state so other tabs can access it
uploaded_file = st.session_state.get("uploader")

# everything below only runs once a file is uploaded
if uploaded_file is None:
    with tab_overview:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_recommend:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_clean:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_validate:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_profile:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_history:
        st.info("Upload a file in the Upload tab to get started.")
else:
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        selected_sheet = None
        uploaded_file.seek(0)
        # streamlit's internal file_id changes on every upload, even for the same file
        file_id = uploaded_file.file_id

        # detect new upload before loading, wipe state and rerun clean
        if st.session_state.get('loaded_file_id') != file_id:
            st.cache_data.clear()
            keys_to_clear = [k for k in st.session_state.keys()
                             if k not in ("uploader", "mode_radio", "sheet_selector")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.session_state["loaded_file_id"] = file_id
            st.session_state["missing_threshold_val"] = 30
            st.session_state["conversion_threshold_val"] = 60
            st.rerun()

        # sheet picker for excel, before the cached load
        if file_extension in ['xlsx', 'xls']:
            xl_bytes = uploaded_file.read()
            xl = pd.ExcelFile(io.BytesIO(xl_bytes))
            sheet_names = xl.sheet_names
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox("Select sheet:", sheet_names, key="sheet_selector")
            else:
                selected_sheet = sheet_names[0]
            file_bytes = xl_bytes
        else:
            file_bytes = uploaded_file.read()

        # include selected_sheet in the load key so switching sheets busts the cache
        load_key = f"{file_id}_{selected_sheet}"
        df = load_file(file_bytes, uploaded_file.name, load_key, sheet_name=selected_sheet)

        # reset state when sheet changes or on first load
        state_key = f"state_{load_key}"
        if st.session_state.get("state_key_id") != state_key:
            # clear stale checkbox widget keys from the previous sheet, avoids ghost selections
            stale_keys = [k for k in st.session_state.keys()
                          if k.startswith(("_vc_", "_va_", "_rc_", "_ra_", "_widget_all_", "_widget_chk_"))]
            for k in stale_keys:
                del st.session_state[k]
            st.session_state.update({
                "original_df": df.copy(),
                "current_df": df.copy(),
                "original_stats": get_dataframe_stats(df),
                "selected_columns": {},
                "val_selected": {},
                "last_success_msg": None,
                "history": [],
                "state_key_id": state_key,
            })

        if "val_selected" not in st.session_state:
            st.session_state.val_selected = {}

        current_stats = get_dataframe_stats(st.session_state.current_df)
        original_stats = st.session_state.original_stats
        all_cols = list(st.session_state.current_df.columns)
        text_cols = list(st.session_state.current_df.select_dtypes(include='object').columns)
        num_cols = list(st.session_state.current_df.select_dtypes(include=np.number).columns)

        st.info(f"{uploaded_file.name}  |  {current_stats['rows']} rows x {current_stats['columns']} columns  |  "
               f"{current_stats['memory_usage']:.2f} MB")

        # popover helper for validation column pickers, defined here so val_selected exists
        def make_val_all_handler(section, cols):
            def handler():
                if st.session_state.get(f"_vall_{section}", False):
                    st.session_state.val_selected[section] = cols.copy()
                else:
                    st.session_state.val_selected[section] = []
            return handler

        def make_val_col_handler(section, col):
            def handler():
                sel = st.session_state.val_selected.get(section, [])
                if st.session_state.get(f"_valc_{section}_{col}", False):
                    if col not in sel:
                        st.session_state.val_selected[section] = sel + [col]
                else:
                    st.session_state.val_selected[section] = [c for c in sel if c != col]
            return handler

        def col_popover(section, available_cols):
            """Renders the column selector popover and returns selected count."""
            n = len(st.session_state.val_selected.get(section, []))
            label = f"{n} selected" if n > 0 else "Select columns"
            with st.popover(label, use_container_width=True):
                st.caption("Select columns to apply this operation")
                st.checkbox(
                    "Apply to all",
                    key=f"_vall_{section}",
                    on_change=make_val_all_handler(section, available_cols)
                )
                for col in available_cols:
                    st.checkbox(
                        col,
                        key=f"_valc_{section}_{col}",
                        on_change=make_val_col_handler(section, col)
                    )
            return n

        # shows and immediately clears any pending success message
        def show_msg():
            if st.session_state.get('last_success_msg'):
                st.success(st.session_state.last_success_msg)
                st.session_state.last_success_msg = None

        # overview tab
        with tab_overview:
            st.subheader("Data Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", current_stats['rows'],
                         delta=current_stats['rows'] - original_stats['rows'],
                         delta_color="inverse")
                st.metric("Columns", current_stats['columns'],
                         delta=current_stats['columns'] - original_stats['columns'],
                         delta_color="inverse")
            with col2:
                st.metric("Missing Cells", current_stats['missing_cells'],
                         delta=current_stats['missing_cells'] - original_stats['missing_cells'],
                         delta_color="inverse")
                st.metric("Duplicate Rows", current_stats['duplicate_rows'],
                         delta=current_stats['duplicate_rows'] - original_stats['duplicate_rows'],
                         delta_color="inverse")
            with col3:
                st.metric("Numeric Columns", current_stats['numeric_cols'])
                st.metric("Categorical Columns", current_stats['categorical_cols'])

            st.divider()

            # data preview
            st.subheader("Data Preview")
            total_rows = len(st.session_state.current_df)
            max_preview = min(50, total_rows)
            default_preview = min(10, total_rows)
            preview_rows = st.slider("Rows to display", min(5, total_rows), max_preview, default_preview, key="preview_rows_slider")
            if total_rows < 50:
                st.caption(f"File has {total_rows} rows, slider capped at {total_rows}.")
            st.dataframe(st.session_state.current_df.head(preview_rows), use_container_width=True)

            st.divider()

            # column info
            with st.expander("Column Data Types & Information", expanded=False):
                col_types = pd.DataFrame({
                    'Column': st.session_state.current_df.columns,
                    'Type': st.session_state.current_df.dtypes.values,
                    'Non-Null': st.session_state.current_df.count().values,
                    'Null': st.session_state.current_df.isna().sum().values,
                    'Unique': st.session_state.current_df.nunique().values
                })
                st.dataframe(col_types, use_container_width=True)

            st.caption("Download and reset options are in the **History & Export** tab.")

        # profile tab
        with tab_profile:
            show_msg()

            # column profiler
            st.subheader("Column Profiler")
            st.caption("Per-column stats: min, max, mean, median, std, skewness, and sample values.")
            profile_df = get_column_profile(st.session_state.current_df)
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
            worst = profile_df[profile_df['Null'] > 0].sort_values('Null', ascending=False)
            if not worst.empty:
                st.caption(
                    f"{len(worst)} column(s) have missing values. "
                    f"Worst: **{worst.iloc[0]['Column']}** ({worst.iloc[0]['Null %']} missing)"
                )

            st.divider()

            # before and after comparison
            st.subheader("Before / After Comparison")
            shared_cols = [c for c in st.session_state.original_df.columns if c in st.session_state.current_df.columns]
            if shared_cols:
                ba_col = st.selectbox("Select column to compare", shared_cols, key="ba_col")
                ba_n = st.slider("Rows to preview", 5, 50, 10, key="ba_n")
                orig_series = st.session_state.original_df[ba_col].head(ba_n).reset_index(drop=True)
                curr_series = st.session_state.current_df[ba_col].head(ba_n).reset_index(drop=True)
                min_len = min(len(orig_series), len(curr_series))
                orig_series, curr_series = orig_series.iloc[:min_len], curr_series.iloc[:min_len]
                changed = orig_series.fillna("").astype(str) != curr_series.fillna("").astype(str)
                st.dataframe(pd.DataFrame({
                    "Original": orig_series,
                    "Current": curr_series,
                    "Changed": changed.map({True: "yes", False: ""}),
                }), use_container_width=True)
                n_changed = int(changed.sum())
                st.caption(f"{n_changed} row(s) changed in this preview window." if n_changed else "No differences found in this preview window.")

        # recommendations tab
        with tab_recommend:
            show_msg()

            st.subheader("Smart Recommendations")

            issues, recommendations = get_analysis_and_recommendations(
                st.session_state.current_df, conversion_threshold
            )

            if len(recommendations) == 0:
                st.success("Your data looks clean! No issues detected.")
            else:
                st.warning(f"Found {len(recommendations)} potential issues in your data")

                if 'selected_columns' not in st.session_state:
                    st.session_state.selected_columns = {}

                for idx, (icon, title, description, action_key) in enumerate(recommendations):

                    affected_columns = []
                    if action_key == "strip_whitespace":
                        affected_columns = issues['whitespace_cols']
                    elif action_key == "convert_currency":
                        affected_columns = issues['currency_cols']
                    elif action_key == "convert_percentage":
                        affected_columns = issues['percentage_cols']
                    elif action_key == "convert_units":
                        affected_columns = issues['unit_cols']
                    elif action_key == "convert_duration":
                        affected_columns = issues['duration_cols']
                    elif action_key == "clean_edges":
                        affected_columns = issues['edge_char_cols']
                    elif action_key == "handle_missing":
                        affected_columns = [col for col, pct in issues['missing_cols']]

                    num_selected = len(st.session_state.selected_columns.get(action_key, []))

                    if affected_columns:
                        dropdown_label = f"{num_selected} selected" if num_selected > 0 else "Select columns"

                        col1, col2, col3 = st.columns([5, 1.4, 1])

                        with col1:
                            st.write(f"**{title}**")
                            st.caption(description)

                        with col2:
                            with st.popover(dropdown_label, use_container_width=True):
                                st.write(f"**Columns for: {title}**")
                                st.caption("Check the columns you want to fix")

                                def make_apply_all_handler(ak, cols):
                                    def handler():
                                        if st.session_state.get(f"_widget_all_{ak}", False):
                                            st.session_state.selected_columns[ak] = cols.copy()
                                        else:
                                            st.session_state.selected_columns[ak] = []
                                    return handler

                                def make_col_handler(ak, col):
                                    def handler():
                                        sel = st.session_state.selected_columns.get(ak, [])
                                        if st.session_state.get(f"_widget_chk_{ak}_{col}", False):
                                            if col not in sel:
                                                st.session_state.selected_columns[ak] = sel + [col]
                                        else:
                                            st.session_state.selected_columns[ak] = [c for c in sel if c != col]
                                    return handler

                                st.checkbox(
                                    "Apply to all columns",
                                    key=f"_widget_all_{action_key}",
                                    on_change=make_apply_all_handler(action_key, affected_columns)
                                )

                                for col in affected_columns:
                                    st.checkbox(
                                        col,
                                        key=f"_widget_chk_{action_key}_{col}",
                                        on_change=make_col_handler(action_key, col)
                                    )

                        with col3:
                            has_selection = num_selected > 0
                            if st.button(
                                "Fix This",
                                key=f"fix_{action_key}",
                                use_container_width=True,
                                disabled=not has_selection,
                                type="primary" if has_selection else "secondary"
                            ):
                                selected_cols = st.session_state.selected_columns.get(action_key, [])
                                try:
                                    with st.spinner(f"Fixing {len(selected_cols)} column(s)..."):
                                        _snap = snapshot()
                                        df_temp = st.session_state.current_df.copy()

                                        if action_key == "strip_whitespace":
                                            for col in selected_cols:
                                                if col in df_temp.columns and df_temp[col].dtype == 'object':
                                                    df_temp[col] = df_temp[col].str.strip()

                                        elif action_key in ["convert_currency", "convert_percentage", "convert_units", "convert_duration"]:
                                            for col in selected_cols:
                                                if col not in df_temp.columns:
                                                    continue
                                                series = df_temp[col].astype(str).str.strip()
                                                non_empty = series.replace('', np.nan).dropna()
                                                if non_empty.empty:
                                                    continue

                                                if action_key == "convert_currency":
                                                    cleaned = (
                                                        non_empty.str.replace(r'[^\d.,\-()]', ' ', regex=True)
                                                        .str.replace(r'\s+', ' ', regex=True)
                                                        .str.replace(r'\((.+?)\)', r'-\1', regex=True)
                                                        .str.extract(r'([-]?\d[\d\.,]*)', expand=False)
                                                        .str.replace(',', '', regex=False)
                                                    )
                                                    df_temp[col] = pd.to_numeric(cleaned, errors='coerce')

                                                elif action_key == "convert_percentage":
                                                    cleaned = non_empty.str.replace('%', '', regex=False).str.replace(r'[^\d.\-]', '', regex=True)
                                                    df_temp[col] = pd.to_numeric(cleaned, errors='coerce') / 100

                                                elif action_key == "convert_units":
                                                    cleaned = non_empty.str.extract(r'([-]?\d+\.?\d*)', expand=False)
                                                    df_temp[col] = pd.to_numeric(cleaned, errors='coerce')

                                                elif action_key == "convert_duration":
                                                    def convert_dur(val):
                                                        val = str(val).lower()
                                                        total = 0
                                                        for num, unit in re.findall(r'(\d+\.?\d*)\s*(h(?:ou?r)?|m(?:in)?|s(?:ec)?)', val):
                                                            num = float(num)
                                                            if unit.startswith('h'): total += num * 3600
                                                            elif unit.startswith('m'): total += num * 60
                                                            elif unit.startswith('s'): total += num
                                                        return total if total > 0 else np.nan
                                                    df_temp[col] = non_empty.apply(convert_dur)

                                        elif action_key == "clean_edges":
                                            for col in selected_cols:
                                                if col in df_temp.columns and df_temp[col].dtype == 'object':
                                                    df_temp[col] = df_temp[col].astype(str).str.replace(r'^\W+', '', regex=True).str.replace(r'\W+$', '', regex=True)

                                        elif action_key == "handle_missing":
                                            valid_cols = [c for c in selected_cols if c in df_temp.columns]
                                            if valid_cols:
                                                df_subset = df_temp[valid_cols].copy()
                                                df_subset = missing_value_handler(df_subset, threshold=missing_threshold, numeric_strategy=numeric_strategy, verbose=False)
                                                for c in valid_cols:
                                                    if c in df_subset.columns:
                                                        df_temp[c] = df_subset[c]

                                        st.session_state.current_df = df_temp
                                        commit_history(f"Fix: {action_key}", _snap)
                                        st.session_state.selected_columns.pop(action_key, None)
                                        st.session_state.last_success_msg = f"Fixed {action_key} on {len(selected_cols)} column(s)!"
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                    else:
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.write(f"**{title}**")
                            st.caption(description)
                        with col2:
                            if st.button("Fix This", key=f"fix_{action_key}", use_container_width=True, type="primary"):
                                try:
                                    with st.spinner("Fixing..."):
                                        _snap = snapshot()
                                        if action_key == "drop_duplicates":
                                            st.session_state.current_df = remove_duplicate_rows(st.session_state.current_df)
                                            commit_history(f"Fix: {action_key}", _snap)
                                            st.session_state.last_success_msg = "Duplicate rows removed!"
                                        elif action_key == "drop_dup_cols":
                                            st.session_state.current_df = remove_duplicate_columns(st.session_state.current_df)
                                            commit_history(f"Fix: {action_key}", _snap)
                                            st.session_state.last_success_msg = "Duplicate columns removed!"
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                    st.write("")

                if st.button("Auto-Fix All Issues", key="auto_fix_all", use_container_width=True, type="primary"):
                    try:
                        _snap = snapshot()
                        with st.spinner("Running complete cleaning pipeline..."):
                            df_temp = st.session_state.current_df.copy()
                            df_temp = strip_whitespace(df_temp)
                            df_temp = remove_duplicate_rows(df_temp)
                            df_temp = remove_duplicate_columns(df_temp)
                            df_temp = clean_string_edges(df_temp, threshold=0.7, verbose=False)
                            df_temp = smart_column_cleaner(df_temp, conversion_threshold=conversion_threshold, verbose=False)
                            df_temp = missing_value_handler(df_temp, threshold=missing_threshold, numeric_strategy=numeric_strategy, verbose=False)
                            st.session_state.current_df = df_temp
                            commit_history("Auto-Fix All", _snap)
                            st.session_state.selected_columns = {}
                            st.session_state.last_success_msg = "All issues fixed automatically!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        # clean tab
        with tab_clean:
            st.subheader("Manual Cleaning Operations")

            st.write("**Basic Cleaning**")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("Strip Whitespace", key="ws_btn", use_container_width=True,
                             help="Removes leading and trailing spaces from all text columns. e.g. '  Alice ' becomes 'Alice'."):
                    try:
                        _snap = snapshot()
                        st.session_state.current_df = strip_whitespace(st.session_state.current_df)
                        commit_history("Strip Whitespace", _snap)
                        st.session_state.last_success_msg = "Whitespace stripped from text columns!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            with col2:
                if st.button("Drop Duplicate Rows", key="ddr_btn", use_container_width=True,
                             help="Removes rows that are completely identical to another row. Keeps the first occurrence."):
                    try:
                        _snap = snapshot()
                        before_rows = st.session_state.current_df.shape[0]
                        st.session_state.current_df = remove_duplicate_rows(st.session_state.current_df)
                        commit_history("Drop Duplicate Rows", _snap)
                        after_rows = st.session_state.current_df.shape[0]
                        st.session_state.last_success_msg = f"Dropped {before_rows-after_rows} duplicate rows!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            with col3:
                if st.button("Drop Duplicate Columns", key="ddc_btn", use_container_width=True,
                             help="Removes columns that share a name or have identical values to another column."):
                    try:
                        _snap = snapshot()
                        before_cols = st.session_state.current_df.shape[1]
                        st.session_state.current_df = remove_duplicate_columns(st.session_state.current_df)
                        commit_history("Drop Duplicate Columns", _snap)
                        after_cols = st.session_state.current_df.shape[1]
                        st.session_state.last_success_msg = f"Dropped {before_cols-after_cols} duplicate columns!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            with col4:
                if st.button("Clean String Edges", key="cse_btn", use_container_width=True,
                             help="Removes unwanted special characters from the start and end of text values. e.g. '$hello$' becomes 'hello'."):
                    try:
                        _snap = snapshot()
                        with st.spinner("Cleaning string edges..."):
                            st.session_state.current_df = clean_string_edges(
                                st.session_state.current_df,
                                threshold=0.7,
                                verbose=False
                            )
                        commit_history("Clean String Edges", _snap)
                        st.session_state.last_success_msg = "String edges cleaned!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            show_msg()

            st.write("")
            st.write("**Advanced Cleaning**")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Smart Column Cleaner", key="scc_btn", use_container_width=True,
                            help="Auto-detects and converts columns that look like currency, percentages, units, or durations into proper numeric values."):
                    try:
                        _snap = snapshot()
                        with st.spinner("Analyzing and converting columns..."):
                            st.session_state.current_df = smart_column_cleaner(
                                st.session_state.current_df,
                                conversion_threshold=conversion_threshold,
                                verbose=False
                            )
                        commit_history("Smart Column Cleaner", _snap)
                        st.session_state.last_success_msg = "Columns converted!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            with col2:
                if st.button("Handle Missing Values", key="hmv_btn", use_container_width=True,
                            help="Fills gaps using KNN imputation for numeric columns (estimates from nearby similar rows) and the most common value for text columns."):
                    try:
                        _snap = snapshot()
                        with st.spinner("Handling missing values, this may take a moment..."):
                            st.session_state.current_df = missing_value_handler(
                                st.session_state.current_df,
                                threshold=missing_threshold,
                                numeric_strategy=numeric_strategy,
                                verbose=False
                            )
                        commit_history("Handle Missing Values", _snap)
                        st.session_state.last_success_msg = "Missing values handled!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            with col3:
                if st.button("Full Pipeline", key="fp_btn", use_container_width=True,
                            help="Apply all cleaning operations in optimal order"):
                    try:
                        _snap = snapshot()
                        with st.spinner("Running full cleaning pipeline..."):
                            df_temp = st.session_state.current_df.copy()
                            df_temp = strip_whitespace(df_temp)
                            df_temp = remove_duplicate_rows(df_temp)
                            df_temp = remove_duplicate_columns(df_temp)
                            df_temp = clean_string_edges(df_temp, threshold=0.7, verbose=False)
                            df_temp = smart_column_cleaner(df_temp, conversion_threshold=conversion_threshold, verbose=False)
                            df_temp = missing_value_handler(
                                df_temp,
                                threshold=missing_threshold,
                                numeric_strategy=numeric_strategy,
                                verbose=False
                            )
                            st.session_state.current_df = df_temp
                        commit_history("Full Pipeline", _snap)
                        st.session_state.last_success_msg = "Full pipeline completed successfully!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            show_msg()

            st.divider()

            # find and replace
            st.write("**Find & Replace**")
            fr1, fr2 = st.columns([3, 1])
            with fr1:
                fr_col = st.selectbox("Column", all_cols, key="fr_col")
            with fr2:
                fr_regex = st.checkbox(
                    "Use Regex", key="fr_regex", value=False,
                    help=(
                        "Regex lets you match patterns instead of exact text.\n\n"
                        "Leave off for simple replacements like swapping 'N/A' with nothing.\n\n"
                        "Turn on when you need patterns:\n"
                        "- Remove all digits: find '\\d+', replace with ''\n"
                        "- Remove all letters: find '[a-zA-Z]+', replace with ''\n"
                        "- Match $ or £ or €: find '[$£€]', replace with ''\n"
                        "- Remove anything in brackets: find '\\(.*?\\)', replace with ''\n"
                        "- Collapse extra spaces: find '\\s+', replace with ' '"
                    )
                )
            fr3, fr4, fr5 = st.columns([2, 2, 1])
            with fr3:
                fr_find = st.text_input("Find", key="fr_find", placeholder="e.g. N/A")
            with fr4:
                fr_replace = st.text_input("Replace with", key="fr_replace", placeholder="leave blank to delete")
            with fr5:
                st.write("")
                st.write("")
                if st.button("Run", key="run_find_replace", use_container_width=True,
                             type="primary" if fr_find else "secondary",
                             disabled=not fr_find):
                    try:
                        _snap = snapshot()
                        st.session_state.current_df = find_and_replace(
                            st.session_state.current_df,
                            col=fr_col,
                            find=fr_find,
                            replace=fr_replace,
                            use_regex=fr_regex
                        )
                        commit_history(f"Find & Replace in {fr_col}", _snap)
                        st.session_state.last_success_msg = f"Find & Replace done on column '{fr_col}'!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            show_msg()

            st.divider()

            # column type override
            st.write("**Column Type Override**")
            to1, to2, to3 = st.columns([3, 2, 1])
            with to1:
                ov_col = st.selectbox("Column", all_cols, key="ov_col")
            with to2:
                ov_type = st.selectbox(
                    "Cast to",
                    ["string (object)", "integer (int64)", "float (float64)", "datetime", "boolean", "category"],
                    key="ov_type",
                    help="string: plain text. integer: whole numbers. float: decimals. datetime: dates and times. boolean: true/false values. category: a fixed set of labels, saves memory."
                )
            with to3:
                st.write("")
                st.write("")
                if st.button("Apply", key="ov_apply", type="primary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        df_temp = st.session_state.current_df.copy()
                        col_data = df_temp[ov_col]
                        if ov_type == "string (object)":
                            df_temp[ov_col] = col_data.astype(str)
                        elif ov_type == "integer (int64)":
                            df_temp[ov_col] = pd.to_numeric(col_data, errors="coerce").astype("Int64")
                        elif ov_type == "float (float64)":
                            df_temp[ov_col] = pd.to_numeric(col_data, errors="coerce")
                        elif ov_type == "datetime":
                            df_temp[ov_col] = pd.to_datetime(col_data, errors="coerce")
                        elif ov_type == "boolean":
                            df_temp[ov_col] = col_data.astype(str).str.lower().map(
                                {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
                            )
                        elif ov_type == "category":
                            df_temp[ov_col] = col_data.astype("category")
                        st.session_state.current_df = df_temp
                        commit_history(f"Type Override: {ov_col} -> {ov_type}", _snap)
                        st.session_state.last_success_msg = f"Column '{ov_col}' cast to {ov_type}!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            show_msg()

        # validate tab
        with tab_validate:
            st.subheader("Validation & Quality")

            # email validation
            if text_cols:
                st.write("**Validate Email**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1:
                    st.caption("Checks email format. Flag adds a boolean column, Remove drops invalid rows.")
                    email_action = st.radio("Action", ["Flag invalid", "Remove invalid rows"],
                                            key="email_action_radio", horizontal=True, label_visibility="collapsed",
                                            help="Flag adds an 'email_valid' boolean column so you can review invalid ones. Remove deletes rows where the email doesn't match the standard format.")
                with v2:
                    n_email = col_popover("email", text_cols)
                with v3:
                    if st.button("Run", key="run_email_val", use_container_width=True,
                                 disabled=n_email == 0, type="primary" if n_email > 0 else "secondary"):
                        try:
                            _snap = snapshot()
                            action_key = 'flag' if 'Flag' in email_action else 'remove'
                            df_temp = st.session_state.current_df.copy()
                            for col in st.session_state.val_selected["email"]:
                                df_temp = validate_email_col(df_temp, col, action=action_key)
                            st.session_state.current_df = df_temp
                            commit_history("Validate Email", _snap)
                            st.session_state.val_selected.pop("email", None)
                            st.session_state.last_success_msg = f"Email validation done on {n_email} column(s)!"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                show_msg()
                st.divider()

                # phone standardization
                st.write("**Standardize Phone Numbers**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1:
                    st.caption("Strips non-digit characters and formats to +[country code][number].")
                with v2:
                    n_phone = col_popover("phone", text_cols)
                with v3:
                    if st.button("Run", key="run_phone_val", use_container_width=True,
                                 disabled=n_phone == 0, type="primary" if n_phone > 0 else "secondary"):
                        try:
                            _snap = snapshot()
                            df_temp = st.session_state.current_df.copy()
                            for col in st.session_state.val_selected["phone"]:
                                df_temp = validate_phone_col(df_temp, col)
                            st.session_state.current_df = df_temp
                            commit_history("Standardize Phone", _snap)
                            st.session_state.val_selected.pop("phone", None)
                            st.session_state.last_success_msg = f"Phone numbers standardized in {n_phone} column(s)!"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                show_msg()
                st.divider()

                # date standardization
                st.write("**Standardize Dates**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1:
                    date_fmt = st.selectbox(
                        "Output format",
                        ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y'],
                        key="date_fmt_select",
                        help="%Y-%m-%d = 2023-01-15, recommended, sorts correctly. %d/%m/%Y = 15/01/2023, common in Europe. %m/%d/%Y = 01/15/2023, US format."
                    )
                with v2:
                    n_date = col_popover("date", text_cols)
                with v3:
                    st.write("")
                    if st.button("Run", key="run_date_val", use_container_width=True,
                                 disabled=n_date == 0, type="primary" if n_date > 0 else "secondary"):
                        try:
                            _snap = snapshot()
                            df_temp = st.session_state.current_df.copy()
                            for col in st.session_state.val_selected["date"]:
                                df_temp = validate_date_col(df_temp, col, output_format=date_fmt)
                            st.session_state.current_df = df_temp
                            commit_history("Standardize Dates", _snap)
                            st.session_state.val_selected.pop("date", None)
                            st.session_state.last_success_msg = f"Dates standardized to {date_fmt} in {n_date} column(s)!"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                show_msg()

            # cap or remove outliers
            if num_cols:
                st.divider()
                st.write("**Cap / Remove Outliers**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1:
                    o1, o2, o3 = st.columns(3)
                    with o1:
                        outlier_method = st.selectbox(
                            "Method", ["iqr", "zscore"], key="outlier_method",
                            help="IQR uses the spread of the middle 50% of data, good for skewed data and most cases. Z-score uses standard deviations from the mean, better for normally distributed data."
                        )
                    with o2:
                        outlier_action = st.selectbox(
                            "Action", ["cap", "remove"], key="outlier_action",
                            help="Cap clips outliers to the boundary value instead of deleting them, safer, keeps row count. Remove deletes the entire row containing the outlier."
                        )
                    with o3:
                        outlier_thresh = st.number_input(
                            "Threshold", min_value=0.5, max_value=10.0,
                            value=1.5, step=0.5, key="outlier_thresh",
                            help="For IQR: multiplier of the IQR range, 1.5 is standard, 3.0 is more lenient. For Z-score: number of standard deviations, 2.0 catches about 5% of data, 3.0 catches about 0.3%."
                        )
                with v2:
                    n_outlier = col_popover("outlier", num_cols)
                with v3:
                    st.write("")
                    st.write("")
                    if st.button("Run", key="run_outlier", use_container_width=True,
                                 disabled=n_outlier == 0, type="primary" if n_outlier > 0 else "secondary"):
                        try:
                            _snap = snapshot()
                            before = len(st.session_state.current_df)
                            df_temp = st.session_state.current_df.copy()
                            for col in st.session_state.val_selected["outlier"]:
                                df_temp = cap_outliers(df_temp, col=col, method=outlier_method,
                                                       action=outlier_action, threshold=outlier_thresh)
                            st.session_state.current_df = df_temp
                            commit_history("Cap Outliers", _snap)
                            after = len(st.session_state.current_df)
                            st.session_state.val_selected.pop("outlier", None)
                            if outlier_action == 'cap':
                                st.session_state.last_success_msg = f"Outliers capped in {n_outlier} column(s)!"
                            else:
                                st.session_state.last_success_msg = f"Removed {before - after} outlier rows across {n_outlier} column(s)!"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                show_msg()

            # validate value range
            if num_cols:
                st.divider()
                st.write("**Validate Value Range**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1:
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        range_min = st.number_input("Min value", value=0.0, key="range_min")
                    with r2:
                        range_max = st.number_input("Max value", value=100.0, key="range_max")
                    with r3:
                        range_action = st.selectbox(
                            "Action", ["flag", "remove"], key="range_action",
                            help="Flag adds a new boolean column showing which rows are in range, non-destructive. Remove deletes rows where the value falls outside the min/max."
                        )
                with v2:
                    n_range = col_popover("range", num_cols)
                with v3:
                    st.write("")
                    st.write("")
                    if st.button("Run", key="run_range_val", use_container_width=True,
                                 disabled=n_range == 0, type="primary" if n_range > 0 else "secondary"):
                        try:
                            _snap = snapshot()
                            before = len(st.session_state.current_df)
                            df_temp = st.session_state.current_df.copy()
                            for col in st.session_state.val_selected["range"]:
                                df_temp = validate_range(df_temp, col=col,
                                                         min_val=range_min, max_val=range_max,
                                                         action=range_action)
                            st.session_state.current_df = df_temp
                            commit_history("Validate Range", _snap)
                            after = len(st.session_state.current_df)
                            st.session_state.val_selected.pop("range", None)
                            if range_action == 'flag':
                                st.session_state.last_success_msg = f"Range flagged across {n_range} column(s)!"
                            else:
                                st.session_state.last_success_msg = f"Removed {before - after} out-of-range rows across {n_range} column(s)!"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                show_msg()

        # history and export tab
        with tab_history:

            # reset to original, prominent at the top
            st.subheader("Reset Data")
            st.warning("This will discard all cleaning and restore the original uploaded file.")
            if st.button("Reset to Original Data", key="reset_to_original_button", use_container_width=True):
                st.session_state.current_df = st.session_state.original_df.copy()
                st.session_state.selected_columns = {}
                st.session_state.history = []
                st.session_state["history_len"] = st.session_state.get("history_len", 0) + 1
                st.session_state.last_success_msg = "Data reset to original!"
                st.rerun()

            st.divider()

            # download section
            st.subheader("Download Cleaned Data")
            col1, col2 = st.columns(2)

            with col1:
                csv = st.session_state.current_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    key="download_csv_button",
                    use_container_width=True
                )

            with col2:
                # only regenerate excel bytes when the df actually changes
                # history_len bumps on every commit_history call, a cheap cache key
                excel_cache_key = st.session_state.get("history_len", 0)
                if st.session_state.get("_excel_cache_key") != excel_cache_key:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        st.session_state.current_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                        st.session_state.original_df.to_excel(writer, index=False, sheet_name='Original Data')
                    st.session_state["_excel_bytes"] = buffer.getvalue()
                    st.session_state["_excel_cache_key"] = excel_cache_key

                st.download_button(
                    label="Download as Excel",
                    data=st.session_state["_excel_bytes"],
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_button",
                    use_container_width=True
                )

            st.divider()

            # cleaning history and undo
            st.subheader("Cleaning History")

            history = st.session_state.get('history', [])

            if not history:
                st.caption("No operations recorded yet. Every cleaning action is saved here.")
            else:
                st.caption(f"{len(history)} operation(s) recorded. Max 20 steps kept.")
                for i, step in enumerate(reversed(history)):
                    step_num = len(history) - i
                    st.write(f"**{step_num}.** {step['label']}, "
                             f"{step['df'].shape[0]} rows x {step['df'].shape[1]} cols")

                st.write("")
                col_undo, col_clear = st.columns(2)
                with col_undo:
                    if st.button("Undo Last Step", key="undo_btn", use_container_width=True, type="primary"):
                        label = undo_last()
                        if label:
                            st.session_state.last_success_msg = f"Undone: {label}"
                        st.rerun()
                with col_clear:
                    if st.button("Clear History", key="clear_history_btn", use_container_width=True):
                        st.session_state.history = []
                        st.rerun()

            st.divider()

            # pipeline export
            st.subheader("Export Cleaning Pipeline")
            st.caption("Download your cleaning steps as a Python script you can rerun on any new file.")
            if not history:
                st.caption("No steps recorded yet. Run some cleaning operations first.")
            else:
                pipeline_script = build_pipeline_script(history)
                st.download_button(
                    "Download pipeline.py",
                    data=pipeline_script.encode("utf-8"),
                    file_name="pipeline.py",
                    mime="text/x-python",
                    key="dl_pipeline",
                    use_container_width=True
                )
                with st.expander("Preview script", expanded=False):
                    st.code(pipeline_script, language="python")

    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.info("Make sure your file is a valid CSV or Excel format.")