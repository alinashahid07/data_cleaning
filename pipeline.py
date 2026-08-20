import json
import streamlit as st

# snapshot before an operation
def snapshot():
    """Takes a snapshot of current_df. Pass the result to commit_history() once the operation succeeds."""
    return st.session_state.current_df.copy()

# saves the current session to disk after every mutating operation
# runs only when a persist key is set, which happens after init_state
# the save is called after the mutation is committed to session state
# so the file on disk always reflects the latest state
def _autosave():
    persist_key = st.session_state.get("_persist_key")
    st.write(f"[autosave] persist_key: {persist_key}")
    if not persist_key:
        st.write("[autosave] no persist key, skipping")
        return
    try:
        from session_persist import save_session
        st.write("[autosave] calling save_session")
        save_session(
            stable_key=persist_key,
            current_df=st.session_state.current_df,
            history=st.session_state.get("history", []),
            original_df=st.session_state.get("original_df", st.session_state.current_df),
        )
        st.write("[autosave] save_session returned")
    except Exception as e:
        st.write(f"[autosave] exception: {e}")

# commit a snapshot after success
def commit_history(label, snap):
    """Commits a previously taken snapshot to history. Only call this after the operation succeeds."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if len(st.session_state.history) >= 20:
        st.session_state.history.pop(0)
    st.session_state.history.append({"label": label, "df": snap})
    st.session_state["history_len"] = st.session_state.get("history_len", 0) + 1
    _autosave()

# restore previous state
def undo_last():
    if st.session_state.get("history"):
        last = st.session_state.history.pop()
        st.session_state.current_df = last["df"]
        st.session_state["history_len"] = st.session_state.get("history_len", 0) + 1
        _autosave()
        return last["label"]
    return None

# save history labels as a small portable recipe file
def export_pipeline_json(history):
    steps = [{"step": i + 1, "label": entry["label"]} for i, entry in enumerate(history)]
    return json.dumps({"version": 1, "steps": steps}, indent=2)

# replay a saved pipeline json against a fresh dataframe
def import_pipeline_json(json_bytes, df, settings):
    """Steps that need manual column selection are skipped and returned to the caller."""
    from cleaning import (
        clean_string_edges,
        remove_duplicate_columns,
        remove_duplicate_rows,
        strip_whitespace,
        missing_value_handler,
        smart_column_cleaner,
    )

    missing_threshold = settings.get("missing_threshold", 0.3)
    numeric_strategy = settings.get("numeric_strategy", "auto")
    conversion_threshold = settings.get("conversion_threshold", 0.6)

    try:
        payload = json.loads(json_bytes)
    except Exception:
        raise ValueError("File is not valid JSON.")
    if not isinstance(payload, dict) or "steps" not in payload:
        raise ValueError("JSON does not look like a saved pipeline file.")

    tmp = df.copy()
    applied = []
    skipped = []
    for entry in payload["steps"]:
        label = entry.get("label", "")
        if label == "Strip Whitespace" or label.startswith("Fix: strip_whitespace"):
            tmp = strip_whitespace(tmp)
            applied.append(label)
        elif label == "Drop Duplicate Rows" or label.startswith("Fix: drop_duplicates"):
            tmp = remove_duplicate_rows(tmp)
            applied.append(label)
        elif label == "Drop Duplicate Columns" or label.startswith("Fix: drop_dup_cols"):
            tmp = remove_duplicate_columns(tmp)
            applied.append(label)
        elif label == "Clean String Edges" or label.startswith("Fix: clean_edges"):
            tmp = clean_string_edges(tmp, threshold=0.7)
            applied.append(label)
        elif label == "Smart Column Cleaner" or label.startswith("Fix: convert_"):
            tmp = smart_column_cleaner(tmp, conversion_threshold=conversion_threshold)
            applied.append(label)
        elif label == "Handle Missing Values" or label.startswith("Fix: handle_missing"):
            tmp = missing_value_handler(tmp, threshold=missing_threshold, numeric_strategy=numeric_strategy)
            applied.append(label)
        elif label == "Auto-Fix All" or label == "Full Pipeline":
            tmp = strip_whitespace(tmp)
            tmp = remove_duplicate_rows(tmp)
            tmp = remove_duplicate_columns(tmp)
            tmp = clean_string_edges(tmp, threshold=0.7)
            tmp = smart_column_cleaner(tmp, conversion_threshold=conversion_threshold)
            tmp = missing_value_handler(tmp, threshold=missing_threshold, numeric_strategy=numeric_strategy)
            applied.append(label)
        else:
            skipped.append(label)
    return tmp, applied, skipped

# turn history into a runnable script
def build_pipeline_script(history):
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
        label = step["label"]
        lines.append(f"# {label}")

        if label == "Strip Whitespace" or label.startswith("Fix: strip_whitespace"):
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")

        elif label == "Drop Duplicate Rows" or label.startswith("Fix: drop_duplicates"):
            lines.append("df = df.drop_duplicates().reset_index(drop=True)")

        elif label == "Drop Duplicate Columns" or label.startswith("Fix: drop_dup_cols"):
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")
            lines.append("df = df.T.drop_duplicates().T")

        elif label == "Clean String Edges" or label.startswith("Fix: clean_edges"):
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    df[col] = df[col].astype(str).str.replace(r'^\\W+', '', regex=True).str.replace(r'\\W+$', '', regex=True)")

        elif label == "Smart Column Cleaner" or label.startswith("Fix: convert_"):
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    cleaned = df[col].str.replace(r'[^\\d.\\-]', '', regex=True)")
            lines.append("    converted = pd.to_numeric(cleaned, errors='coerce')")
            lines.append("    if converted.notna().mean() > 0.6:")
            lines.append("        df[col] = converted")

        elif label == "Handle Missing Values" or label.startswith("Fix: handle_missing"):
            lines.append("df = df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan)")
            lines.append("num_cols = df.select_dtypes(include=np.number).columns")
            lines.append("if not num_cols.empty and df[num_cols].isna().any().any():")
            lines.append("    df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])")
            lines.append("for col in df.select_dtypes(exclude=np.number).columns:")
            lines.append("    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Missing')")

        elif label == "Auto-Fix All" or label == "Full Pipeline":
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")
            lines.append("df = df.drop_duplicates().reset_index(drop=True)")
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")
            lines.append("df = df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan)")
            lines.append("num_cols = df.select_dtypes(include=np.number).columns")
            lines.append("if not num_cols.empty and df[num_cols].isna().any().any():")
            lines.append("    df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])")

        elif label.startswith("Find and Replace in"):
            col = label.replace("Find and Replace in ", "").strip()
            lines.append(f"df['{col}'] = df['{col}'].astype(str).str.replace('FIND', 'REPLACE', regex=False)")

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

        elif label.startswith("Split column"):
            lines.append("# split column, update col name and delimiter below")
            lines.append("# parts = df['col'].str.split('delimiter', expand=True)")
            lines.append("# df['part1'] = parts[0]")
            lines.append("# df['part2'] = parts[1]")

        elif label.startswith("Merge columns into"):
            new_col = label.replace("Merge columns into ", "").strip()
            lines.append(f"# df['{new_col}'] = df['col1'].astype(str) + ' ' + df['col2'].astype(str)")

        elif label.startswith("Rename"):
            lines.append("# df = df.rename(columns={'old_name': 'new_name'})")

        elif label == "Validate Email":
            lines.append("pattern = r'^[\\w\\.\\+\\-]+@[\\w\\-]+\\.[a-zA-Z]{2,}$'")
            lines.append("# df['email_valid'] = df['email_col'].astype(str).str.match(pattern)")

        elif label == "Standardize Phone":
            lines.append("def standardize_phone(val):")
            lines.append("    digits = re.sub(r'\\D', '', str(val))")
            lines.append("    if len(digits) == 10: return f'+1{digits}'")
            lines.append("    elif len(digits) >= 7: return f'+{digits}'")
            lines.append("    return float('nan')")
            lines.append("# df['phone_col'] = df['phone_col'].apply(standardize_phone)")

        elif label == "Standardize Dates":
            lines.append("# df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce').dt.strftime('%Y-%m-%d')")

        elif label == "Cap Outliers":
            lines.append("# col = 'your_column'")
            lines.append("# q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)")
            lines.append("# df[col] = df[col].clip(lower=q1 - 1.5*(q3-q1), upper=q3 + 1.5*(q3-q1))")

        elif label == "Validate Range":
            lines.append("# df = df[df['your_column'].between(0, 100)]")

        else:
            lines.append("# manual step, no code generated")

        lines.append("")

    lines.append("print('Pipeline complete. Shape:', df.shape)")
    return "\n".join(lines)
