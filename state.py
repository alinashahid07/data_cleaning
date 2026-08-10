import io
import streamlit as st
from cache import get_dataframe_stats, load_file

# sidebar settings
def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Mode", ["Simple", "Advanced"], horizontal=True, key="mode_radio",
            help="Simple uses sensible defaults so you can start cleaning right away. Advanced lets you tune the thresholds manually."
        )
        if mode == "Simple":
            st.caption("Using default settings. Switch to Advanced to unlock extra features.")
            return 0.30, "auto", 0.60, "Simple"

        st.subheader("Missing Value Handler")
        missing_threshold = st.slider(
            "Drop columns with missing % >", 0, 100,
            value=st.session_state.get("missing_threshold_val", 30),
            help="Columns where more than this percent of values are missing get dropped entirely."
        ) / 100
        st.session_state["missing_threshold_val"] = int(missing_threshold * 100)

        numeric_strategy = st.selectbox(
            "Numeric Imputation Strategy", ["auto", "knn", "mice"],
            key="numeric_strategy_select",
            help="auto picks KNN for small files and MICE for large ones."
        )

        st.subheader("Smart Cleaner")
        conversion_threshold = st.slider(
            "Conversion Threshold %", 0, 100,
            value=st.session_state.get("conversion_threshold_val", 60),
            help="How many values in a column must match a pattern before the whole column gets converted."
        ) / 100
        st.session_state["conversion_threshold_val"] = int(conversion_threshold * 100)
    return missing_threshold, numeric_strategy, conversion_threshold, "Advanced"

# read and prep an uploaded file
def resolve_upload(uploaded):
    ext = uploaded.name.split(".")[-1].lower()
    selected_sheet = None
    if ext in ["xlsx", "xls"]:
        import pandas as pd
        xl_bytes = uploaded.read()
        xl = pd.ExcelFile(io.BytesIO(xl_bytes))
        sheets = xl.sheet_names
        selected_sheet = (
            st.selectbox("Select sheet:", sheets, key="sheet_selector")
            if len(sheets) > 1 else sheets[0]
        )
        return xl_bytes, selected_sheet
    return uploaded.read(), selected_sheet

# wipe state on a genuinely new upload
def maybe_reset_on_new_upload(file_id):
    if st.session_state.get("loaded_file_id") != file_id:
        st.cache_data.clear()
        keys_to_clear = [
            k for k in st.session_state.keys()
            if k not in ("uploader", "mode_radio", "sheet_selector")
        ]
        for k in keys_to_clear:
            del st.session_state[k]
        st.session_state["loaded_file_id"] = file_id
        st.session_state["missing_threshold_val"] = 30
        st.session_state["conversion_threshold_val"] = 60
        st.rerun()

# set up session state for the loaded dataframe
def init_state(df, load_key):
    state_key = f"state_{load_key}"
    if st.session_state.get("state_key_id") != state_key:
        # clear stale checkbox widget keys from the previous sheet, avoids ghost selections
        stale = [k for k in st.session_state.keys() if k.startswith(("_vc_", "_va_", "_rc_", "_ra_"))]
        for k in stale:
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

# show and immediately clear any pending success message
def show_msg():
    if st.session_state.get("last_success_msg"):
        st.success(st.session_state.last_success_msg)
        st.session_state.last_success_msg = None

def _make_all_handler(section, cols):
    def h():
        checked = st.session_state.get(f"_va_{section}", False)
        st.session_state.val_selected[section] = cols.copy() if checked else []
    return h

def _make_col_handler(section, col):
    def h():
        sel = st.session_state.val_selected.get(section, [])
        if st.session_state.get(f"_vc_{section}_{col}"):
            if col not in sel:
                st.session_state.val_selected[section] = sel + [col]
        else:
            st.session_state.val_selected[section] = [c for c in sel if c != col]
    return h

# renders the column selector popover and returns selected count
def col_popover(section, available_cols):
    n = len(st.session_state.val_selected.get(section, []))
    label = f"{n} selected" if n else "Select columns"
    with st.popover(label, use_container_width=True):
        st.checkbox("Apply to all", key=f"_va_{section}", on_change=_make_all_handler(section, available_cols))
        for c in available_cols:
            st.checkbox(c, key=f"_vc_{section}_{c}", on_change=_make_col_handler(section, c))
    return n
