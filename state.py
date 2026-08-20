import io
import streamlit as st
from cache import get_dataframe_stats, load_file, make_df_key

# sidebar settings
def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        st.divider()
        mode = st.radio(
            "Mode", ["Simple", "Advanced"], horizontal=True, key="mode_radio",
            help="Simple uses sensible defaults. Switch to Advanced to unlock extra features.",
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
        # only clear session state, don't nuke all cache
        # cache entries miss naturally on the new df_key
        keys_to_clear = [
            k for k in st.session_state.keys()
            if k not in ("uploader", "mode_radio", "sheet_selector",
                         "tour_active", "tour_step", "tour_seen")
        ]
        for k in keys_to_clear:
            del st.session_state[k]
        st.session_state["loaded_file_id"] = file_id
        st.session_state["missing_threshold_val"] = 30
        st.session_state["conversion_threshold_val"] = 60
        st.rerun()

# set up session state for the loaded dataframe
def init_state(df, load_key, file_bytes: bytes = b"", filename: str = ""):
    """
    Initialises session state for a loaded file.
    stable_key is built from filename plus file size so it survives page reloads.
    Streamlit file_id changes on every reload so it is only used for the in-memory
    load_key, never for the on-disk session key.

    Flow:
    1. If this load_key is already initialised just sync df_key and return.
    2. If a persisted session exists on disk and user has not yet been asked
       show the resume dialog and halt further init until they choose.
    3. If user chose continue restore from disk.
    4. If user chose fresh or no saved session exists init from scratch.
    """
    from session_persist import (
        cleanup_old_sessions,
        delete_session,
        load_session,
        make_stable_file_key,
        session_exists,
    )

    state_key = f"state_{load_key}"
    stable_key = make_stable_file_key(filename, file_bytes) if filename else load_key

    if st.session_state.get("state_key_id") == state_key:
        st.session_state["file_just_loaded"] = False
        # keep current_df_key in sync whenever df changes after a clean op
        # make_df_key only hashes a tiny sample so this is cheap
        st.session_state["current_df_key"] = make_df_key(st.session_state.current_df)
        return

    cleanup_old_sessions()

    # clear stale checkbox widget keys from the previous sheet, avoids ghost selections
    stale = [k for k in st.session_state.keys() if k.startswith(("_vc_", "_va_", "_rc_", "_ra_"))]
    for k in stale:
        del st.session_state[k]

    resume_choice = st.session_state.get("_resume_choice")
    resume_stable_key = st.session_state.get("_resume_stable_key")

    if resume_choice is None and session_exists(stable_key):
        saved = load_session(stable_key)
        if saved:
            _show_resume_dialog(stable_key, saved)
            st.stop()

    if resume_choice == "continue" and resume_stable_key == stable_key:
        saved = load_session(stable_key)
        if saved:
            st.session_state.pop("_resume_choice", None)
            st.session_state.pop("_resume_stable_key", None)
            orig_df = saved["original_df"]
            curr_df = saved["current_df"]
            df_key = make_df_key(curr_df)
            orig_key = make_df_key(orig_df)
            st.session_state.update({
                "original_df": orig_df,
                "current_df": curr_df,
                "original_df_key": orig_key,
                "current_df_key": df_key,
                "original_stats": get_dataframe_stats(orig_key, orig_df),
                "selected_columns": {},
                "val_selected": {},
                "last_success_msg": None,
                "history": saved.get("history", []),
                "history_len": saved.get("history_len", 0),
                "state_key_id": state_key,
                "file_just_loaded": True,
                "_persist_key": stable_key,
            })
            if "val_selected" not in st.session_state:
                st.session_state.val_selected = {}
            return

    if resume_choice == "fresh" and resume_stable_key == stable_key:
        delete_session(stable_key)
        st.session_state.pop("_resume_choice", None)
        st.session_state.pop("_resume_stable_key", None)

    df_key = make_df_key(df)
    st.session_state.update({
        "original_df": df.copy(),
        "current_df": df.copy(),
        "original_df_key": df_key,
        "current_df_key": df_key,
        "original_stats": get_dataframe_stats(df_key, df),
        "selected_columns": {},
        "val_selected": {},
        "last_success_msg": None,
        "history": [],
        "history_len": 0,
        "state_key_id": state_key,
        "file_just_loaded": True,
        "_persist_key": stable_key,
    })

    if "val_selected" not in st.session_state:
        st.session_state.val_selected = {}

# dialog shown when a saved session is found for the uploaded file
# user picks continue or start fresh, the choice is written to session
# state and the dialog closes via st.rerun so normal init can proceed
@st.dialog("Resume your previous session?")
def _show_resume_dialog(stable_key: str, saved_info: dict):
    from session_persist import format_saved_time
    saved_ago = format_saved_time(saved_info["saved_at"])
    n_steps = len(saved_info.get("history", []))
    st.write(f"A saved session was found for this file from **{saved_ago}**.")
    st.write(f"It has **{n_steps} cleaning step{'s' if n_steps != 1 else ''}** recorded.")
    st.write("Do you want to continue where you left off, or start fresh?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continue where I left off", type="primary", use_container_width=True):
            st.session_state["_resume_choice"] = "continue"
            st.session_state["_resume_stable_key"] = stable_key
            st.rerun()
    with col2:
        if st.button("Start fresh", use_container_width=True):
            st.session_state["_resume_choice"] = "fresh"
            st.session_state["_resume_stable_key"] = stable_key
            st.rerun()

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
