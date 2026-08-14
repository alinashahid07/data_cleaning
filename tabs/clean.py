import pandas as pd
import streamlit as st
from cache import get_type_suggestions
from cleaning import (
    apply_type_suggestions,
    clean_string_edges,
    remove_duplicate_columns,
    remove_duplicate_rows,
    find_and_replace,
    merge_columns,
    missing_value_handler,
    rename_columns,
    smart_column_cleaner,
    split_column,
    strip_whitespace,
)
from nl_cleaner import render_nl_cleaner
from pipeline import commit_history, snapshot

def _render_basic_cleaning(cdf):
    st.write("**Basic Cleaning**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Strip Whitespace", key="ws_btn", use_container_width=True,
                     help="Removes leading and trailing spaces from all text columns."):
            try:
                _snap = snapshot()
                st.session_state.current_df = strip_whitespace(cdf)
                commit_history("Strip Whitespace", _snap)
                st.session_state["_omsg"] = ("ws_btn", "Whitespace stripped.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if st.session_state.get("_omsg", ("",))[0] == "ws_btn":
            st.success(st.session_state.pop("_omsg")[1])

    with c2:
        if st.button("Drop Duplicate Rows", key="ddr_btn", use_container_width=True,
                     help="Removes rows that are completely identical. Keeps the first occurrence."):
            try:
                _snap = snapshot()
                before = len(cdf)
                st.session_state.current_df = remove_duplicate_rows(cdf)
                commit_history("Drop Duplicate Rows", _snap)
                dropped = before - len(st.session_state.current_df)
                st.session_state["_omsg"] = ("ddr_btn", f"Dropped {dropped} duplicate rows.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if st.session_state.get("_omsg", ("",))[0] == "ddr_btn":
            st.success(st.session_state.pop("_omsg")[1])

    with c3:
        if st.button("Drop Duplicate Cols", key="ddc_btn", use_container_width=True,
                     help="Removes columns with the same name or identical values as another column."):
            try:
                _snap = snapshot()
                before = cdf.shape[1]
                st.session_state.current_df = remove_duplicate_columns(cdf)
                commit_history("Drop Duplicate Columns", _snap)
                dropped = before - st.session_state.current_df.shape[1]
                st.session_state["_omsg"] = ("ddc_btn", f"Dropped {dropped} duplicate columns.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if st.session_state.get("_omsg", ("",))[0] == "ddc_btn":
            st.success(st.session_state.pop("_omsg")[1])

    with c4:
        if st.button("Clean String Edges", key="cse_btn", use_container_width=True,
                     help="Removes unwanted special characters from the start and end of text values."):
            try:
                _snap = snapshot()
                st.session_state.current_df = clean_string_edges(cdf, threshold=0.7)
                commit_history("Clean String Edges", _snap)
                st.session_state["_omsg"] = ("cse_btn", "String edges cleaned.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if st.session_state.get("_omsg", ("",))[0] == "cse_btn":
            st.success(st.session_state.pop("_omsg")[1])

def _render_advanced_cleaning(cdf, missing_threshold, numeric_strategy, conversion_threshold):
    st.write("**Advanced Cleaning**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Smart Column Cleaner", key="scc_btn", use_container_width=True,
                     help="Auto-detects and converts columns that look like currency, percentages, units, or durations."):
            try:
                _snap = snapshot()
                with st.spinner("Converting..."):
                    st.session_state.current_df = smart_column_cleaner(
                        cdf, conversion_threshold=conversion_threshold
                    )
                commit_history("Smart Column Cleaner", _snap)
                st.session_state["_omsg"] = ("scc_btn", "Columns converted.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if st.session_state.get("_omsg", ("",))[0] == "scc_btn":
            st.success(st.session_state.pop("_omsg")[1])

    with c2:
        if st.button("Handle Missing Values", key="hmv_btn", use_container_width=True,
                     help="Fills missing values using KNN for numeric columns and mode for text columns."):
            try:
                _snap = snapshot()
                with st.spinner("Imputing..."):
                    st.session_state.current_df = missing_value_handler(
                        cdf, threshold=missing_threshold, numeric_strategy=numeric_strategy
                    )
                commit_history("Handle Missing Values", _snap)
                st.session_state["_omsg"] = ("hmv_btn", "Missing values handled.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if st.session_state.get("_omsg", ("",))[0] == "hmv_btn":
            st.success(st.session_state.pop("_omsg")[1])

def _render_find_replace(cdf, all_cols):
    st.write("**Find and Replace**")
    fr1, fr2 = st.columns([3, 1])
    with fr1:
        fr_col = st.selectbox("Column", all_cols, key="fr_col")
    with fr2:
        fr_regex = st.checkbox(
            "Use Regex",
            key="fr_regex",
            help=(
                "Leave off for simple text replacements like swapping N/A with nothing. "
                "Turn on when you need patterns such as removing all digits or matching currency symbols."
            ),
        )
    fr3, fr4, fr5 = st.columns([2, 2, 1])
    with fr3:
        fr_find = st.text_input("Find", key="fr_find", placeholder="e.g. N/A")
    with fr4:
        fr_replace = st.text_input("Replace with", key="fr_replace", placeholder="leave blank to delete")
    with fr5:
        st.write("")
        st.write("")
        if st.button("Run", key="fr_run", disabled=not fr_find,
                     type="primary" if fr_find else "secondary", use_container_width=True):
            try:
                _snap = snapshot()
                st.session_state.current_df = find_and_replace(cdf, fr_col, fr_find, fr_replace, fr_regex)
                commit_history(f"Find and Replace in {fr_col}", _snap)
                st.session_state["_omsg"] = ("fr_run", f"Find and Replace done on '{fr_col}'.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    if st.session_state.get("_omsg", ("",))[0] == "fr_run":
        st.success(st.session_state.pop("_omsg")[1])

def _render_type_override(cdf, all_cols):
    st.write("**Column Type Override**")
    to1, to2, to3 = st.columns([3, 2, 1])
    with to1:
        ov_col = st.selectbox("Column", all_cols, key="ov_col")
    with to2:
        ov_type = st.selectbox(
            "Cast to",
            ["string (object)", "integer (int64)", "float (float64)", "datetime", "boolean", "category"],
            key="ov_type",
            help=(
                "string is plain text. integer is whole numbers. float is decimals. "
                "datetime is dates and times. boolean is true or false. "
                "category is a fixed set of labels that saves memory."
            ),
        )
    with to3:
        st.write("")
        st.write("")
        if st.button("Apply", key="ov_apply", type="primary", use_container_width=True):
            try:
                _snap = snapshot()
                tmp = cdf.copy()
                cd = tmp[ov_col]
                if ov_type == "string (object)":
                    tmp[ov_col] = cd.astype(str)
                elif ov_type == "integer (int64)":
                    tmp[ov_col] = pd.to_numeric(cd, errors="coerce").astype("Int64")
                elif ov_type == "float (float64)":
                    tmp[ov_col] = pd.to_numeric(cd, errors="coerce")
                elif ov_type == "datetime":
                    tmp[ov_col] = pd.to_datetime(cd, errors="coerce")
                elif ov_type == "boolean":
                    tmp[ov_col] = cd.astype(str).str.lower().map(
                        {"true": True, "1": True, "yes": True,
                         "false": False, "0": False, "no": False}
                    )
                elif ov_type == "category":
                    tmp[ov_col] = cd.astype("category")
                st.session_state.current_df = tmp
                commit_history(f"Type Override: {ov_col} -> {ov_type}", _snap)
                st.session_state["_omsg"] = ("ov_apply", f"Column '{ov_col}' cast to {ov_type}.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    if st.session_state.get("_omsg", ("",))[0] == "ov_apply":
        st.success(st.session_state.pop("_omsg")[1])

def _render_split_column(cdf, text_cols, all_cols):
    st.write("**Split Column**")
    st.caption(
        "Split one column into several new columns by a delimiter. "
        "For example split a full name column on a space to get first name and last name."
    )
    sp1, sp2 = st.columns([3, 1])
    with sp1:
        sp_col = st.selectbox("Column to split", all_cols, key="sp_col")
    with sp2:
        sp_delim = st.text_input("Delimiter", key="sp_delim", placeholder=", or space or -")
    sp3, sp4 = st.columns([3, 1])
    with sp3:
        sp_names_raw = st.text_input(
            "Output column names (comma separated)",
            key="sp_names",
            placeholder="e.g. first_name, last_name",
            help=(
                "Enter one name per resulting column, separated by commas. "
                "If a row has fewer parts than names, the extra columns get a blank value. "
                "If you enter fewer names than parts, the extra parts are ignored."
            ),
        )
    with sp4:
        sp_keep = st.checkbox(
            "Keep original",
            key="sp_keep",
            help="Keep the source column in the dataframe after splitting.",
        )
    sp_names = [n.strip() for n in sp_names_raw.split(",") if n.strip()] if sp_names_raw else []
    sp_ready = bool(sp_col and sp_delim and sp_names)
    if st.button(
        "Split",
        key="sp_run",
        disabled=not sp_ready,
        type="primary" if sp_ready else "secondary",
        use_container_width=True,
    ):
        try:
            _snap = snapshot()
            result = split_column(cdf, sp_col, sp_delim, sp_names, keep_original=sp_keep)
            st.session_state.current_df = result
            commit_history(f"Split column {sp_col} on '{sp_delim}'", _snap)
            st.session_state["_omsg"] = ("sp_run", f"Split '{sp_col}' into {len(sp_names)} columns.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if st.session_state.get("_omsg", ("",))[0] == "sp_run":
        st.success(st.session_state.pop("_omsg")[1])

def _render_merge_columns(cdf, all_cols):
    st.write("**Merge Columns**")
    st.caption(
        "Concatenate two or more columns into a single new column. "
        "For example combine city and country columns into one address column."
    )
    mg1, mg2 = st.columns([3, 1])
    with mg1:
        mg_cols = st.multiselect(
            "Columns to merge (select two or more in order)",
            all_cols,
            key="mg_cols",
        )
    with mg2:
        mg_sep = st.text_input("Separator", value=" ", key="mg_sep",
                               help="The string placed between each column value. Use a single space, comma, or anything you like.")
    mg3, mg4 = st.columns([3, 1])
    with mg3:
        mg_new = st.text_input("New column name", key="mg_new", placeholder="e.g. full_address")
    with mg4:
        mg_keep = st.checkbox(
            "Keep originals",
            key="mg_keep",
            help="Keep the source columns after merging.",
        )
    mg_ready = len(mg_cols) >= 2 and bool(mg_new and mg_new.strip())
    if st.button(
        "Merge",
        key="mg_run",
        disabled=not mg_ready,
        type="primary" if mg_ready else "secondary",
        use_container_width=True,
    ):
        try:
            _snap = snapshot()
            result = merge_columns(cdf, mg_cols, mg_new.strip(), mg_sep, keep_originals=mg_keep)
            st.session_state.current_df = result
            commit_history(f"Merge columns into {mg_new.strip()}", _snap)
            st.session_state["_omsg"] = (
                "mg_run",
                f"Merged {len(mg_cols)} columns into '{mg_new.strip()}'.",
            )
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if st.session_state.get("_omsg", ("",))[0] == "mg_run":
        st.success(st.session_state.pop("_omsg")[1])

def _render_rename_columns(cdf, all_cols):
    st.write("**Rename Columns**")
    st.caption("Edit any column name directly in the table below, then press Apply.")
    if "rename_state" not in st.session_state or st.session_state.get("rename_state_cols") != all_cols:
        st.session_state.rename_state = {c: c for c in all_cols}
        st.session_state.rename_state_cols = all_cols
    rename_df = pd.DataFrame({
        "Current name": all_cols,
        "New name": [st.session_state.rename_state.get(c, c) for c in all_cols],
    })
    edited = st.data_editor(
        rename_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Current name"],
        key="rename_editor",
        column_config={
            "Current name": st.column_config.TextColumn("Current name", disabled=True),
            "New name": st.column_config.TextColumn("New name"),
        },
    )
    rename_map = dict(zip(edited["Current name"].tolist(), edited["New name"].tolist()))
    changes = {old: new for old, new in rename_map.items() if new.strip() and new.strip() != old}
    rn1, rn2 = st.columns([3, 1])
    with rn1:
        if changes:
            st.caption(f"{len(changes)} column(s) will be renamed.")
        else:
            st.caption("No changes detected. Edit the New name column above.")
    with rn2:
        if st.button(
            "Apply",
            key="rn_apply",
            disabled=not changes,
            type="primary" if changes else "secondary",
            use_container_width=True,
        ):
            try:
                _snap = snapshot()
                result = rename_columns(cdf, rename_map)
                st.session_state.current_df = result
                st.session_state.rename_state = {}
                commit_history(f"Rename {len(changes)} column(s)", _snap)
                st.session_state["_omsg"] = ("rn_apply", f"Renamed {len(changes)} column(s).")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    if st.session_state.get("_omsg", ("",))[0] == "rn_apply":
        st.success(st.session_state.pop("_omsg")[1])

def _render_type_guesser(cdf):
    st.write("**Data Type Guesser**")
    st.caption(
        "Scans every column and suggests the correct type based on what the values actually look like. "
        "Select the suggestions you want to apply then press Apply Selected."
    )
    suggestions = get_type_suggestions(cdf)
    if not suggestions:
        st.success("All columns already look like the right type. Nothing to suggest.")
        return

    if "type_guesser_selected" not in st.session_state:
        st.session_state.type_guesser_selected = {}

    rows = []
    for s in suggestions:
        checked = st.session_state.type_guesser_selected.get(s["column"], True)
        rows.append({
            "Apply": checked,
            "Column": s["column"],
            "Current Type": s["current_type"],
            "Suggested Type": s["suggested_label"],
            "Confidence": f"{s['confidence']}%",
            "Reason": s["reason"],
            "Sample Values": s["sample"],
        })

    display_df = pd.DataFrame(rows)
    edited = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Column", "Current Type", "Suggested Type", "Confidence", "Reason", "Sample Values"],
        column_config={
            "Apply": st.column_config.CheckboxColumn("Apply", default=True),
            "Confidence": st.column_config.TextColumn("Confidence"),
        },
        key="type_guesser_editor",
    )
    selected = [suggestions[i] for i, row in edited.iterrows() if row["Apply"]]
    n_selected = len(selected)

    tg1, tg2 = st.columns([4, 1])
    with tg1:
        if n_selected:
            st.caption(f"{n_selected} suggestion(s) selected. Press Apply to convert.")
        else:
            st.caption("Tick at least one suggestion to enable Apply.")
    with tg2:
        if st.button(
            "Apply Selected",
            key="tg_apply",
            disabled=n_selected == 0,
            type="primary" if n_selected else "secondary",
            use_container_width=True,
        ):
            try:
                _snap = snapshot()
                result, applied = apply_type_suggestions(cdf, selected)
                st.session_state.current_df = result
                st.session_state.type_guesser_selected = {}
                commit_history(f"Type Guesser: fixed {len(applied)} column(s)", _snap)
                st.session_state["_omsg"] = (
                    "tg_apply",
                    f"Converted {len(applied)} column(s): {', '.join(applied)}.",
                )
                st.rerun()
            except Exception as e:
                st.error(str(e))
    if st.session_state.get("_omsg", ("",))[0] == "tg_apply":
        st.success(st.session_state.pop("_omsg")[1])

def _render_ai_cleaner(cdf):
    st.write("**AI Cleaner**")
    st.caption(
        "Describe what you want to do in plain English. Try to be as specific as possible.  "
        "Gemini generates the code and shows it to you before running anything."
    )
    render_nl_cleaner(cdf)

def render(tab, cdf, all_cols, missing_threshold, numeric_strategy, conversion_threshold):
    with tab:
        st.subheader("Manual Cleaning Operations")

        with st.expander("AI Cleaner: describe what you want in plain English", expanded=False):
            _render_ai_cleaner(cdf)
        st.divider()
        _render_basic_cleaning(cdf)
        st.write("")
        _render_advanced_cleaning(cdf, missing_threshold, numeric_strategy, conversion_threshold)
        st.divider()
        _render_find_replace(cdf, all_cols)
        st.divider()
        _render_type_override(cdf, all_cols)
        st.divider()
        _render_split_column(cdf, list(cdf.select_dtypes(include="object").columns), all_cols)
        st.divider()
        _render_merge_columns(cdf, all_cols)
        st.divider()
        _render_rename_columns(cdf, all_cols)

        st.divider()
        _render_type_guesser(cdf)
