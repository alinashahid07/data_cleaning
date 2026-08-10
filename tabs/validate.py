import streamlit as st
from cleaning import cap_outliers, validate_date_col, validate_email_col, validate_phone_col, validate_range
from pipeline import commit_history, snapshot
from state import col_popover

def render(tab, cdf, text_cols, num_cols):
    with tab:
        st.subheader("Validation and Quality")

        if text_cols:
            st.write("**Validate Email**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                st.caption("Flag adds a boolean column. Remove drops invalid rows.")
                ea = st.radio(
                    "",
                    ["Flag invalid", "Remove invalid rows"],
                    key="email_radio",
                    horizontal=True,
                    label_visibility="collapsed",
                    help="Flag adds an email valid boolean column. Remove deletes rows where the email does not match the standard format.",
                )
            with v2:
                n_em = col_popover("email", text_cols)
            with v3:
                if st.button("Run", key="run_email", disabled=n_em == 0,
                             type="primary" if n_em else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        tmp = cdf.copy()
                        for c in st.session_state.val_selected.get("email", []):
                            tmp = validate_email_col(tmp, c, "flag" if "Flag" in ea else "remove")
                        st.session_state.current_df = tmp
                        commit_history("Validate Email", _snap)
                        st.session_state.val_selected.pop("email", None)
                        st.session_state["_omsg"] = ("run_email", f"Email validation done on {n_em} column(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            if st.session_state.get("_omsg", ("",))[0] == "run_email":
                st.success(st.session_state.pop("_omsg")[1])

            st.divider()
            st.write("**Standardize Phone Numbers**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                st.caption("Strips non-digit characters and formats to +[country code][number].")
            with v2:
                n_ph = col_popover("phone", text_cols)
            with v3:
                if st.button("Run", key="run_phone", disabled=n_ph == 0,
                             type="primary" if n_ph else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        tmp = cdf.copy()
                        for c in st.session_state.val_selected.get("phone", []):
                            tmp = validate_phone_col(tmp, c)
                        st.session_state.current_df = tmp
                        commit_history("Standardize Phone", _snap)
                        st.session_state.val_selected.pop("phone", None)
                        st.session_state["_omsg"] = ("run_phone", f"Phone standardized in {n_ph} column(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            if st.session_state.get("_omsg", ("",))[0] == "run_phone":
                st.success(st.session_state.pop("_omsg")[1])

            st.divider()
            st.write("**Standardize Dates**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                date_fmt = st.selectbox(
                    "Output format",
                    ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"],
                    key="date_fmt",
                    help="YYYY-MM-DD is recommended because it sorts correctly. DD/MM/YYYY is common in Europe. MM/DD/YYYY is the US format.",
                )
            with v2:
                n_dt = col_popover("date", text_cols)
            with v3:
                st.write("")
                if st.button("Run", key="run_date", disabled=n_dt == 0,
                             type="primary" if n_dt else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        tmp = cdf.copy()
                        for c in st.session_state.val_selected.get("date", []):
                            tmp = validate_date_col(tmp, c, output_format=date_fmt)
                        st.session_state.current_df = tmp
                        commit_history("Standardize Dates", _snap)
                        st.session_state.val_selected.pop("date", None)
                        st.session_state["_omsg"] = ("run_date", f"Dates standardized in {n_dt} column(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            if st.session_state.get("_omsg", ("",))[0] == "run_date":
                st.success(st.session_state.pop("_omsg")[1])

        if num_cols:
            st.divider()
            st.write("**Cap and Remove Outliers**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                o1, o2, o3 = st.columns(3)
                with o1:
                    o_method = st.selectbox(
                        "Method", ["iqr", "zscore"], key="o_method",
                        help="IQR uses the spread of the middle 50% of data and works well for skewed data. Z-score uses standard deviations from the mean and is better for normally distributed data.",
                    )
                with o2:
                    o_action = st.selectbox(
                        "Action", ["cap", "remove"], key="o_action",
                        help="Cap clips outliers to the boundary value and keeps the row count. Remove deletes the entire row containing the outlier.",
                    )
                with o3:
                    o_thresh = st.number_input(
                        "Threshold", min_value=0.5, max_value=10.0, value=1.5, step=0.5, key="o_thresh",
                        help="For IQR, 1.5 is standard and 3.0 is more lenient. For Z-score, 2.0 catches about 5% of data and 3.0 catches about 0.3%.",
                    )
            with v2:
                n_out = col_popover("outlier", num_cols)
            with v3:
                st.write("")
                st.write("")
                if st.button("Run", key="run_outlier", disabled=n_out == 0,
                             type="primary" if n_out else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        before = len(cdf)
                        tmp = cdf.copy()
                        for c in st.session_state.val_selected.get("outlier", []):
                            tmp = cap_outliers(tmp, c, method=o_method, action=o_action, threshold=o_thresh)
                        st.session_state.current_df = tmp
                        after = len(tmp)
                        commit_history("Cap Outliers", _snap)
                        st.session_state.val_selected.pop("outlier", None)
                        msg = (
                            f"Outliers capped in {n_out} column(s)."
                            if o_action == "cap"
                            else f"Removed {before - after} outlier rows."
                        )
                        st.session_state["_omsg"] = ("run_outlier", msg)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            if st.session_state.get("_omsg", ("",))[0] == "run_outlier":
                st.success(st.session_state.pop("_omsg")[1])

            st.divider()
            st.write("**Validate Value Range**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                r1, r2, r3 = st.columns(3)
                with r1:
                    rng_min = st.number_input("Min", value=0.0, key="rng_min")
                with r2:
                    rng_max = st.number_input("Max", value=100.0, key="rng_max")
                with r3:
                    rng_act = st.selectbox(
                        "Action", ["flag", "remove"], key="rng_act",
                        help="Flag adds a boolean column showing which rows are in range. Remove deletes rows where the value falls outside the min or max.",
                    )
            with v2:
                n_rng = col_popover("range", num_cols)
            with v3:
                st.write("")
                st.write("")
                if st.button("Run", key="run_range", disabled=n_rng == 0,
                             type="primary" if n_rng else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        before = len(cdf)
                        tmp = cdf.copy()
                        for c in st.session_state.val_selected.get("range", []):
                            tmp = validate_range(tmp, c, rng_min, rng_max, rng_act)
                        st.session_state.current_df = tmp
                        after = len(tmp)
                        commit_history("Validate Range", _snap)
                        st.session_state.val_selected.pop("range", None)
                        msg = (
                            f"Range flagged across {n_rng} column(s)."
                            if rng_act == "flag"
                            else f"Removed {before - after} out of range rows."
                        )
                        st.session_state["_omsg"] = ("run_range", msg)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            if st.session_state.get("_omsg", ("",))[0] == "run_range":
                st.success(st.session_state.pop("_omsg")[1])
