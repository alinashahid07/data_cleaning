import io
import pandas as pd
import streamlit as st
from pipeline import (
    build_pipeline_script,
    commit_history,
    export_pipeline_json,
    import_pipeline_json,
    undo_last,
)
from reporting import build_report_pdf

def _render_reset():
    st.subheader("Reset Data")
    st.warning("This will discard all cleaning and restore the original uploaded file.")
    if st.button("Reset to Original Data", key="reset_orig", use_container_width=True):
        st.session_state.current_df = st.session_state.original_df.copy()
        st.session_state.selected_columns = {}
        st.session_state.history = []
        st.session_state["history_len"] = st.session_state.get("history_len", 0) + 1
        st.session_state.last_success_msg = "Data reset to original."
        st.rerun()

def _render_download(cdf):
    st.subheader("Download Cleaned Data")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download as CSV",
            data=cdf.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
            key="dl_csv",
            use_container_width=True,
        )
    with d2:
        # only regenerate excel bytes when the df actually changes
        # history_len bumps on every commit_history call, a cheap cache key
        excel_cache_key = st.session_state.get("history_len", 0)
        if st.session_state.get("_excel_cache_key") != excel_cache_key:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                cdf.to_excel(w, index=False, sheet_name="Cleaned Data")
                st.session_state.original_df.to_excel(w, index=False, sheet_name="Original Data")
            st.session_state["_excel_bytes"] = buf.getvalue()
            st.session_state["_excel_cache_key"] = excel_cache_key
        st.download_button(
            "Download as Excel",
            data=st.session_state["_excel_bytes"],
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx",
            use_container_width=True,
        )

def _render_history(hist):
    st.subheader("Cleaning History")
    if not hist:
        st.caption("No operations recorded yet. Every cleaning action is saved here.")
        return
    st.caption(f"{len(hist)} operation(s) recorded. Max 20 steps kept.")
    for i, step in enumerate(reversed(hist)):
        st.write(
            f"**{len(hist) - i}.** {step['label']}, "
            f"{step['df'].shape[0]} rows by {step['df'].shape[1]} cols"
        )
    st.write("")
    h1, h2 = st.columns(2)
    with h1:
        if st.button("Undo Last Step", key="undo_btn", type="primary", use_container_width=True):
            label = undo_last()
            if label:
                st.session_state.last_success_msg = f"Undone: {label}"
            st.rerun()
    with h2:
        if st.button("Clear History", key="clear_hist", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_success_msg = "History cleared."
            st.rerun()

def _render_pipeline_json(hist, settings):
    st.subheader("Save and Reload Pipeline")
    st.caption(
        "Save your cleaning steps as a JSON file. "
        "Upload that file later on any dataset to replay the exact same steps automatically."
    )
    save_col, load_col = st.columns(2)
    with save_col:
        st.write("**Save current pipeline**")
        if not hist:
            st.caption("Run at least one cleaning operation to enable saving.")
        else:
            json_bytes = export_pipeline_json(hist).encode("utf-8")
            st.download_button(
                "Download pipeline.json",
                data=json_bytes,
                file_name="pipeline.json",
                mime="application/json",
                key="dl_json",
                use_container_width=True,
            )
            with st.expander("Preview JSON", expanded=False):
                st.code(export_pipeline_json(hist), language="json")

    with load_col:
        st.write("**Reload a saved pipeline**")
        st.caption(
            "Upload a pipeline.json file to replay its steps on the current dataset. "
            "Steps that require a specific column name that no longer exists will be skipped."
        )
        uploaded_json = st.file_uploader(
            "Upload pipeline.json",
            type=["json"],
            key="json_uploader",
            label_visibility="collapsed",
        )
        # read bytes immediately into session state so the buffer is not
        # exhausted by the time the button click triggers a second render pass
        if uploaded_json is not None:
            file_key = getattr(uploaded_json, "file_id", uploaded_json.name)
            if st.session_state.get("_json_file_key") != file_key:
                st.session_state["_json_bytes"] = uploaded_json.read()
                st.session_state["_json_file_key"] = file_key

        if uploaded_json is not None and st.session_state.get("_json_bytes"):
            if st.button("Apply pipeline", key="apply_json", type="primary", use_container_width=True):
                try:
                    json_content = st.session_state["_json_bytes"]
                    original = st.session_state.current_df.copy()
                    result, applied, skipped = import_pipeline_json(
                        json_content,
                        st.session_state.current_df,
                        settings,
                    )
                    st.session_state.current_df = result
                    for label in applied:
                        commit_history(f"Replayed: {label}", original)
                    msg_parts = [f"{len(applied)} step(s) applied."]
                    if skipped:
                        msg_parts.append(
                            f"{len(skipped)} step(s) skipped (require manual column selection): "
                            + ", ".join(skipped[:5])
                            + ("..." if len(skipped) > 5 else "")
                        )
                    st.session_state.last_success_msg = " ".join(msg_parts)
                    st.session_state.pop("_json_bytes", None)
                    st.session_state.pop("_json_file_key", None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not apply pipeline: {e}")

def _render_python_export(hist):
    st.subheader("Export as Python Script")
    st.caption("Download your cleaning steps as a standalone Python script.")
    if not hist:
        st.caption("No steps recorded yet. Run some cleaning operations first.")
        return
    script = build_pipeline_script(hist)
    st.download_button(
        "Download pipeline.py",
        data=script.encode("utf-8"),
        file_name="pipeline.py",
        mime="text/x-python",
        key="dl_pipeline",
        use_container_width=True,
    )
    with st.expander("Preview script", expanded=False):
        st.code(script, language="python")

def _render_report_pdf(cdf, hist, filename):
    st.subheader("Export Cleaning Report")
    st.caption(
        "Generates a PDF report with a before and after summary, "
        "column profiles, missing value breakdown, cleaning steps, "
        "and a sample of the cleaned data."
    )
    if not hist:
        st.caption("Run at least one cleaning operation to generate a report.")
        return

    if st.button("Generate Report", key="gen_pdf", type="primary", use_container_width=True):
        try:
            with st.spinner("Building report..."):
                pdf_bytes = build_report_pdf(
                    original_df=st.session_state.original_df,
                    cleaned_df=cdf,
                    history=hist,
                    filename=filename,
                )
            st.session_state["_pdf_bytes"] = pdf_bytes
            st.session_state["_pdf_ready"] = True
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")
            st.session_state["_pdf_ready"] = False

    if st.session_state.get("_pdf_ready") and st.session_state.get("_pdf_bytes"):
        st.success("Report ready.")
        st.download_button(
            "Download cleaning_report.pdf",
            data=st.session_state["_pdf_bytes"],
            file_name="cleaning_report.pdf",
            mime="application/pdf",
            key="dl_pdf",
            use_container_width=True,
        )

def render(tab, cdf, settings):
    filename = settings.get("filename", "dataset")
    with tab:
        _render_reset()
        st.divider()
        _render_download(cdf)
        st.divider()
        _render_report_pdf(cdf, st.session_state.get("history", []), filename)
        st.divider()
        _render_history(st.session_state.get("history", []))
        st.divider()
        _render_pipeline_json(st.session_state.get("history", []), settings)
        st.divider()
        _render_python_export(st.session_state.get("history", []))
