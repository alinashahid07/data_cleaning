import pandas as pd
import streamlit as st

from ai_insights import render_summary

def render(tab, cdf, stats, orig_stats, file_id=None, df_key=""):
    with tab:
        # large file warning banner shown when load_file detected downcasting was applied
        large_warn = st.session_state.get("_large_file_warning")
        if large_warn:
            st.warning(large_warn)

        # sampled analysis note shown on large files so user knows stats are estimated
        if stats.get("is_large"):
            st.info(
                f"This file has {stats['rows']:,} rows. "
                "Recommendations, correlation, and quality score are computed on a 20,000 row sample "
                "for performance. All cleaning operations still run on the full dataset."
            )

        if file_id:
            render_summary(cdf, file_id)
            st.divider()
        st.subheader("Data Statistics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", f"{stats['rows']:,}", delta=stats["rows"] - orig_stats["rows"], delta_color="inverse")
            st.metric("Columns", stats["columns"], delta=stats["columns"] - orig_stats["columns"], delta_color="inverse")
        with c2:
            st.metric("Missing Cells", f"{stats['missing_cells']:,}", delta=stats["missing_cells"] - orig_stats["missing_cells"], delta_color="inverse")
            st.metric("Duplicate Rows", f"{stats['duplicate_rows']:,}", delta=stats["duplicate_rows"] - orig_stats["duplicate_rows"], delta_color="inverse")
        with c3:
            st.metric("Numeric Columns", stats["numeric_cols"])
            st.metric("Categorical Columns", stats["categorical_cols"])

        st.divider()

        # data preview
        st.subheader("Data Preview")
        total_rows = len(cdf)
        # cap the preview slider max at 200 on large files to avoid rendering lag
        if stats.get("is_large"):
            max_preview = 200
            default_preview = 20
        else:
            max_preview = min(50, total_rows)
            default_preview = min(10, total_rows)
        n_prev = st.slider(
            "Rows to display",
            min(5, total_rows),
            max(min(5, total_rows), max_preview),
            default_preview,
            key="prev_slider",
        )
        if total_rows < 50:
            st.caption(f"File has {total_rows} rows.")
        st.dataframe(cdf.head(n_prev), use_container_width=True)

        st.divider()

        # column info
        with st.expander("Column Types and Info", expanded=False):
            st.dataframe(
                pd.DataFrame({
                    "Column": cdf.columns,
                    "Type": cdf.dtypes.values,
                    "Non-Null": cdf.count().values,
                    "Null": cdf.isna().sum().values,
                    "Unique": cdf.nunique().values,
                }),
                use_container_width=True,
            )
        st.caption("Download and reset options are in the History and Export tab.")
