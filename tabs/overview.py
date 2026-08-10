import pandas as pd
import streamlit as st

def render(tab, cdf, stats, orig_stats):
    with tab:
        st.subheader("Data Statistics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", stats["rows"], delta=stats["rows"] - orig_stats["rows"], delta_color="inverse")
            st.metric("Columns", stats["columns"], delta=stats["columns"] - orig_stats["columns"], delta_color="inverse")
        with c2:
            st.metric("Missing Cells", stats["missing_cells"], delta=stats["missing_cells"] - orig_stats["missing_cells"], delta_color="inverse")
            st.metric("Duplicate Rows", stats["duplicate_rows"], delta=stats["duplicate_rows"] - orig_stats["duplicate_rows"], delta_color="inverse")
        with c3:
            st.metric("Numeric Columns", stats["numeric_cols"])
            st.metric("Categorical Columns", stats["categorical_cols"])

        st.divider()

        # data preview
        st.subheader("Data Preview")
        total_rows = len(cdf)
        max_preview = min(50, total_rows)
        default_preview = min(10, total_rows)
        n_prev = st.slider("Rows to display", min(5, total_rows), max_preview, default_preview, key="prev_slider")
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
