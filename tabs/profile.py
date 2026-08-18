import pandas as pd
import streamlit as st
from cache import (
    get_bar_data,
    get_column_profile,
    get_correlation_data,
    get_histogram_data,
    get_missing_heatmap_data,
)
from state import show_msg

# histogram with a kde overlay and iqr fence lines
def _render_histogram(col, data):
    if not data:
        st.caption("No numeric data to plot.")
        return
    bin_df = pd.DataFrame({"x": data["bin_centers"], "count": data["counts"]})
    kde_df = pd.DataFrame({"x": data["kde_x"], "density": data["kde_y"]})
    bin_width = data["bin_centers"][1] - data["bin_centers"][0] if len(data["bin_centers"]) > 1 else 1
    total = sum(data["counts"])
    scale_factor = total * bin_width
    kde_df["scaled_density"] = [v * scale_factor for v in kde_df["density"]]
    mean_val = data["mean"]
    median_val = data["median"]
    lower_fence = data["lower_fence"]
    upper_fence = data["upper_fence"]
    n_outliers = data["n_outliers"]

    bar_layer = {
        "mark": {"type": "bar", "color": "#1f77b4", "opacity": 0.7},
        "data": {"values": bin_df.to_dict("records")},
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": col},
            "y": {"field": "count", "type": "quantitative", "title": "Count"},
            "tooltip": [
                {"field": "x", "type": "quantitative", "title": "Value", "format": ".3f"},
                {"field": "count", "type": "quantitative", "title": "Count"},
            ],
        },
    }
    kde_layer = {
        "mark": {"type": "line", "color": "#ff7f0e", "strokeWidth": 2},
        "data": {"values": kde_df.to_dict("records")},
        "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "scaled_density", "type": "quantitative"},
        },
    }
    rule_data = [
        {"value": mean_val, "label": "Mean"},
        {"value": median_val, "label": "Median"},
        {"value": lower_fence, "label": "IQR lower fence"},
        {"value": upper_fence, "label": "IQR upper fence"},
    ]
    rule_layer = {
        "mark": {"type": "rule", "strokeDash": [4, 4], "strokeWidth": 1.5},
        "data": {"values": rule_data},
        "encoding": {
            "x": {"field": "value", "type": "quantitative"},
            "color": {
                "field": "label",
                "type": "nominal",
                "scale": {
                    "domain": ["Mean", "Median", "IQR lower fence", "IQR upper fence"],
                    "range": ["#d62728", "#2ca02c", "#9467bd", "#9467bd"],
                },
                "legend": {"title": "Reference lines"},
            },
            "tooltip": [
                {"field": "label", "type": "nominal", "title": "Line"},
                {"field": "value", "type": "quantitative", "title": "Value", "format": ".3f"},
            ],
        },
    }
    chart = {
        "layer": [bar_layer, kde_layer, rule_layer],
        "resolve": {"scale": {"y": "shared"}},
        "width": "container",
        "height": 280,
    }
    st.vega_lite_chart(chart, use_container_width=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Mean", f"{mean_val:.4g}")
    mc2.metric("Median", f"{median_val:.4g}")
    mc3.metric("IQR fences", f"{lower_fence:.3g} / {upper_fence:.3g}")
    mc4.metric("Outliers (IQR)", n_outliers)

# horizontal bar chart for categorical value counts
def _render_bar_chart(col, data):
    if not data:
        st.caption("No categorical data to plot.")
        return
    bar_df = pd.DataFrame({
        "label": data["labels"],
        "count": data["counts"],
        "pct": data["pct"],
    })
    chart = {
        "mark": {"type": "bar", "color": "#1f77b4"},
        "data": {"values": bar_df.to_dict("records")},
        "encoding": {
            "y": {
                "field": "label",
                "type": "nominal",
                "sort": "-x",
                "title": col,
                "axis": {"labelLimit": 180},
            },
            "x": {"field": "count", "type": "quantitative", "title": "Count"},
            "tooltip": [
                {"field": "label", "type": "nominal", "title": "Value"},
                {"field": "count", "type": "quantitative", "title": "Count"},
                {"field": "pct", "type": "quantitative", "title": "%", "format": ".1f"},
            ],
        },
        "width": "container",
        "height": max(140, min(40 * len(data["labels"]), 400)),
    }
    st.vega_lite_chart(chart, use_container_width=True)
    if data["shown"] < data["unique"]:
        st.caption(
            f"Showing top {data['shown']} of {data['unique']} unique values "
            f"across {data['total']} non-null rows."
        )
    else:
        st.caption(f"{data['unique']} unique values across {data['total']} non-null rows.")

# grid colored by whether each cell is null or present
def _render_missing_heatmap(data):
    if not data:
        st.success("No missing values found in this dataset.")
        return
    columns = data["columns"]
    matrix = data["matrix"]
    null_pct = data["null_pct"]
    n_rows_shown = data["n_rows_shown"]
    total_rows = data["total_rows"]

    rows_list = []
    for row_idx, row in enumerate(matrix):
        for col_idx, is_null in enumerate(row):
            rows_list.append({
                "row": row_idx,
                "column": columns[col_idx],
                "status": "Missing" if is_null else "Present",
            })

    chart = {
        "mark": "rect",
        "data": {"values": rows_list},
        "encoding": {
            "x": {
                "field": "row",
                "type": "ordinal",
                "title": f"Row index (sample of {n_rows_shown})",
                "axis": {"labels": False, "ticks": False},
            },
            "y": {
                "field": "column",
                "type": "nominal",
                "title": "Column",
                "sort": columns,
            },
            "color": {
                "field": "status",
                "type": "nominal",
                "scale": {
                    "domain": ["Present", "Missing"],
                    "range": ["#1f77b4", "#d62728"],
                },
                "legend": {"title": "Value status"},
            },
            "tooltip": [
                {"field": "column", "type": "nominal", "title": "Column"},
                {"field": "status", "type": "nominal", "title": "Status"},
                {"field": "row", "type": "ordinal", "title": "Sample row"},
            ],
        },
        "width": "container",
        "height": max(120, 28 * len(columns)),
    }
    st.vega_lite_chart(chart, use_container_width=True)
    if total_rows > n_rows_shown:
        st.caption(f"Heatmap shows a random sample of {n_rows_shown} rows out of {total_rows} total.")
    pct_items = sorted(null_pct.items(), key=lambda x: x[1], reverse=True)
    pct_cols = st.columns(min(len(pct_items), 4))
    for i, (col, pct) in enumerate(pct_items):
        pct_cols[i % 4].metric(col, f"{pct}% missing")

# symmetric correlation matrix as a color encoded rect heatmap
def _render_correlation_heatmap(data):
    if not data:
        st.caption("No correlation data to display.")
        return
    col_order = data["col_order"]
    n_cols = data["n_cols"]

    chart = {
        "mark": "rect",
        "data": {"values": data["rows"]},
        "encoding": {
            "x": {
                "field": "col_a",
                "type": "nominal",
                "sort": col_order,
                "title": None,
                "axis": {"labelAngle": -40, "labelLimit": 120},
            },
            "y": {
                "field": "col_b",
                "type": "nominal",
                "sort": col_order,
                "title": None,
                "axis": {"labelLimit": 120},
            },
            "color": {
                "field": "value",
                "type": "quantitative",
                "scale": {
                    "domain": [-1, 0, 1],
                    "range": ["#d62728", "#f5f5f5", "#1f77b4"],
                },
                "legend": {"title": "Correlation", "gradientLength": 120},
            },
            "tooltip": [
                {"field": "col_a", "type": "nominal", "title": "Column A"},
                {"field": "col_b", "type": "nominal", "title": "Column B"},
                {"field": "value", "type": "quantitative", "title": "Correlation", "format": ".3f"},
            ],
        },
        "width": "container",
        "height": max(200, 36 * n_cols),
    }
    text_layer = {
        "mark": {"type": "text", "fontSize": 11},
        "data": {"values": data["rows"]},
        "encoding": {
            "x": {"field": "col_a", "type": "nominal", "sort": col_order},
            "y": {"field": "col_b", "type": "nominal", "sort": col_order},
            "text": {"field": "value", "type": "quantitative", "format": ".2f"},
            "color": {
                "condition": {"test": "abs(datum.value) > 0.6", "value": "white"},
                "value": "#333333",
            },
        },
    }
    # only show text labels when the grid is small enough to be readable
    if n_cols <= 12:
        final_chart = {
            "layer": [chart, text_layer],
            "width": "container",
            "height": max(200, 36 * n_cols),
        }
    else:
        final_chart = chart
    st.vega_lite_chart(final_chart, use_container_width=True)

    # surface the strongest off diagonal pairs as a quick summary
    off_diag = [r for r in data["rows"] if r["col_a"] != r["col_b"]]
    if off_diag:
        seen = set()
        unique_pairs = []
        for r in sorted(off_diag, key=lambda x: abs(x["value"]), reverse=True):
            pair = tuple(sorted([r["col_a"], r["col_b"]]))
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(r)
        top = unique_pairs[:3]
        if top:
            st.caption(
                "Strongest pairs: "
                + "   |   ".join(
                    f"{r['col_a']} vs {r['col_b']} ({r['value']:+.2f})"
                    for r in top
                )
            )

def render(tab, cdf, mode="Simple"):
    with tab:
        show_msg()

        df_key = st.session_state.get("current_df_key", "")

        # column profiler
        st.subheader("Column Profiler")
        st.caption("Per-column stats including min, max, mean, median, std, skewness, and sample values.")
        profile = get_column_profile(df_key, cdf)
        st.dataframe(profile, use_container_width=True, hide_index=True)
        worst = profile[profile["Null"] > 0].sort_values("Null", ascending=False)
        if not worst.empty:
            st.caption(
                f"{len(worst)} column(s) have missing values. "
                f"Worst: **{worst.iloc[0]['Column']}** ({worst.iloc[0]['Null %']} missing)"
            )

        st.divider()

        # distribution charts
        st.subheader("Distribution Charts")
        st.caption(
            "Select a column to see its distribution. "
            "Numeric columns show a histogram with a KDE curve and IQR outlier fences. "
            "Categorical columns show a value frequency bar chart."
        )
        all_cols = list(cdf.columns)
        num_cols = list(cdf.select_dtypes(include="number").columns)
        cat_cols = list(cdf.select_dtypes(include="object").columns)
        dist_col = st.selectbox("Column to plot", all_cols, key="dist_col")
        if dist_col in num_cols:
            n_bins = st.slider("Number of bins", 5, 100, 30, key="hist_bins")
            hist_data = get_histogram_data(df_key, cdf[dist_col], n_bins=n_bins)
            _render_histogram(dist_col, hist_data)
        elif dist_col in cat_cols:
            top_n = st.slider("Max categories to show", 5, 50, 20, key="bar_topn")
            bar_data = get_bar_data(df_key, cdf[dist_col], top_n=top_n)
            _render_bar_chart(dist_col, bar_data)
        else:
            st.caption(f"Column type {cdf[dist_col].dtype} is not supported for distribution plots.")

        st.divider()

        # missing value heatmap
        st.subheader("Missing Value Heatmap")
        st.caption(
            "Each row is a data row and each column strip is a dataset column. "
            "Red cells are missing values. Only columns with at least one missing value are shown."
        )
        heatmap_data = get_missing_heatmap_data(df_key, cdf)
        _render_missing_heatmap(heatmap_data)

        st.divider()

        # correlation heatmap
        st.subheader("Correlation Heatmap")
        if mode != "Advanced":
            st.caption("Switch to Advanced mode in the sidebar to unlock the correlation heatmap.")
        else:
            st.caption(
                "Shows how strongly each pair of numeric columns moves together. "
                "Values close to 1 or -1 are strongly correlated. "
                "Values near 0 have little linear relationship. "
                "Pairs near 1.0 may be redundant columns worth dropping."
            )
            num_cols_for_corr = list(cdf.select_dtypes(include="number").columns)
            if len(num_cols_for_corr) < 2:
                st.caption("Need at least two numeric columns to compute correlations.")
            else:
                corr_method = st.selectbox(
                    "Method",
                    ["pearson", "spearman", "kendall"],
                    key="corr_method",
                    help=(
                        "Pearson measures linear relationships and is the standard choice. "
                        "Spearman is rank based and handles non-linear relationships and outliers better. "
                        "Kendall is also rank based and more robust on small datasets."
                    ),
                )
                corr_data = get_correlation_data(df_key, cdf, method=corr_method)
                _render_correlation_heatmap(corr_data)

        st.divider()

        # before and after comparison
        st.subheader("Before and After Comparison")
        original_df = st.session_state.original_df
        shared = [c for c in original_df.columns if c in cdf.columns]
        if shared:
            ba_col = st.selectbox("Select column to compare", shared, key="ba_col")
            ba_n = st.slider("Rows to preview", 5, 50, 10, key="ba_n")
            orig_s = original_df[ba_col].head(ba_n).reset_index(drop=True)
            curr_s = cdf[ba_col].head(ba_n).reset_index(drop=True)
            ml = min(len(orig_s), len(curr_s))
            orig_s, curr_s = orig_s.iloc[:ml], curr_s.iloc[:ml]
            changed = orig_s.fillna("").astype(str) != curr_s.fillna("").astype(str)
            st.dataframe(
                pd.DataFrame({
                    "Original": orig_s,
                    "Current": curr_s,
                    "Changed": changed.map({True: "yes", False: ""}),
                }),
                use_container_width=True,
            )
            n_ch = int(changed.sum())
            st.caption(
                f"{n_ch} row(s) changed in this preview window."
                if n_ch
                else "No differences found in this preview window."
            )
