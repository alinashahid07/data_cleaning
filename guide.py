import streamlit as st

GUIDE_ITEMS = [
    {
        "tab": "Upload",
        "icon": "upload",
        "title": "Upload your file",
        "desc": (
            "Drop in a CSV or Excel file. "
            "If your Excel has multiple sheets you will be asked to pick one. "
            "The app resets automatically when you swap to a different file."
        ),
        "tips": [
            "CSV and XLSX are both supported.",
            "Files stay in your browser session only.",
            "Swap files any time, state resets cleanly.",
        ],
    },
    {
        "tab": "Overview",
        "icon": "bar_chart",
        "title": "Check your data quality",
        "desc": (
            "See your overall quality score, row and column counts, "
            "missing cells, and duplicate rows at a glance. "
            "Read this before touching anything else."
        ),
        "tips": [
            "Quality score is a weighted mix of completeness and consistency.",
            "Duplicate count includes fully identical rows only.",
            "The live preview shows the first rows of your current working copy.",
        ],
    },
    {
        "tab": "Recommendations",
        "icon": "auto_fix_high",
        "title": "Let the app find issues",
        "desc": (
            "Every issue in your data is flagged here automatically. "
            "Fix them one by one or hit Auto-Fix All to resolve everything in one click. "
            "Always start here before doing anything manually."
        ),
        "tips": [
            "Auto-Fix All applies all safe fixes in the recommended order.",
            "Each fix is logged in History so you can undo it.",
            "Run this again after manual cleaning to catch anything new.",
        ],
    },
    {
        "tab": "Clean",
        "icon": "cleaning_services",
        "title": "Clean manually",
        "desc": (
            "Handle anything the recommendations missed. "
            "Fix columns storing numbers as text, split columns that hold two values, "
            "or swap out specific bad values with Find and Replace."
        ),
        "tips": [
            "Data Type Guesser converts text columns that look numeric or date-like.",
            "Split Column is useful for things like a full name in one column.",
            "Find and Replace supports exact match and case-insensitive mode.",
        ],
    },
    {
        "tab": "Validate",
        "icon": "rule",
        "title": "Validate formats and outliers",
        "desc": (
            "Check correctness rather than cleanliness. "
            "Flag invalid emails, standardize phone numbers, normalize date formats, "
            "and cap extreme outliers. Run this after cleaning, not before."
        ),
        "tips": [
            "Email validation checks format only, not whether the address exists.",
            "Outlier capping uses the IQR method by default.",
            "Date normalization converts mixed formats to a single consistent one.",
        ],
    },
    {
        "tab": "Profile",
        "icon": "insights",
        "title": "Explore visually",
        "desc": (
            "See distribution charts, a missing value heatmap, "
            "and a before vs after comparison for every column. "
            "Switch to Advanced mode in the sidebar to unlock the correlation heatmap."
        ),
        "tips": [
            "Before vs after shows what changed since you loaded the file.",
            "The heatmap makes sparse columns obvious at a glance.",
            "Advanced mode also unlocks extra profiling statistics.",
        ],
    },
    {
        "tab": "History & Export",
        "icon": "download",
        "title": "Export when ready",
        "desc": (
            "Once your quality score looks good, download your cleaned data. "
            "Grab a PDF report to share with others, "
            "or save your pipeline as JSON to replay the same steps on a new file later."
        ),
        "tips": [
            "CSV and Excel download options are both available.",
            "The PDF report includes a before vs after summary.",
            "The JSON pipeline lets you automate the same cleaning on future files.",
        ],
    },
]

def _init_guide():
    if "guide_checked" not in st.session_state:
        st.session_state.guide_checked = set()

def render(tab):
    _init_guide()
    checked = st.session_state.guide_checked

    with tab:
        st.markdown("## Workflow Guide")
        st.caption(
            "Work through each step at your own pace. "
            "Tick items off as you go. You can come back here any time."
        )

        done = len(checked)
        total = len(GUIDE_ITEMS)
        pct = int(done / total * 100)

        # progress bar
        st.markdown(f"""
<div style="margin:16px 0 24px 0;">
  <div style="display:flex;justify-content:space-between;
              font-size:0.75rem;color:#888;margin-bottom:6px;">
    <span>{done} of {total} steps marked done</span>
    <span>{pct}%</span>
  </div>
  <div style="background:#1a1a2e;border-radius:99px;height:6px;overflow:hidden;">
    <div style="background:#1f77b4;width:{pct}%;height:100%;
                border-radius:99px;transition:width .3s;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        for i, item in enumerate(GUIDE_ITEMS):
            key = f"guide_check_{i}"
            is_checked = i in checked
            col_check, col_content = st.columns([0.06, 0.94])

            with col_check:
                ticked = st.checkbox(
                    label="",
                    value=is_checked,
                    key=key,
                    label_visibility="collapsed",
                )
                if ticked and i not in checked:
                    checked.add(i)
                    st.session_state.guide_checked = checked
                    st.rerun()
                elif not ticked and i in checked:
                    checked.discard(i)
                    st.session_state.guide_checked = checked
                    st.rerun()

            with col_content:
                title_style = (
                    "text-decoration:line-through;color:#555;"
                    if is_checked else "color:#fff;"
                )
                tab_badge = (
                    f'<span style="font-size:0.65rem;font-weight:700;'
                    f'color:#1f77b4;text-transform:uppercase;'
                    f'letter-spacing:1px;margin-left:8px;">'
                    f'{item["tab"]} tab</span>'
                )
                st.markdown(
                    f'<p style="margin:0 0 4px 0;font-size:0.95rem;'
                    f'font-weight:700;{title_style}">'
                    f'{item["title"]}{tab_badge}</p>',
                    unsafe_allow_html=True,
                )
                st.caption(item["desc"])
                with st.expander("Tips"):
                    for tip in item["tips"]:
                        st.markdown(f"- {tip}")

            st.divider()

        # reset button at the bottom
        if checked:
            if st.button("Reset all checkboxes", key="guide_reset"):
                st.session_state.guide_checked = set()
                st.rerun()
