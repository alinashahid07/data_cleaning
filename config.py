import streamlit as st

# page setup
def apply_page_config():
    st.set_page_config(
        page_title="Advanced Data Cleaning Pipeline",
        page_icon="*",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# custom css
def inject_css():
    st.markdown("""
<style>
.stButton > button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
    font-weight: 500;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0px !important;
    background-color: #0e1117;
    padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 18px 32px !important;
    color: #aaaaaa !important;
    border-radius: 0 !important;
    border: none !important;
    background: transparent !important;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255,255,255,0.05) !important;
}
.stTabs [aria-selected="true"] {
    color: #1f77b4 !important;
    border-bottom: 3px solid #1f77b4 !important;
    background: transparent !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #1f77b4 !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: transparent !important;
}
.stTabs {
    margin-top: -1rem;
}
</style>
""", unsafe_allow_html=True)
