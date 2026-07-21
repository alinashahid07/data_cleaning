
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

# Set page config
st.set_page_config(
    page_title="Data Cleaning Pipeline",
    page_icon="🧹",
    layout="wide"
)

# Validate input DataFrame
def checking_valid_input(
    df: pd.DataFrame
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")

# Drop duplicate rows
def drop_duplicate_rows(
    df: pd.DataFrame
):
    checking_valid_input(df)
    return df.drop_duplicates()

# Drop duplicate columns
def drop_duplicate_columns(
    df: pd.DataFrame
):
    checking_valid_input(df)
    return df.drop_duplicates().T.drop_duplicates().T

# Strip whitespaces from strings
def stripping_whitespace(
    df: pd.DataFrame
):
    checking_valid_input(df)
    return df.apply(
        lambda x: x.str.strip() 
        if x.dtype == "object" 
        else x
    )

# Remove specific character
def stripping_character_user_wants(
    df: pd.DataFrame,
    colname: str,
    ch_toremove: str
):
    checking_valid_input(df)
    if colname in df.columns:
        df[colname] = df[colname].astype(str).str.replace(
            ch_toremove, 
            "", 
            regex=False
        )
    return df

# App header layout
st.title("Data Cleaning Pipeline")
st.markdown("Upload a csv file to clean it using the pipeline.")
st.divider()

# File upload section
st.subheader("Upload your data file")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file containing missing values etc"
)

# Process uploaded file
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    try:
        df = pd.read_csv(uploaded_file)
        st.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Initialize session state
        if 'original_df' not in st.session_state:
            st.session_state.original_df = df.copy()
        if 'current_df' not in st.session_state:
            st.session_state.current_df = df.copy()

        # Preview dataset
        st.subheader("Data Preview")
        st.dataframe(
            st.session_state.current_df.head(10), 
            use_container_width=True
        )
        st.divider()

        # Cleaning controls
        st.subheader("Cleaning Data")
        st.markdown("Choose the cleaning operations to perform on your data")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Whitespace button
        with col1:
            if st.button("Remove Whitespaces", use_container_width=True):
                try:
                    st.session_state.current_df = stripping_whitespace(
                        st.session_state.current_df
                    )
                    st.success("Whitespace stripped!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
        # Row duplicate button
        with col2:
            if st.button("Drop Dup Rows", use_container_width=True):
                try:
                    b_rows = st.session_state.current_df.shape[0]
                    st.session_state.current_df = drop_duplicate_rows(
                        st.session_state.current_df
                    )
                    a_rows = st.session_state.current_df.shape[0]
                    st.success(f"Dropped {b_rows - a_rows} rows!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
        # Column duplicate button
        with col3:
            if st.button("Drop Dup Cols", use_container_width=True):
                try:
                    b_cols = st.session_state.current_df.shape[1]
                    st.session_state.current_df = drop_duplicate_columns(
                        st.session_state.current_df
                    )
                    a_cols = st.session_state.current_df.shape[1]
                    st.success(f"Dropped {b_cols - a_cols} columns!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.warning("Please upload a file to proceed")