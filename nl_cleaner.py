import re
import traceback
import pandas as pd
import streamlit as st
from pipeline import commit_history, snapshot

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# read api key from env or secrets
def _get_api_key():
    import os
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        key = st.secrets.get("GEMINI_API_KEY", "")
    return key

# build prompt for gemini
def _build_prompt(df, instruction):
    col_info = "\n".join(f"  {col} ({df[col].dtype})" for col in df.columns)
    sample = df.head(5).to_string(index=False)
    return f"""You are a pandas code assistant. The user has a dataframe called `df`.

Column names and types:
{col_info}

First 5 rows:
{sample}

User instruction: {instruction}

Respond with ONLY a JSON object in this exact format, nothing else:
{{
  "code": "df = df.dropna(subset=['Age'])",
  "explanation": "This will remove all rows where the Age column is empty."
}}

Rules for the code field:
- Write clean pandas code, as many lines as needed
- Must assign back to df or modify df in place
- No imports
- No comments
- No print statements
- If the instruction is impossible or unclear set code to CANNOT_DO and explanation to why

Do not include any text outside the JSON object."""

# call gemini api
def _call_gemini(instruction, df):
    import json
    import urllib.request

    api_key = _get_api_key()
    if not api_key:
        return None, None, "GEMINI_API_KEY not set. Add it to your .env file."

    url = GEMINI_API_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": _build_prompt(df, instruction)}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            msg = json.loads(error_body).get("error", {}).get("message", error_body)
        except Exception:
            msg = error_body
        return None, None, f"Gemini API error: {msg}"
    except Exception as e:
        return None, None, f"Request failed: {e}"

    # parse model response
    try:
        raw = body["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        code = parsed.get("code", "").strip()
        explanation = parsed.get("explanation", "").strip()
    except Exception:
        return None, None, "Gemini returned an unexpected format. Try rephrasing your instruction."

    if "CANNOT_DO" in code:
        return None, None, explanation or "Gemini could not understand that instruction. Try rephrasing."

    return code, explanation, None

# run generated code against a copy
def _run_code(df, code):
    local_vars = {"df": df.copy(), "pd": pd}
    try:
        exec(code, {}, local_vars)  # noqa: S102
        result = local_vars["df"]
        if not isinstance(result, pd.DataFrame):
            return df, "code did not produce a dataframe"
        return result, None
    except Exception:
        return df, traceback.format_exc(limit=3)

def render_nl_cleaner(cdf):
    api_key = _get_api_key()
    if not api_key:
        st.warning(
            "Gemini API key not found. "
            "Add GEMINI_API_KEY to your .env file to use this feature."
        )
        return

    # show previous result first
    if st.session_state.get("_nl_success"):
        st.success(st.session_state.pop("_nl_success"))
    if st.session_state.get("_nl_error"):
        st.error(st.session_state.pop("_nl_error"))

    instruction = st.text_area(
        "Describe what you want to do",
        placeholder=(
            "e.g. drop rows where Age is empty\n"
            "e.g. fill missing City with Unknown\n"
            "e.g. convert the Price column to a number\n"
            "e.g. remove all duplicate rows"
        ),
        key="nl_instruction",
        height=100,
        help="Write in plain English. AI will generate the pandas code for you.",
    )

    if st.button("Generate code", key="nl_generate", use_container_width=True, type="primary"):
        if not instruction.strip():
            st.warning("Type an instruction first.")
            return
        with st.spinner("Asking Gemini..."):
            code, explanation, err = _call_gemini(instruction.strip(), cdf)
        if err:
            st.error(err)
            return
        st.session_state["nl_pending_code"] = code
        st.session_state["nl_pending_explanation"] = explanation
        st.session_state["nl_pending_instruction"] = instruction.strip()
        st.rerun()

    code = st.session_state.get("nl_pending_code")
    explanation = st.session_state.get("nl_pending_explanation")
    if not code:
        return

    st.divider()
    # explain before applying
    st.info(f"Verify that this matches what you want: {explanation}")

    # editable code box
    edited_code = st.text_area(
        "Code (you can edit this before applying)",
        value=code,
        key="nl_code_editor",
        height=150,
    )

    col_apply, col_cancel = st.columns(2)
    with col_apply:
        if st.button("Apply", key="nl_apply", use_container_width=True, type="primary"):
            new_df, run_err = _run_code(cdf, edited_code)
            if run_err:
                st.session_state["_nl_error"] = (
                    f"Something went wrong while running the code. "
                    f"Try rephrasing your instruction or editing the code above.\n\n{run_err}"
                )
                st.session_state.pop("nl_pending_code", None)
                st.session_state.pop("nl_pending_explanation", None)
                st.session_state.pop("nl_pending_instruction", None)
                st.rerun()
                return
            _snap = snapshot()
            st.session_state.current_df = new_df
            commit_history(
                f"AI Clean: {st.session_state.get('nl_pending_instruction', 'natural language clean')}",
                _snap,
            )
            st.session_state["_nl_success"] = (
                f"Applied successfully. "
                f"Rows: {len(cdf)} to {len(new_df)}. "
                f"Columns: {len(cdf.columns)} to {len(new_df.columns)}."
            )
            st.session_state.pop("nl_pending_code", None)
            st.session_state.pop("nl_pending_explanation", None)
            st.session_state.pop("nl_pending_instruction", None)
            st.rerun()
    with col_cancel:
        if st.button("Cancel", key="nl_cancel", use_container_width=True):
            st.session_state.pop("nl_pending_code", None)
            st.session_state.pop("nl_pending_explanation", None)
            st.session_state.pop("nl_pending_instruction", None)
            st.rerun()
