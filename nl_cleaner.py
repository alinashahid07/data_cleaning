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

    # build a sample that shows raw values so gemini can spot dirty data
    dirty_hints = []
    for col in df.columns:
        non_null = df[col].dropna().astype(str)
        if non_null.empty:
            continue
        examples = non_null.head(5).tolist()
        dirty_hints.append(f"  {col}: {examples}")
    dirty_hints_str = "\n".join(dirty_hints)

    return f"""You are a careful data analyst and pandas code assistant.
The user has a dataframe called `df`.

Column names and types:
{col_info}

First 5 rows (formatted):
{sample}

Raw sample values per column (first 5 non-null):
{dirty_hints_str}

User instruction: {instruction}

--- APP FEATURE MAP (use this to guide the user to the right place) ---
The app the user is working in has these tabs and features. When telling the user
to fix a pre-condition, reference the EXACT tab name and feature name from this list:

  RECOMMENDATIONS TAB:
    - "Convert Currency"   -> columns containing currency symbols like $, EUR, GBP
    - "Convert Percentage" -> columns containing percentage signs like %
    - "Convert Units"      -> columns containing unit suffixes like kg, km, hrs, lbs
    - "Convert Duration"   -> columns containing duration values like 1h30m, 2:30
    - "Strip Whitespace"   -> columns with leading or trailing spaces
    - "Clean String Edges" -> columns with special characters at start or end of values
    - "Handle Missing Values" -> columns with null or empty values, auto fill using KNN or mode
    - "Drop Duplicates"    -> duplicate rows or duplicate columns

  CLEAN TAB:
    - "Smart Column Cleaner"   -> bulk auto convert currency, percentage, unit, duration columns
    - "Handle Missing Values"  -> fill missing values, same as Recommendations but manual trigger
    - "Column Type Override"   -> manually cast a column to string, integer, float, datetime, boolean, category
    - "Data Type Guesser"      -> auto detect and suggest correct types for all columns
    - "Find and Replace"       -> find and replace text values in a column
    - "Split Column"           -> split one column into multiple using a delimiter
    - "Merge Columns"          -> combine two or more columns into one
    - "Rename Columns"         -> rename column headers
    - "AI Cleaner"             -> this feature itself, natural language instructions

  VALIDATE TAB:
    - "Validate Email"             -> flag or remove rows with invalid email addresses
    - "Standardize Phone Numbers"  -> strip non digit chars and format to +[country][number]
    - "Standardize Dates"          -> parse and reformat date columns to a chosen format
    - "Cap and Remove Outliers"    -> cap or remove statistical outliers using IQR or Z-score
    - "Validate Value Range"       -> flag or remove rows outside a min or max numeric range

--- YOUR JOB ---
Before writing any code, reason about whether the instruction can actually succeed
given the current state of the data. Ask yourself:
1. Which columns does this instruction touch?
2. What is the user's intent?
3. Does the app already have a feature that handles this?

STEP 1: check if the app already handles it.
Look at the app feature map above. If the user's request maps to an existing app feature,
set the explanation to clearly say two things:
  a) the guaranteed path: the recommended way is to go to the matching tab and use that
     feature, since it is a tested and guaranteed method that will work correctly.
  b) the code fallback: if they prefer to do it here, the code below should work, but it
     may not be fully accurate, so they should verify the result in the Profile tab and use
     the History and Export tab to undo if something looks off.
Then still generate the working pandas code for it.
Do not use PRE_CONDITION_FAILED when an app feature exists, always provide both.

STEP 2: check for pre-conditions, only if no app feature covers it.
If the instruction assumes a clean column but the column is not clean, block it.
Examples:
  - "fill missing Salary with the median" and Salary has currency symbols -> block
  - "sort by Salary descending" and Salary has currency symbols -> block
  - point the user to the right app feature to fix the column first, then come back.

Critical rule, never block cleaning or conversion instructions:
  if the instruction is about cleaning or converting a column, for example removing
  currency symbols, converting to numeric, stripping the percent sign, extracting the
  number, and no app feature covers this exact operation, generate the code. Do not warn
  them to fix it first, they are already fixing it.

STEP 3: generate code, only if steps 1 and 2 did not block.
The operation is feasible and no app feature covers it. Generate working pandas code.
Always end the explanation with a note that AI can make mistakes, so they should verify
the result in the Profile tab and use the History and Export tab to undo if needed.

If a pre-condition is needed, set code to PRE_CONDITION_FAILED and write a clear,
friendly explanation that states the specific problem and points to the exact app feature.
If the instruction is truly impossible or completely unclear, set code to CANNOT_DO.

Respond with ONLY a JSON object in this exact format, nothing else:
{{
  "code": "df = df.dropna(subset=['Age'])",
  "explanation": "This will remove all rows where the Age column is empty."
}}

Rules for the code field:
- Write clean pandas code, as many lines as needed
- Must assign back to df or modify df in place
- No imports, no comments, no print statements
- Use PRE_CONDITION_FAILED if a pre-condition must be met first
- Use CANNOT_DO if the instruction is truly impossible or completely unclear
- Never use .astype(float) or .astype(int) directly on a column that contains
  non numeric characters, it will crash. Always strip first, then convert.
- Use these safe patterns depending on what the column contains:
  currency or commas, e.g. "$1,200", "GBP2,300", "EUR500":
    df['col'] = pd.to_numeric(df['col'].astype(str).str.replace(r'[^\\d.\\-]', '', regex=True), errors='coerce')
  parentheses as negatives, e.g. "(1,200)":
    df['col'] = pd.to_numeric(df['col'].astype(str).str.replace(r'\\((.+?)\\)', r'-\\1', regex=True).str.replace(r'[^\\d.\\-]', '', regex=True), errors='coerce')
  percentage, e.g. "45%", "3.5%":
    df['col'] = pd.to_numeric(df['col'].astype(str).str.replace('%', '', regex=False).str.replace(r'[^\\d.\\-]', '', regex=True), errors='coerce') / 100
  unit suffixes, e.g. "10kg", "5.5km", "200lbs":
    df['col'] = pd.to_numeric(df['col'].astype(str).str.extract(r'([-]?\\d+\\.?\\d*)', expand=False), errors='coerce')
  mixed types or general dirty numeric, anything with stray non numeric chars:
    df['col'] = pd.to_numeric(df['col'].astype(str).str.replace(r'[^\\d.\\-]', '', regex=True), errors='coerce')
  datetime strings, e.g. "01/02/2023", "March 5 2022":
    df['col'] = pd.to_datetime(df['col'], errors='coerce')
  boolean like strings, e.g. "yes/no", "true/false", "1/0":
    df['col'] = df['col'].astype(str).str.lower().map({{'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}})
  Always pick the pattern that matches what you actually see in the raw sample values.

Do not include any text outside the JSON object."""

# call gemini api, retries once on transient failure
def _call_gemini(instruction, df):
    import json
    import urllib.request

    api_key = _get_api_key()
    if not api_key:
        return None, None, "GEMINI_API_KEY not set. Add it to your .env file.", False

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

    last_err = None
    body = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                msg = json.loads(error_body).get("error", {}).get("message", error_body)
            except Exception:
                msg = error_body
            return None, None, f"Gemini API error: {msg}", False
        except Exception as e:
            last_err = e
            if attempt == 0:
                continue

    if body is None:
        return None, None, f"Request failed after 2 attempts: {last_err}", False

    # parse model response
    try:
        raw = body["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        code = parsed.get("code", "").strip()
        explanation = parsed.get("explanation", "").strip()
    except Exception:
        return None, None, "Gemini returned an unexpected format. Try rephrasing your instruction.", False

    # pre_condition_failed, data not ready for this operation
    if "PRE_CONDITION_FAILED" in code:
        return None, None, explanation or "A pre-condition must be met before this operation can run.", True

    # cannot_do, instruction is impossible or unclear
    if "CANNOT_DO" in code:
        return None, None, explanation or "Gemini could not understand that instruction. Try rephrasing.", False

    return code, explanation, None, False

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

    # show success, pre-condition warning, or hard error from the previous action
    if st.session_state.get("_nl_success"):
        st.success(st.session_state.pop("_nl_success"))
    if st.session_state.get("_nl_precondition"):
        st.warning(st.session_state.pop("_nl_precondition"))
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
            code, explanation, err, is_precondition = _call_gemini(instruction.strip(), cdf)
        if err:
            if is_precondition:
                st.session_state["_nl_precondition"] = err
            else:
                st.session_state["_nl_error"] = err
            st.rerun()
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
    # always show the explanation card so user knows what will happen
    st.info(explanation)

    # editable code box, always shown, labelled as optional fallback
    edited_code = st.text_area(
        "Code -- review and edit before applying (AI-generated, may not be 100% accurate)",
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
