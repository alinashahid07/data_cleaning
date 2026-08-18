import json
import re
import urllib.request
import urllib.error
import streamlit as st

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
def _build_prompt(df):
    col_info = "\n".join(f"  {col} ({df[col].dtype})" for col in df.columns)
    sample = df.head(8).to_string(index=False)
    missing = df.isna().sum()
    missing_info = "\n".join(
        f"  {col}: {cnt} missing ({round(cnt / len(df) * 100, 1)}%)"
        for col, cnt in missing.items() if cnt > 0
    ) or "  none"
    duplicate_rows = int(df.duplicated().sum())

    return f"""You are a data quality analyst helping a non-technical user clean their data using a specific app.

IMPORTANT: Only reference features that exist exactly as listed below. Do NOT invent button names or tools.

APP FEATURES BY TAB:

Recommendations tab handles these automatically. If you detect any of these issues, set tab to "Recommendations" and write the action as ONLY: "Click Select columns next to the issue, then click Fix. Or press Auto-Fix All at the bottom to fix everything at once.":
- Extra whitespace in text columns
- Duplicate rows
- Duplicate columns
- Unwanted edge characters in text
- Currency columns stored as text
- Percentage columns stored as text
- Columns with measurement units stored as text
- Time duration columns stored as text
- Missing values

Clean tab features (use exact names only):
- AI Cleaner: user types a plain English instruction and AI generates the code
- Strip Whitespace button
- Drop Duplicate Rows button
- Drop Duplicate Cols button
- Clean String Edges button
- Smart Column Cleaner button
- Handle Missing Values button
- Find and Replace section: pick a column, type find value, type replace value, click Run
- Column Type Override section: pick a column, pick a type from the dropdown, click Apply
- Split Column section: pick a column, enter a delimiter, enter output column names, click Split
- Merge Columns section: pick columns, enter a separator, enter a new column name, click Merge
- Rename Columns section: edit names in the table, click Apply
- Data Type Guesser section: click Apply Selected to convert suggested columns

Validate tab features (use exact names only):
- Validate Email: choose Flag invalid or Remove invalid rows, select columns, click Run
- Standardize Phone Numbers: select columns, click Run
- Standardize Dates: pick an output format, select columns, click Run
- Cap and Remove Outliers: pick Method (iqr or zscore), pick Action (cap or remove), set Threshold, select columns, click Run
- Validate Value Range: set Min and Max, pick Action (flag or remove), select columns, click Run

Column names and types:
{col_info}

First 8 rows:
{sample}

Missing value counts:
{missing_info}

Duplicate rows: {duplicate_rows}
Total rows: {len(df)}
Total columns: {len(df.columns)}

Respond with ONLY a JSON object in this exact format, nothing else:
{{
  "summary": "2 to 4 plain English sentences describing what this dataset is about, what the main columns represent, and the overall data quality. No jargon.",
  "fixes": [
    {{
      "priority": 1,
      "issue": "short 3 to 6 word issue title",
      "column": "column name or ALL if dataset-wide",
      "reason": "one plain English sentence explaining why this matters",
      "tab": "Recommendations or Clean or Validate",
      "action": "one sentence only, using exact feature names. do NOT start with the tab name, that is shown separately.",
      "shortcut": "Or use the AI Cleaner in the Clean tab and type: then one plain English instruction. If fix is in Recommendations tab write: Or press Auto-Fix All in the Recommendations tab to fix this and all other detected issues at once."
    }}
  ]
}}

Rules:
- summary is plain English, 2 to 4 sentences, no jargon
- include 3 to 6 fixes sorted by priority, most important first
- tab must be one of: Recommendations, Clean, Validate
- action must NOT repeat or start with the tab name
- action must use ONLY exact feature names from the list above
- do not include any text outside the JSON object"""

# call gemini api
def _call_gemini(df):
    api_key = _get_api_key()
    if not api_key:
        return None, "GEMINI_API_KEY not set."

    url = GEMINI_API_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": _build_prompt(df)}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024}
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            msg = json.loads(error_body).get("error", {}).get("message", error_body)
        except Exception:
            msg = error_body
        return None, f"Gemini API error: {msg}"
    except Exception as e:
        return None, f"Request failed: {e}"

    # parse model response
    try:
        raw = body["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw), None
    except Exception:
        return None, "Gemini returned an unexpected format."

# fetches ai insights for the given dataframe
# caches result in session state so subsequent calls are instant
# always triggers the gemini call if not cached
def get_ai_insights(df, file_id):
    cache_key = f"ai_insights_{file_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key], None

    result, err = _call_gemini(df)
    if err:
        return None, err

    st.session_state[cache_key] = result
    return result, None

# renders the ai summary in the overview tab
# never triggers the api call, only reads from cache
# the actual api call is made in app.py after all tabs render
def render_summary(df, file_id):
    cache_key = f"ai_insights_{file_id}"
    insights = st.session_state.get(cache_key)

    if not insights:
        # still loading, show a subtle placeholder, not a spinner
        st.caption("AI summary loading...")
        return

    summary = insights.get("summary", "")
    if summary:
        st.info(f"**AI Summary:** {summary}")
