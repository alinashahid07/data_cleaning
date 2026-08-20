import hashlib
import os
import pickle
import time
import pandas as pd
import streamlit as st

PERSIST_DIR = os.path.join(os.path.expanduser("~"), ".dp_sessions")
MAX_SESSION_AGE_SECONDS = 60 * 60 * 24 * 2
MAX_HISTORY_STEPS = 20

# set this to True to show debug info on screen, set to False when everything works
DEBUG = True

def _debug(msg):
    if DEBUG:
        st.write(f"[session_persist] {msg}")

def _ensure_dir():
    try:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _debug(f"dir ready: {PERSIST_DIR}")
    except Exception as e:
        _debug(f"makedirs failed: {e}")

# builds a session key from filename and file size
# file_id from streamlit changes on every reload so it cannot be used
# filename plus size is stable as long as the user uploads the same file
def make_stable_file_key(filename: str, file_bytes: bytes) -> str:
    size = len(file_bytes)
    raw = f"{filename}_{size}"
    key = hashlib.md5(raw.encode()).hexdigest()
    _debug(f"stable key for {filename} ({size} bytes): {key}")
    return key

def _session_path(stable_key: str) -> str:
    return os.path.join(PERSIST_DIR, f"session_{stable_key}.pkl")

def _cleanup_old_sessions():
    try:
        now = time.time()
        for fname in os.listdir(PERSIST_DIR):
            fpath = os.path.join(PERSIST_DIR, fname)
            if os.path.isfile(fpath):
                age = now - os.path.getmtime(fpath)
                if age > MAX_SESSION_AGE_SECONDS:
                    os.remove(fpath)
    except Exception:
        pass

# saves current df, original df, and history to disk under the users home dir
# home dir works on both windows and linux unlike tmp which has path separator issues
# pickle is used because it handles pandas dataframes natively and is fast
# only the last max_history_steps entries are kept to cap file size
def save_session(stable_key: str, current_df: pd.DataFrame, history: list, original_df: pd.DataFrame):
    _ensure_dir()
    path = _session_path(stable_key)
    _debug(f"saving to: {path}")
    try:
        trimmed_history = history[-MAX_HISTORY_STEPS:]
        payload = {
            "stable_key": stable_key,
            "saved_at": time.time(),
            "current_df": current_df,
            "original_df": original_df,
            "history": trimmed_history,
            "history_len": len(trimmed_history),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        _debug(f"save ok, file size: {os.path.getsize(path)} bytes")
    except Exception as e:
        _debug(f"save failed: {e}")

# returns saved session dict or none if nothing exists or file is unreadable
# dict keys: current_df, original_df, history, history_len, saved_at
def load_session(stable_key: str):
    path = _session_path(stable_key)
    _debug(f"loading from: {path}, exists: {os.path.exists(path)}")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("stable_key") != stable_key:
            _debug("stable key mismatch in loaded file")
            return None
        _debug(f"load ok, {len(payload.get('history', []))} history steps")
        return payload
    except Exception as e:
        _debug(f"load failed: {e}")
        return None

# wipes the persisted session for this key
# called when user picks start fresh or resets data
def delete_session(stable_key: str):
    path = _session_path(stable_key)
    try:
        if os.path.exists(path):
            os.remove(path)
            _debug(f"deleted: {path}")
    except Exception as e:
        _debug(f"delete failed: {e}")

def session_exists(stable_key: str) -> bool:
    exists = os.path.exists(_session_path(stable_key))
    _debug(f"session_exists({stable_key}): {exists}")
    return exists

# returns a human readable string like 3 minutes ago or 2 hours ago
def format_saved_time(saved_at: float) -> str:
    delta = time.time() - saved_at
    if delta < 60:
        return "just now"
    if delta < 3600:
        mins = int(delta // 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    if delta < 86400:
        hrs = int(delta // 3600)
        return f"{hrs} hour{'s' if hrs != 1 else ''} ago"
    days = int(delta // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"

def cleanup_old_sessions():
    _cleanup_old_sessions()
