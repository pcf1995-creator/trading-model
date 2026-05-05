"""
shared/db_common.py — Supabase client + JSON fallback helpers shared by
kalshi/db.py and stocks/db.py.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    url = key = None
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
    except Exception:
        pass
    if not url:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

    if url and key:
        try:
            from supabase import create_client
            _client = create_client(url, key)
            logger.info("Supabase connected")
        except Exception as e:
            logger.warning(f"Supabase init failed: {e}")
    return _client


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
