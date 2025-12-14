import os
from pathlib import Path


def _load_token():
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    root = Path(__file__).resolve().parent.parent
    p = root / "secrets" / "tushare_token.txt"
    try:
        if p.exists():
            t = p.read_text(encoding="utf-8", errors="ignore").strip()
            if t:
                return t
    except Exception:
        return ""
    return ""


TUSHARE_TOKEN = _load_token()

