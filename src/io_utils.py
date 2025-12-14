import hashlib
import json
import re
from io import BytesIO
from pathlib import Path

import pandas as pd


def ensure_output_dirs(root):
    root_path = Path(root).resolve()
    outputs = root_path / "outputs"
    figures = outputs / "figures"
    tables = outputs / "tables"
    forum_db = outputs / "forum_db"
    models = outputs / "models"
    logs = outputs / "logs"
    cache_dir = outputs / "cache"
    data_dir = outputs / "data"
    reports = outputs / "reports"
    for p in [outputs, figures, tables, forum_db, models, logs, cache_dir, data_dir, reports]:
        p.mkdir(parents=True, exist_ok=True)
    return {
        "root": root_path,
        "outputs": outputs,
        "figures": figures,
        "tables": tables,
        "forum_db": forum_db,
        "models": models,
        "logs": logs,
        "cache": cache_dir,
        "data": forum_db,
        "reports": tables,
    }


def safe_filename(text, max_len=140):
    s = str(text).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\u4e00-\u9fff\-\._]+", "_", s, flags=re.UNICODE)
    s = s.strip("._-")
    if not s:
        s = "file"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def stable_hash(payload, length=16):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return h[:length]


def parse_kv_text(text):
    if text is None:
        return {}
    s = str(text).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    out = {}
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            k, v = line.split(":", 1)
        elif "=" in line:
            k, v = line.split("=", 1)
        else:
            continue
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def read_uploaded_bytes(uploaded):
    if uploaded is None:
        return None
    if isinstance(uploaded, (str, Path)):
        p = Path(uploaded)
        if p.exists() and p.is_file():
            return p.read_bytes()
        return None
    if hasattr(uploaded, "getvalue"):
        return uploaded.getvalue()
    if hasattr(uploaded, "read"):
        return uploaded.read()
    return None


def read_csv_bytes(data_bytes):
    if data_bytes is None:
        return pd.DataFrame()
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "big5"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(BytesIO(data_bytes), encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def detect_ohlcv_columns(df):
    cols = list(df.columns)
    lower = {c: str(c).strip().lower() for c in cols}

    def find(candidates):
        for c in cols:
            name = lower[c].replace(" ", "")
            if name in candidates:
                return c
        for c in cols:
            name = lower[c]
            for cand in candidates:
                if cand in name:
                    return c
        return None

    date_col = find({"date", "datetime", "time", "timestamp", "trade_date", "交易日期", "日期"})
    close_col = find({"close", "收盘", "收盘价"})
    adj_col = find({"adjclose", "adj_close", "adjustedclose", "adjustclose", "adj", "复权收盘", "复权收盘价"})
    open_col = find({"open", "开盘", "开盘价"})
    high_col = find({"high", "最高", "最高价"})
    low_col = find({"low", "最低", "最低价"})
    volume_col = find({"volume", "vol", "成交量"})
    return {
        "date": date_col,
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
        "adj_close": adj_col,
        "volume": volume_col,
    }


def normalize_datetime_column(df, date_col):
    if date_col is None or date_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            out = df.copy()
            out = out.sort_index()
            return out
        raise ValueError("未找到日期列")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out = out.sort_values(date_col)
    out = out.set_index(date_col)
    return out


def save_df_csv(df, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def save_excel(sheets, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        for name, obj in sheets.items():
            if obj is None:
                continue
            if isinstance(obj, dict):
                df = pd.DataFrame([obj])
            else:
                df = obj
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)
    return p


def cache_path(dirs, prefix, payload, ext=".csv"):
    h = stable_hash(payload)
    name = safe_filename(prefix) + "_" + h + ext
    return Path(dirs["cache"]) / name


def build_user_agent():
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def request_get(url, headers=None, timeout=12, retries=3, backoff=1.8, session=None):
    import time

    try:
        import requests
    except Exception as e:
        raise RuntimeError("缺少 requests: " + str(e))
    sess = session or requests.Session()
    h = dict(headers or {})
    if "User-Agent" not in h:
        h["User-Agent"] = build_user_agent()
    last = None
    for i in range(int(retries)):
        try:
            resp = sess.get(url, headers=h, timeout=timeout)
            if resp.status_code >= 400:
                last = RuntimeError("HTTP " + str(resp.status_code))
            else:
                return resp
        except Exception as e:
            last = e
        time.sleep(backoff ** i)
    raise last


def parse_datetime_text(text):
    s = str(text).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return None

