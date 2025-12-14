import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config, io_utils


def _to_yyyymmdd(d):
    if d is None:
        return ""
    s = str(d).strip()
    if not s:
        return ""
    s = s.replace("-", "").replace("/", "")
    if len(s) == 8 and s.isdigit():
        return s
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y%m%d")


def _parse_trade_date(s):
    text = str(s).strip()
    if not text:
        return pd.NaT
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


@dataclass
class MarketDataRequest:
    symbol: str
    start_date: str
    end_date: str
    freq: str = "日"
    asset_type: str = "auto"


class MarketDataProvider:
    def get_ohlcv(self, req: MarketDataRequest):
        raise NotImplementedError


class YFinanceDataProvider(MarketDataProvider):
    def __init__(self, dirs):
        self.dirs = dirs
        self.logger = logging.getLogger("market.yfinance")

    def get_ohlcv(self, req: MarketDataRequest):
        try:
            import yfinance as yf
        except Exception as e:
            raise RuntimeError("缺少 yfinance: " + str(e))
        interval = "1wk" if str(req.freq) == "周" else "1d"
        df = yf.download(
            tickers=req.symbol,
            start=req.start_date,
            end=req.end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("yfinance 无数据")
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df = io_utils.normalize_datetime_column(df, date_col)
        df = df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns})
        return df


class TushareDataProvider(MarketDataProvider):
    def __init__(self, dirs, token=None, max_retries=3, backoff=1.8):
        self.dirs = dirs
        self.token = token or config.TUSHARE_TOKEN
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.logger = logging.getLogger("market.tushare")
        self._pro = None

    def _get_pro(self):
        if self._pro is not None:
            return self._pro
        try:
            import tushare as ts
        except Exception as e:
            raise RuntimeError("缺少 tushare: " + str(e))
        if not str(self.token or "").strip():
            raise RuntimeError("缺少 Tushare Token，请设置环境变量 TUSHARE_TOKEN 或创建 secrets/tushare_token.txt")
        ts.set_token(self.token)
        self._pro = ts.pro_api()
        return self._pro

    def _call_with_retry(self, fn, desc):
        last = None
        for i in range(self.max_retries):
            try:
                return fn()
            except Exception as e:
                last = e
                wait = self.backoff ** i
                self.logger.warning(desc + " 失败: " + str(e) + " , 等待 " + str(round(wait, 2)) + "s 重试")
                time.sleep(wait)
        raise last

    def _fetch_daily(self, ts_code, start_date, end_date):
        pro = self._get_pro()
        s = _to_yyyymmdd(start_date)
        e = _to_yyyymmdd(end_date)
        def call():
            return pro.daily(ts_code=ts_code, start_date=s, end_date=e)
        return self._call_with_retry(call, "tushare daily")

    def _fetch_index_daily(self, ts_code, start_date, end_date):
        pro = self._get_pro()
        s = _to_yyyymmdd(start_date)
        e = _to_yyyymmdd(end_date)
        def call():
            return pro.index_daily(ts_code=ts_code, start_date=s, end_date=e)
        return self._call_with_retry(call, "tushare index_daily")

    def _normalize_df(self, df):
        if df is None or df.empty:
            return pd.DataFrame()
        x = df.copy()
        if "trade_date" in x.columns:
            x["date"] = x["trade_date"].map(_parse_trade_date)
        elif "date" in x.columns:
            x["date"] = x["date"].map(_parse_trade_date)
        else:
            return pd.DataFrame()
        x = x.dropna(subset=["date"])
        rename = {}
        for c in ["open", "high", "low", "close"]:
            if c in x.columns:
                rename[c] = c
        if "vol" in x.columns and "volume" not in x.columns:
            rename["vol"] = "volume"
        if "volume" in x.columns:
            rename["volume"] = "volume"
        x = x.rename(columns=rename)
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in x.columns]
        out = x[["date"] + cols].sort_values("date")
        out = out.set_index("date")
        out["adj_close"] = out["close"]
        return out

    def _resample_weekly(self, df):
        if df is None or df.empty:
            return df
        res = {}
        for c in ["open", "high", "low", "close", "adj_close"]:
            if c in df.columns:
                if c == "open":
                    res[c] = df[c].astype(float).resample("W-FRI").first()
                elif c in ["high"]:
                    res[c] = df[c].astype(float).resample("W-FRI").max()
                elif c in ["low"]:
                    res[c] = df[c].astype(float).resample("W-FRI").min()
                else:
                    res[c] = df[c].astype(float).resample("W-FRI").last()
        if "volume" in df.columns:
            res["volume"] = df["volume"].astype(float).resample("W-FRI").sum()
        out = pd.concat(res.values(), axis=1)
        out.columns = list(res.keys())
        return out.dropna(how="all")

    def get_ohlcv(self, req: MarketDataRequest):
        symbol = str(req.symbol).strip()
        if not symbol:
            raise ValueError("ts_code 为空")
        payload = {
            "provider": "tushare",
            "symbol": symbol,
            "start": _to_yyyymmdd(req.start_date),
            "end": _to_yyyymmdd(req.end_date),
            "freq": str(req.freq),
            "asset_type": str(req.asset_type),
        }
        cache_file = io_utils.cache_path(self.dirs, "tushare_ohlcv", payload, ext=".csv")
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, encoding="utf-8-sig")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date")
                df = df.set_index("date")
                return df
            except Exception:
                pass

        asset_type = str(req.asset_type or "auto")
        raw = None
        if asset_type == "stock":
            raw = self._fetch_daily(symbol, req.start_date, req.end_date)
        elif asset_type == "index":
            raw = self._fetch_index_daily(symbol, req.start_date, req.end_date)
        else:
            raw = self._fetch_daily(symbol, req.start_date, req.end_date)
            if raw is None or raw.empty:
                raw = self._fetch_index_daily(symbol, req.start_date, req.end_date)
        df = self._normalize_df(raw)
        if df.empty:
            self.logger.warning("tushare 返回空数据: " + symbol)
            return df
        if str(req.freq) == "周":
            df = self._resample_weekly(df)
        out = df.reset_index()
        out["date"] = out["date"].astype("datetime64[ns]")
        out.to_csv(cache_file, index=False, encoding="utf-8-sig")
        return df
