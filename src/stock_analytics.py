import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import io_utils
from src.data.market_data import MarketDataRequest, TushareDataProvider, YFinanceDataProvider


def _periods_per_year(freq):
    return 52 if str(freq) == "周" else 252


def _annualize_series(r, freq, log_return):
    k = _periods_per_year(freq)
    if not bool(log_return):
        return (1.0 + r).pow(k) - 1.0
    return r * float(k)


def _rf_per_period(rf_annual, freq, log_return):
    rf = float(rf_annual or 0.0)
    if rf <= 0:
        return 0.0
    k = _periods_per_year(freq)
    if bool(log_return):
        return float(np.log1p(rf) / k)
    return float((1.0 + rf) ** (1.0 / k) - 1.0)


def _compute_returns(price, log_return):
    s = pd.Series(price).astype(float)
    if bool(log_return):
        r = np.log(s / s.shift(1))
    else:
        r = s.pct_change()
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def _diff_stats(returns, enable_ttest):
    delta = pd.Series(returns).diff().dropna()
    n = int(delta.shape[0])
    mean = float(delta.mean()) if n else float("nan")
    std = float(delta.std(ddof=1)) if n > 1 else float("nan")
    var = float(std ** 2) if np.isfinite(std) else float("nan")
    t_stat = None
    p_value = None
    if bool(enable_ttest) and n > 1 and std and not math.isnan(std) and std > 0:
        t_stat = mean / (std / math.sqrt(n))
        try:
            from scipy import stats

            p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
        except Exception:
            z = abs(float(t_stat))
            p_value = float(math.erfc(z / math.sqrt(2.0)))
    return {
        "n": n,
        "mean_delta": mean,
        "std_delta": std,
        "var_delta": var,
        "t_stat": None if t_stat is None else float(t_stat),
        "p_value": None if p_value is None else float(p_value),
    }, delta


def _ols_alpha_beta(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.shape[0])
    if n < 3:
        raise ValueError("样本不足")
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except Exception:
        XtX_inv = np.linalg.pinv(XtX)
    coef = XtX_inv @ (X.T @ y)
    alpha = float(coef[0])
    beta = float(coef[1])
    y_hat = X @ coef
    resid = y - y_hat
    k = 2
    sse = float(resid.T @ resid)
    s2 = sse / float(n - k)
    cov = s2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    t_vals = coef / se
    try:
        from scipy import stats

        p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_vals), df=n - k))
        p_alpha = float(p_vals[0])
        p_beta = float(p_vals[1])
    except Exception:
        p_alpha = float(math.erfc(abs(float(t_vals[0])) / math.sqrt(2.0)))
        p_beta = float(math.erfc(abs(float(t_vals[1])) / math.sqrt(2.0)))
    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")
    return {
        "n": n,
        "alpha": alpha,
        "beta": beta,
        "t_alpha": float(t_vals[0]),
        "t_beta": float(t_vals[1]),
        "p_alpha": p_alpha,
        "p_beta": p_beta,
        "r2": float(r2),
    }


def _mpl_available():
    try:
        import matplotlib
        from matplotlib.figure import Figure

        try:
            matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
        return Figure, None
    except Exception as e:
        return None, str(e)


def _plot_price(df, price_col, title):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    fig = Figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.plot(df.index, df[price_col].astype(float).values, color="blue", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(price_col)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _plot_returns(r_asset, r_mkt, title):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    fig = Figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.plot(r_asset.index, r_asset.values, color="blue", linewidth=1.0, label="asset")
    ax.plot(r_mkt.index, r_mkt.values, color="orange", linewidth=1.0, label="market")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("return")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_diff(delta, mean_delta, title):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    fig = Figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.plot(delta.index, delta.values, color="purple", linewidth=0.9)
    if np.isfinite(mean_delta):
        ax.axhline(mean_delta, color="red", linestyle="--", linewidth=1.0)
        ax.text(0.01, 0.95, "mean Δr = " + str(round(float(mean_delta), 8)), transform=ax.transAxes, va="top")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Δr")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _plot_regression(x, y, alpha, beta, title):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=14, alpha=0.6, color="blue")
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ys = alpha + beta * xs
    ax.plot(xs, ys, color="red", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("market return")
    ax.set_ylabel("asset return")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _prepare_price_df_from_csv(csv_file, freq):
    b = io_utils.read_uploaded_bytes(csv_file)
    df = io_utils.read_csv_bytes(b)
    cols = io_utils.detect_ohlcv_columns(df)
    df = io_utils.normalize_datetime_column(df, cols.get("date"))
    col_close = cols.get("close")
    col_adj = cols.get("adj_close")
    if col_close is None and col_adj is None:
        raise ValueError("CSV 未找到 close 或 adj close 列")
    out = df.copy()
    if str(freq) == "周":
        res = {}
        for c in [col_close, col_adj, cols.get("open"), cols.get("high"), cols.get("low")]:
            if c and c in out.columns:
                res[c] = out[c].astype(float).resample("W-FRI").last()
        if cols.get("volume") and cols["volume"] in out.columns:
            res[cols["volume"]] = out[cols["volume"]].astype(float).resample("W-FRI").sum()
        out = pd.concat(res.values(), axis=1)
        out.columns = list(res.keys())
        out = out.dropna(how="all")
    out = out.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in out.columns})
    return out


def _filter_by_date(df, start_date, end_date):
    if df is None or df.empty:
        return df
    start_ts = pd.to_datetime(start_date, errors="coerce") if start_date else None
    end_ts = pd.to_datetime(end_date, errors="coerce") if end_date else None
    out = df.copy()
    if start_ts is not None and not pd.isna(start_ts):
        out = out.loc[out.index >= start_ts]
    if end_ts is not None and not pd.isna(end_ts):
        out = out.loc[out.index <= end_ts]
    return out


def _pick_price_column(df, price_field):
    tmp = df.reset_index()
    cols = io_utils.detect_ohlcv_columns(tmp)
    if str(price_field) == "Adj Close":
        return cols.get("adj_close") or cols.get("close")
    return cols.get("close") or cols.get("adj_close")


def run_stock_analytics(params, asset_csv_file, market_csv_file, dirs):
    try:
        freq = params.get("freq")
        price_field = params.get("price_field")
        log_return = bool(params.get("log_return"))
        annualize = bool(params.get("annualize"))
        rf_annual = float(params.get("rf_annual") or 0.0)
        capm_excess = bool(params.get("capm_excess"))
        enable_ttest = bool(params.get("enable_ttest"))

        mode = str(params.get("data_mode") or "Tushare (CN)")

        if mode == "CSV 上传":
            asset_df = _prepare_price_df_from_csv(asset_csv_file, freq=freq)
            if market_csv_file is None:
                raise ValueError("CSV 模式需要提供市场基准 CSV")
            market_df = _prepare_price_df_from_csv(market_csv_file, freq=freq)
        elif mode == "yfinance":
            provider = YFinanceDataProvider(dirs=dirs)
            asset_df = provider.get_ohlcv(MarketDataRequest(symbol=params.get("ticker"), start_date=params.get("start_date"), end_date=params.get("end_date"), freq=freq))
            market_df = provider.get_ohlcv(MarketDataRequest(symbol=params.get("benchmark_ticker"), start_date=params.get("start_date"), end_date=params.get("end_date"), freq=freq))
        else:
            provider = TushareDataProvider(dirs=dirs)
            asset_type = params.get("asset_type") or "auto"
            bench_type = params.get("benchmark_type") or "auto"
            asset_df = provider.get_ohlcv(MarketDataRequest(symbol=params.get("ticker"), start_date=params.get("start_date"), end_date=params.get("end_date"), freq=freq, asset_type=asset_type))
            market_df = provider.get_ohlcv(MarketDataRequest(symbol=params.get("benchmark_ticker"), start_date=params.get("start_date"), end_date=params.get("end_date"), freq=freq, asset_type=bench_type))
            if asset_df is None or asset_df.empty:
                raise ValueError("tushare 标的返回空数据")
            if market_df is None or market_df.empty:
                raise ValueError("tushare 市场基准返回空数据")

        asset_df = _filter_by_date(asset_df, params.get("start_date"), params.get("end_date"))
        market_df = _filter_by_date(market_df, params.get("start_date"), params.get("end_date"))

        asset_price_col = _pick_price_column(asset_df, price_field)
        market_price_col = _pick_price_column(market_df, price_field)
        if asset_price_col is None:
            raise ValueError("标的价格列缺失")
        if market_price_col is None:
            raise ValueError("市场基准价格列缺失")

        asset_price = asset_df[asset_price_col].astype(float)
        market_price = market_df[market_price_col].astype(float)

        r_asset = _compute_returns(asset_price, log_return=log_return)
        r_mkt = _compute_returns(market_price, log_return=log_return)

        aligned = pd.DataFrame({"asset_ret": r_asset, "market_ret": r_mkt}).dropna()
        if aligned.empty:
            raise ValueError("收益率对齐后为空")

        if annualize:
            aligned["asset_ret"] = _annualize_series(aligned["asset_ret"], freq=freq, log_return=log_return)
            aligned["market_ret"] = _annualize_series(aligned["market_ret"], freq=freq, log_return=log_return)

        rf_used = float(rf_annual if annualize else _rf_per_period(rf_annual, freq=freq, log_return=log_return))
        x = aligned["market_ret"].astype(float).values
        y = aligned["asset_ret"].astype(float).values
        if capm_excess:
            x = x - rf_used
            y = y - rf_used

        reg = _ols_alpha_beta(x=x, y=y)
        diff_stats, delta = _diff_stats(aligned["asset_ret"], enable_ttest=enable_ttest)

        figures = {}
        price_fig = _plot_price(asset_df, asset_price_col, "Price " + str(params.get("ticker")))
        returns_fig = _plot_returns(aligned["asset_ret"], aligned["market_ret"], "Returns")
        diff_fig = _plot_diff(delta, diff_stats.get("mean_delta"), "Δr statistics")
        reg_fig = _plot_regression(x, y, reg["alpha"], reg["beta"], "Regression")
        figures["price"] = price_fig
        figures["returns"] = returns_fig
        figures["diff"] = diff_fig
        figures["regression"] = reg_fig

        out_df = aligned.copy()
        out_df.index = out_df.index.astype("datetime64[ns]")
        out_df = out_df.reset_index().rename(columns={"index": "date"})
        metrics = {"regression": reg, "diff_stats": diff_stats}

        return {"ok": True, "data": out_df, "metrics": metrics, "figures": figures, "params": params}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def export_stock_analytics(result, dirs, tag, export_format):
    if not isinstance(result, dict) or not result.get("ok"):
        return []
    params = result.get("params") or {}
    payload = {"kind": "stock", "tag": str(tag or ""), "params": params}
    suffix = io_utils.safe_filename(tag) if str(tag or "").strip() else io_utils.stable_hash(payload)
    data_df = result.get("data")
    if not isinstance(data_df, pd.DataFrame):
        data_df = pd.DataFrame()
    reg = (result.get("metrics") or {}).get("regression") or {}
    diff_stats = (result.get("metrics") or {}).get("diff_stats") or {}
    base = "stock_analytics_" + suffix
    out = []
    if str(export_format) == "Excel":
        path = Path(dirs["tables"]) / (base + ".xlsx")
        io_utils.save_excel(
            {
                "data": data_df,
                "regression": reg,
                "diff_stats": diff_stats,
                "params": pd.DataFrame([params]),
            },
            path,
        )
        out.append(path)
    else:
        out.append(io_utils.save_df_csv(data_df, Path(dirs["tables"]) / (base + "_data.csv")))
        out.append(io_utils.save_df_csv(pd.DataFrame([reg]), Path(dirs["tables"]) / (base + "_regression.csv")))
        out.append(io_utils.save_df_csv(pd.DataFrame([diff_stats]), Path(dirs["tables"]) / (base + "_diff_stats.csv")))
        out.append(io_utils.save_df_csv(pd.DataFrame([params]), Path(dirs["tables"]) / (base + "_params.csv")))
    return out


def save_stock_figures(result, dirs, tag):
    if not isinstance(result, dict) or not result.get("ok"):
        return []
    params = result.get("params") or {}
    payload = {"kind": "stock_fig", "tag": str(tag or ""), "params": params}
    suffix = io_utils.safe_filename(tag) if str(tag or "").strip() else io_utils.stable_hash(payload)
    figs = result.get("figures") or {}
    out_paths = []
    for key in ["price", "returns", "diff", "regression"]:
        fig = figs.get(key)
        if fig is None:
            continue
        path = Path(dirs["figures"]) / ("stock_" + key + "_" + suffix + ".png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        out_paths.append(path)
    return out_paths
