import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import io_utils
from src.data.market_data import MarketDataRequest, TushareDataProvider, YFinanceDataProvider


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


def _torch_available():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        return {"torch": torch, "nn": nn, "DataLoader": DataLoader, "TensorDataset": TensorDataset}, None
    except Exception as e:
        return None, str(e)


def _download_yfinance(tickers, start_date, end_date, freq, dirs):
    provider = YFinanceDataProvider(dirs=dirs)
    frames = []
    for t in tickers:
        df = provider.get_ohlcv(MarketDataRequest(symbol=t, start_date=start_date, end_date=end_date, freq=freq))
        if df is None or df.empty:
            continue
        x = df.copy()
        x["ticker"] = t
        frames.append(x)
    if not frames:
        raise RuntimeError("yfinance 无数据")
    out = pd.concat(frames, axis=0).sort_index()
    return out


def _download_tushare(tickers, start_date, end_date, freq, dirs):
    provider = TushareDataProvider(dirs=dirs)
    frames = []
    for t in tickers:
        df = provider.get_ohlcv(MarketDataRequest(symbol=t, start_date=start_date, end_date=end_date, freq=freq, asset_type="auto"))
        if df is None or df.empty:
            continue
        x = df.copy()
        x["ticker"] = t
        frames.append(x)
    if not frames:
        raise RuntimeError("tushare 无数据")
    out = pd.concat(frames, axis=0).sort_index()
    return out


def _load_ohlcv_csv(csv_file):
    b = io_utils.read_uploaded_bytes(csv_file)
    df = io_utils.read_csv_bytes(b)
    cols = io_utils.detect_ohlcv_columns(df)
    df = io_utils.normalize_datetime_column(df, cols.get("date"))
    rename = {}
    if cols.get("open"):
        rename[cols["open"]] = "open"
    if cols.get("high"):
        rename[cols["high"]] = "high"
    if cols.get("low"):
        rename[cols["low"]] = "low"
    if cols.get("close"):
        rename[cols["close"]] = "close"
    if cols.get("adj_close"):
        rename[cols["adj_close"]] = "adj_close"
    if cols.get("volume"):
        rename[cols["volume"]] = "volume"
    df = df.rename(columns=rename)
    if "ticker" not in df.columns:
        df["ticker"] = "UPLOAD"
    df = df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns})
    return df


def _ema(s, span):
    return s.ewm(span=int(span), adjust=False).mean()


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(int(period)).mean()
    avg_loss = loss.rolling(int(period)).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close, fast=12, slow=26, signal=9):
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd = fast_ema - slow_ema
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def _build_features(df, use_ohlcv, use_ma, use_rsi, use_macd, use_volatility):
    x = df.copy()
    if "adj_close" in x.columns and "close" not in x.columns:
        x["close"] = x["adj_close"]
    cols = []
    if bool(use_ohlcv):
        for c in ["open", "high", "low", "close", "volume"]:
            if c in x.columns:
                cols.append(c)
    if "close" in x.columns and (bool(use_ma) or bool(use_rsi) or bool(use_macd) or bool(use_volatility)):
        x["ret1"] = x["close"].pct_change()
        cols.append("ret1")
        if bool(use_ma):
            x["ma5"] = x["close"].rolling(5).mean()
            x["ma10"] = x["close"].rolling(10).mean()
            cols.extend(["ma5", "ma10"])
        if bool(use_rsi):
            x["rsi14"] = _rsi(x["close"], 14)
            cols.append("rsi14")
        if bool(use_macd):
            macd, sig, hist = _macd(x["close"], 12, 26, 9)
            x["macd"] = macd
            x["macd_signal"] = sig
            x["macd_hist"] = hist
            cols.extend(["macd", "macd_signal", "macd_hist"])
        if bool(use_volatility):
            x["vol10"] = x["ret1"].rolling(10).std()
            cols.append("vol10")
    cols = [c for c in cols if c in x.columns]
    feat = x[cols].astype(float)
    return feat


def _make_samples(feat_df, close_series, lookback, task):
    base = feat_df.copy()
    base["target_return"] = close_series.astype(float).pct_change().shift(-1)
    base = base.dropna()
    features = base.drop(columns=["target_return"])
    target_ret = base["target_return"].astype(float)
    X = []
    y = []
    idx = []
    L = int(lookback)
    for t in range(L - 1, len(base)):
        window = features.iloc[t - L + 1 : t + 1].values.astype(np.float32)
        X.append(window)
        if str(task).startswith("分类"):
            y.append(1.0 if float(target_ret.iloc[t]) > 0 else 0.0)
        else:
            y.append(float(target_ret.iloc[t]))
        idx.append(base.index[t])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y, idx, list(features.columns)


def _time_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = int(X.shape[0])
    n_train = max(1, int(n * float(train_ratio)))
    n_val = max(1, int(n * float(val_ratio)))
    n_train = min(n_train, n - 2) if n >= 3 else max(1, n - 1)
    n_val = min(n_val, n - n_train - 1) if n - n_train >= 2 else max(0, n - n_train)
    train_end = n_train
    val_end = n_train + n_val
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _standardize(train_X, X_list):
    flat = train_X.reshape(-1, train_X.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    out = []
    for X in X_list:
        out.append(((X - mean) / std).astype(np.float32))
    return out, mean.astype(np.float32), std.astype(np.float32)


def _build_model(toolkit, model_name, input_dim, lookback, task, hidden_size, layers, dropout):
    nn = toolkit["nn"]
    out_dim = 1
    name = str(model_name)
    if name == "MLP":
        return MLP(nn, input_dim * int(lookback), int(hidden_size), int(layers), float(dropout), out_dim)
    if name == "LSTM":
        return LSTMNet(nn, input_dim, int(hidden_size), int(layers), float(dropout), out_dim)
    if name == "1D-CNN":
        return CNN1D(nn, input_dim, int(hidden_size), out_dim, float(dropout))
    if name == "TransformerEncoder":
        return TransformerNet(nn, input_dim, int(hidden_size), int(layers), float(dropout), out_dim)
    return MLP(nn, input_dim * int(lookback), int(hidden_size), int(layers), float(dropout), out_dim)


class MLP:
    def __init__(self, nn, input_dim, hidden, layers, dropout, out_dim):
        mods = []
        d = int(input_dim)
        h = int(hidden)
        for _ in range(max(1, int(layers))):
            mods.append(nn.Linear(d, h))
            mods.append(nn.ReLU())
            if float(dropout) > 0:
                mods.append(nn.Dropout(float(dropout)))
            d = h
        mods.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*mods)

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.net(x)


class LSTMNet:
    def __init__(self, nn, input_dim, hidden, layers, dropout, out_dim):
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden),
            num_layers=int(layers),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.head = nn.Linear(int(hidden), int(out_dim))

    def to(self, device):
        self.lstm = self.lstm.to(device)
        self.head = self.head.to(device)
        return self

    def train(self):
        self.lstm.train()
        self.head.train()

    def eval(self):
        self.lstm.eval()
        self.head.eval()

    def parameters(self):
        for p in self.lstm.parameters():
            yield p
        for p in self.head.parameters():
            yield p

    def state_dict(self):
        return {"lstm": self.lstm.state_dict(), "head": self.head.state_dict()}

    def load_state_dict(self, sd):
        self.lstm.load_state_dict(sd["lstm"])
        self.head.load_state_dict(sd["head"])

    def __call__(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class CNN1D:
    def __init__(self, nn, input_dim, hidden, out_dim, dropout):
        self.conv1 = nn.Conv1d(int(input_dim), int(hidden), kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else None
        self.fc = nn.Linear(int(hidden), int(out_dim))

    def to(self, device):
        self.conv1 = self.conv1.to(device)
        self.fc = self.fc.to(device)
        if self.drop is not None:
            self.drop = self.drop.to(device)
        return self

    def train(self):
        self.conv1.train()
        self.fc.train()
        if self.drop is not None:
            self.drop.train()

    def eval(self):
        self.conv1.eval()
        self.fc.eval()
        if self.drop is not None:
            self.drop.eval()

    def parameters(self):
        for p in self.conv1.parameters():
            yield p
        for p in self.fc.parameters():
            yield p
        if self.drop is not None:
            for p in self.drop.parameters():
                yield p

    def state_dict(self):
        return {"conv1": self.conv1.state_dict(), "fc": self.fc.state_dict()}

    def load_state_dict(self, sd):
        self.conv1.load_state_dict(sd["conv1"])
        self.fc.load_state_dict(sd["fc"])

    def __call__(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x).squeeze(-1)
        if self.drop is not None:
            x = self.drop(x)
        return self.fc(x)


class TransformerNet:
    def __init__(self, nn, input_dim, hidden, layers, dropout, out_dim):
        self.proj = nn.Linear(int(input_dim), int(hidden))
        nhead = 4
        if int(hidden) % nhead != 0:
            nhead = 1
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden),
            nhead=int(nhead),
            dim_feedforward=int(hidden) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=int(layers))
        self.head = nn.Linear(int(hidden), int(out_dim))

    def to(self, device):
        self.proj = self.proj.to(device)
        self.enc = self.enc.to(device)
        self.head = self.head.to(device)
        return self

    def train(self):
        self.proj.train()
        self.enc.train()
        self.head.train()

    def eval(self):
        self.proj.eval()
        self.enc.eval()
        self.head.eval()

    def parameters(self):
        for p in self.proj.parameters():
            yield p
        for p in self.enc.parameters():
            yield p
        for p in self.head.parameters():
            yield p

    def state_dict(self):
        return {"proj": self.proj.state_dict(), "enc": self.enc.state_dict(), "head": self.head.state_dict()}

    def load_state_dict(self, sd):
        self.proj.load_state_dict(sd["proj"])
        self.enc.load_state_dict(sd["enc"])
        self.head.load_state_dict(sd["head"])

    def __call__(self, x):
        x = self.proj(x)
        x = self.enc(x)
        x = x[:, -1, :]
        return self.head(x)


def _loss_fn(toolkit, task):
    nn = toolkit["nn"]
    if str(task).startswith("分类"):
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()


def _train_loop(toolkit, model, loaders, task, epochs, lr):
    torch = toolkit["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = _loss_fn(toolkit, task)
    opt = torch.optim.Adam(list(model.parameters()), lr=float(lr))
    hist = {"train": [], "val": []}
    for _ in range(int(epochs)):
        model.train()
        tr_losses = []
        for xb, yb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.detach().cpu().item()))
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_losses.append(float(loss.detach().cpu().item()))
        hist["train"].append(float(np.mean(tr_losses)) if tr_losses else float("nan"))
        hist["val"].append(float(np.mean(va_losses)) if va_losses else float("nan"))
    return model, hist, device


def _predict(toolkit, model, loader, device):
    torch = toolkit["torch"]
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            ps.append(pred)
            ys.append(yb.numpy())
    y = np.vstack(ys) if ys else np.zeros((0, 1), dtype=np.float32)
    p = np.vstack(ps) if ps else np.zeros((0, 1), dtype=np.float32)
    return y.reshape(-1), p.reshape(-1)


def _metrics_regression(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mse = float(np.mean((p - y) ** 2)) if y.size else float("nan")
    mae = float(np.mean(np.abs(p - y))) if y.size else float("nan")
    rmse = float(math.sqrt(mse)) if np.isfinite(mse) else float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse}


def _metrics_classification(y, logits):
    y = np.asarray(y, dtype=float).reshape(-1)
    logits = np.asarray(logits, dtype=float).reshape(-1)
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)
    yt = y.astype(int)
    acc = float(np.mean(pred == yt)) if yt.size else float("nan")
    tp = int(np.sum((pred == 1) & (yt == 1)))
    tn = int(np.sum((pred == 0) & (yt == 0)))
    fp = int(np.sum((pred == 1) & (yt == 0)))
    fn = int(np.sum((pred == 0) & (yt == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    auc = None
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(yt, prob)) if yt.size else None
    except Exception:
        auc = None
    return {
        "accuracy": acc,
        "f1": float(f1),
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }, {"y_true": yt, "y_pred": pred, "prob": prob}


def _plot_loss(hist):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(hist.get("train", []), label="train", color="blue")
    ax.plot(hist.get("val", []), label="val", color="orange")
    ax.set_title("loss 曲线")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_pred_reg(y, p):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    n = min(300, int(len(y)))
    fig = Figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.plot(y[:n], color="blue", linewidth=1.2, label="true")
    ax.plot(p[:n], color="red", linewidth=1.2, label="pred")
    ax.set_title("预测 vs 真实")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_confusion(cm_dict):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    tp = int(cm_dict.get("tp", 0))
    tn = int(cm_dict.get("tn", 0))
    fp = int(cm_dict.get("fp", 0))
    fn = int(cm_dict.get("fn", 0))
    mat = np.array([[tn, fp], [fn, tp]], dtype=int)
    fig = Figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111)
    ax.imshow(mat, cmap="Blues")
    ax.set_title("混淆矩阵")
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center")
    fig.tight_layout()
    return fig


def _plot_feature_mean_std(feature_names, mean, std):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    if not feature_names:
        return None
    df = pd.DataFrame({"feature": feature_names, "mean": mean, "std": std}).sort_values("std", ascending=False).head(20)
    fig = Figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.barh(df["feature"].values[::-1], df["std"].values[::-1], color="green")
    ax.set_title("特征标准差 Top")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def run_dl_lab(params, uploaded_file, dirs):
    toolkit, torch_err = _torch_available()
    if toolkit is None:
        return {"ok": False, "error": "缺少 torch: " + str(torch_err)}
    try:
        mode = str(params.get("data_mode") or "Tushare (CN)")
        if mode == "CSV 上传":
            df_all = _load_ohlcv_csv(uploaded_file)
        elif mode == "yfinance":
            df_all = _download_yfinance(params.get("tickers") or [], params.get("start_date"), params.get("end_date"), params.get("freq"), dirs=dirs)
        else:
            df_all = _download_tushare(params.get("tickers") or [], params.get("start_date"), params.get("end_date"), params.get("freq"), dirs=dirs)

        if df_all is None or df_all.empty:
            return {"ok": False, "error": "数据为空"}

        lookback = int(params.get("lookback") or 30)
        task = params.get("task")
        use_ohlcv = bool(params.get("use_ohlcv"))
        use_ma = bool(params.get("use_ma")) if "use_ma" in params else bool(params.get("use_indicators"))
        use_rsi = bool(params.get("use_rsi")) if "use_rsi" in params else bool(params.get("use_indicators"))
        use_macd = bool(params.get("use_macd")) if "use_macd" in params else bool(params.get("use_indicators"))
        use_volatility = bool(params.get("use_volatility")) if "use_volatility" in params else bool(params.get("use_indicators"))
        quick = bool(params.get("quick_mode"))
        model_name = params.get("model_name")

        X_parts = []
        y_parts = []
        meta_parts = []
        feature_names = None
        for ticker, g in df_all.groupby("ticker"):
            g = g.sort_index()
            if quick and len(g) > 1200:
                g = g.iloc[-1200:]
            feat = _build_features(g, use_ohlcv, use_ma, use_rsi, use_macd, use_volatility)
            if "close" in g.columns:
                close = g["close"]
            elif "adj_close" in g.columns:
                close = g["adj_close"]
            else:
                continue
            X, y, idx, names = _make_samples(feat, close, lookback, task)
            if X.shape[0] < 50:
                continue
            feature_names = names
            X_parts.append(X)
            y_parts.append(y)
            meta_parts.append(pd.DataFrame({"time": pd.to_datetime(idx, errors="coerce"), "ticker": str(ticker)}))
        if not X_parts:
            return {"ok": False, "error": "样本不足"}
        X_all = np.concatenate(X_parts, axis=0)
        y_all = np.concatenate(y_parts, axis=0)
        meta_all = pd.concat(meta_parts, axis=0, ignore_index=True) if meta_parts else pd.DataFrame()
        if not meta_all.empty and len(meta_all) == X_all.shape[0]:
            sort_idx = meta_all.sort_values(["time", "ticker"], kind="mergesort").index.to_numpy()
            X_all = X_all[sort_idx]
            y_all = y_all[sort_idx]
            meta_all = meta_all.iloc[sort_idx].reset_index(drop=True)

        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _time_split(X_all, y_all, train_ratio=0.7, val_ratio=0.15)
        (X_tr, X_va, X_te), mean, std = _standardize(X_tr, [X_tr, X_va, X_te])

        torch = toolkit["torch"]
        TensorDataset = toolkit["TensorDataset"]
        DataLoader = toolkit["DataLoader"]

        def to_loader(X, y, bs):
            ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
            return DataLoader(ds, batch_size=int(bs), shuffle=False, drop_last=False)

        bs = int(params.get("batch_size") or 64)
        loaders = {
            "train": to_loader(X_tr, y_tr, bs),
            "val": to_loader(X_va, y_va, bs),
            "test": to_loader(X_te, y_te, bs),
        }

        input_dim = int(X_all.shape[-1])
        model = _build_model(
            toolkit,
            model_name=model_name,
            input_dim=input_dim,
            lookback=lookback,
            task=task,
            hidden_size=params.get("hidden_size") or 64,
            layers=params.get("layers") or 1,
            dropout=params.get("dropout") or 0.0,
        )

        epochs = int(params.get("epochs") or 10)
        if quick:
            epochs = min(epochs, 5)
        model, hist, device = _train_loop(toolkit, model, loaders, task=task, epochs=epochs, lr=params.get("lr") or 1e-3)

        y_true, y_pred = _predict(toolkit, model, loaders["test"], device)
        figs = {"loss": _plot_loss(hist), "feature": _plot_feature_mean_std(feature_names or [], mean, std)}
        report = {"model": str(model_name), "task": str(task), "n_train": int(len(X_tr)), "n_val": int(len(X_va)), "n_test": int(len(X_te))}
        extra = {}
        if str(task).startswith("分类"):
            met, extra = _metrics_classification(y_true, y_pred)
            report.update(met)
            figs["confusion"] = _plot_confusion(met)
        else:
            met = _metrics_regression(y_true, y_pred)
            report.update(met)
            figs["pred"] = _plot_pred_reg(y_true, y_pred)

        config = dict(params)
        config["feature_names"] = feature_names or []
        config["standardize_mean"] = mean.tolist()
        config["standardize_std"] = std.tolist()

        return {
            "ok": True,
            "model_obj": model,
            "history": hist,
            "report": report,
            "figures": figs,
            "config": config,
            "test_pred": {"y_true": y_true.tolist(), "y_pred": y_pred.tolist(), "extra": extra},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def export_dl_lab(result, dirs, tag):
    toolkit, torch_err = _torch_available()
    if toolkit is None:
        return []
    torch = toolkit["torch"]
    if not isinstance(result, dict) or not result.get("ok"):
        return []
    payload = {"kind": "dl", "tag": str(tag or ""), "model": (result.get("report") or {}).get("model"), "task": (result.get("report") or {}).get("task")}
    suffix = io_utils.safe_filename(tag) if str(tag or "").strip() else io_utils.stable_hash(payload)
    out = []
    model = result.get("model_obj")
    if model is None:
        return []
    model_path = Path(dirs["models"]) / ("dl_model_" + suffix + ".pt")
    torch.save(model.state_dict(), model_path)
    out.append(model_path)

    config_path = Path(dirs["models"]) / ("dl_config_" + suffix + ".json")
    cfg = result.get("config") or {}
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    out.append(config_path)

    report_path_csv = Path(dirs["tables"]) / ("dl_report_" + suffix + ".csv")
    report_path_json = Path(dirs["tables"]) / ("dl_report_" + suffix + ".json")
    rep = result.get("report") or {}
    pd.DataFrame([rep]).to_csv(report_path_csv, index=False, encoding="utf-8-sig")
    report_path_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    out.extend([report_path_csv, report_path_json])

    pred_path = Path(dirs["tables"]) / ("dl_pred_" + suffix + ".json")
    pred_path.write_text(json.dumps(result.get("test_pred") or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    out.append(pred_path)

    figs = result.get("figures") or {}
    for k, fig in figs.items():
        if fig is None:
            continue
        p = Path(dirs["figures"]) / ("dl_" + str(k) + "_" + suffix + ".png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        out.append(p)
    return out
