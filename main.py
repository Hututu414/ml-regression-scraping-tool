import json
import importlib.util
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk

import pandas as pd

from src import dl_lab, io_utils, stock_analytics, text_mining

ROOT = Path(__file__).resolve().parent
DIRS = io_utils.ensure_output_dirs(ROOT)

PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_TRUSTED_HOST = "pypi.tuna.tsinghua.edu.cn"
PIP_IMPORT_MAP = {
    "beautifulsoup4": "bs4",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "snownlp": "snownlp",
    "wordcloud": "wordcloud",
    "tushare": "tushare",
    "yfinance": "yfinance",
    "openpyxl": "openpyxl",
    "torch": "torch",
    "scipy": "scipy",
    "lxml": "lxml",
    "jieba": "jieba",
    "requests": "requests",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
}


def _req_name(line):
    s = str(line or "").strip()
    if not s or s.startswith(("-", "#")):
        return None
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    for op in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
        if op in s:
            s = s.split(op, 1)[0].strip()
            break
    if "[" in s:
        s = s.split("[", 1)[0].strip()
    return s or None


def parse_requirements(path):
    p = Path(path)
    if not p.exists():
        return []
    names = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        name = _req_name(line)
        if name:
            names.append(name)
    uniq = []
    seen = set()
    for n in names:
        k = n.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    return uniq


def missing_packages(packages):
    out = []
    for pkg in packages:
        name = str(pkg).strip()
        if not name:
            continue
        imp = PIP_IMPORT_MAP.get(name.lower(), name.replace("-", "_"))
        try:
            ok = importlib.util.find_spec(imp) is not None
        except Exception:
            ok = False
        if not ok:
            out.append(name)
    return out


def in_venv():
    try:
        return hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix
    except Exception:
        return False


class QueueLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = str(record.getMessage())
        self.q.put({"type": "log", "message": msg})


def setup_logging(event_queue):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    qh = QueueLogHandler(event_queue)
    qh.setFormatter(fmt)
    logger.addHandler(qh)
    log_file = Path(DIRS["logs"]) / "app.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logger


def open_path(path):
    p = Path(path)
    if not p.exists():
        return False
    try:
        os.startfile(str(p))
        return True
    except Exception:
        try:
            import webbrowser

            return webbrowser.open(p.as_uri())
        except Exception:
            return False


def pretty_json(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)

def as_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame()


class FigureGrid:
    def __init__(self, parent, rows, cols):
        self.parent = parent
        self.rows = int(rows)
        self.cols = int(cols)
        self.canvases = []
        self.widgets = []

    def clear(self):
        for w in self.widgets:
            try:
                w.destroy()
            except Exception:
                pass
        self.widgets = []
        self.canvases = []

    def show_figures(self, figures, keys, max_figs=None):
        self.clear()
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            lbl = ttk.Label(self.parent, text="缺少 matplotlib TkAgg: " + str(e))
            lbl.grid(row=0, column=0, sticky="w")
            self.widgets.append(lbl)
            return
        figs = []
        for k in keys:
            fig = figures.get(k)
            if fig is not None:
                figs.append((k, fig))
        if max_figs is not None:
            figs = figs[: int(max_figs)]
        for i, (k, fig) in enumerate(figs):
            r = i // self.cols
            c = i % self.cols
            frame = ttk.Frame(self.parent)
            frame.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            self.parent.grid_rowconfigure(r, weight=1)
            self.parent.grid_columnconfigure(c, weight=1)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            widget = canvas.get_tk_widget()
            widget.pack(fill="both", expand=True)
            try:
                canvas.draw()
                try:
                    fig.tight_layout(pad=2.2)
                except Exception:
                    pass
                canvas.draw()
            except Exception:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
            self.canvases.append(canvas)
            self.widgets.append(frame)


class StockTab(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.last_result = None
        self._vars()
        self._ui()

    def _vars(self):
        self.data_mode = StringVar(value="Tushare (CN)")
        self.ticker = StringVar(value="600519.SH")
        self.benchmark = StringVar(value="000300.SH")
        self.asset_type = StringVar(value="stock")
        self.benchmark_type = StringVar(value="index")
        self.start_date = StringVar(value="2020-01-01")
        self.end_date = StringVar(value=pd.Timestamp.today().strftime("%Y-%m-%d"))
        self.freq = StringVar(value="日")
        self.price_field = StringVar(value="Close")
        self.log_return = BooleanVar(value=True)
        self.annualize = BooleanVar(value=False)
        self.rf_annual = DoubleVar(value=0.0)
        self.capm_excess = BooleanVar(value=False)
        self.enable_ttest = BooleanVar(value=False)
        self.asset_csv = StringVar(value="")
        self.market_csv = StringVar(value="")
        self.tag = StringVar(value="")
        self.export_format = StringVar(value="Excel")

    def _ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        left.grid_columnconfigure(1, weight=1)
        r = 0

        ttk.Label(left, text="数据源").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.mode_cb = ttk.Combobox(left, textvariable=self.data_mode, values=["Tushare (CN)", "yfinance", "CSV 上传"], state="readonly", width=16)
        self.mode_cb.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="标的 ts_code/ticker").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.ticker).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="市场基准 ts_code/ticker").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.benchmark).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="标的类型(Tushare)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.asset_type_cb = ttk.Combobox(left, textvariable=self.asset_type, values=["auto", "stock", "index"], state="readonly", width=16)
        self.asset_type_cb.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="基准类型(Tushare)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.benchmark_type_cb = ttk.Combobox(left, textvariable=self.benchmark_type, values=["auto", "stock", "index"], state="readonly", width=16)
        self.benchmark_type_cb.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="开始日期 YYYY-MM-DD").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.start_date).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="结束日期 YYYY-MM-DD").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.end_date).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="频率").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(left, textvariable=self.freq, values=["日", "周"], state="readonly", width=16).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="价格字段").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(left, textvariable=self.price_field, values=["Adj Close", "Close"], state="readonly", width=16).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Checkbutton(left, text="对数收益率", variable=self.log_return).grid(row=r, column=0, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(left, text="是否年化", variable=self.annualize).grid(row=r, column=1, sticky="w", padx=6, pady=2)
        r += 1

        ttk.Label(left, text="无风险利率(年化,小数)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.rf_annual).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Checkbutton(left, text="使用超额收益 CAPM", variable=self.capm_excess).grid(row=r, column=0, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(left, text="差分均值 t 检验", variable=self.enable_ttest).grid(row=r, column=1, sticky="w", padx=6, pady=2)
        r += 1

        ttk.Label(left, text="标的 CSV(离线)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        f1 = ttk.Frame(left)
        f1.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        f1.grid_columnconfigure(0, weight=1)
        self.asset_entry = ttk.Entry(f1, textvariable=self.asset_csv)
        self.asset_entry.grid(row=0, column=0, sticky="ew")
        self.asset_btn = ttk.Button(f1, text="选择", command=self._select_asset_csv)
        self.asset_btn.grid(row=0, column=1, padx=4)
        r += 1

        ttk.Label(left, text="基准 CSV(离线)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        f2 = ttk.Frame(left)
        f2.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        f2.grid_columnconfigure(0, weight=1)
        self.market_entry = ttk.Entry(f2, textvariable=self.market_csv)
        self.market_entry.grid(row=0, column=0, sticky="ew")
        self.market_btn = ttk.Button(f2, text="选择", command=self._select_market_csv)
        self.market_btn.grid(row=0, column=1, padx=4)
        r += 1

        ttk.Label(left, text="输出 tag(可选)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.tag).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="导出格式").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(left, textvariable=self.export_format, values=["Excel", "CSV"], state="readonly", width=16).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        btn_row = ttk.Frame(left)
        btn_row.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=8)
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        btn_row.grid_columnconfigure(2, weight=1)
        self.run_btn = ttk.Button(btn_row, text="运行分析", command=self.run)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=3)
        self.export_btn = ttk.Button(btn_row, text="一键导出", command=self.export)
        self.export_btn.grid(row=0, column=1, sticky="ew", padx=3)
        self.fig_btn = ttk.Button(btn_row, text="保存图像", command=self.save_figures)
        self.fig_btn.grid(row=0, column=2, sticky="ew", padx=3)
        r += 1

        btn_row2 = ttk.Frame(left)
        btn_row2.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        btn_row2.grid_columnconfigure(0, weight=1)
        btn_row2.grid_columnconfigure(1, weight=1)
        ttk.Button(btn_row2, text="打开 outputs", command=self.controller.open_outputs).grid(row=0, column=0, sticky="ew", padx=3)
        ttk.Button(btn_row2, text="打开 figures", command=lambda: self.controller.open_folder(self.controller.dirs["figures"])).grid(row=0, column=1, sticky="ew", padx=3)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        vpaned = ttk.Panedwindow(right, orient="vertical")
        vpaned.grid(row=0, column=0, sticky="nsew")
        chart_frame = ttk.Frame(vpaned)
        summary_frame = ttk.Frame(vpaned)
        vpaned.add(chart_frame, weight=3)
        vpaned.add(summary_frame, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.fig_grid = FigureGrid(chart_frame, rows=2, cols=2)
        self.summary = ScrolledText(summary_frame, height=12)
        self.summary.pack(fill="both", expand=True)
        self._set_summary("等待运行...")

        self.data_mode.trace_add("write", lambda *args: self._sync_mode_state())
        self._sync_mode_state()

    def _select_asset_csv(self):
        p = filedialog.askopenfilename(title="选择标的 CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.asset_csv.set(p)

    def _select_market_csv(self):
        p = filedialog.askopenfilename(title="选择市场基准 CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.market_csv.set(p)

    def _set_summary(self, text):
        if not hasattr(self, "summary"):
            return
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", str(text))
        self.summary.configure(state="disabled")

    def _build_params(self):
        return {
            "data_mode": str(self.data_mode.get()),
            "ticker": str(self.ticker.get()).strip(),
            "benchmark_ticker": str(self.benchmark.get()).strip(),
            "asset_type": str(self.asset_type.get()).strip() or "auto",
            "benchmark_type": str(self.benchmark_type.get()).strip() or "auto",
            "start_date": str(self.start_date.get()).strip(),
            "end_date": str(self.end_date.get()).strip(),
            "freq": str(self.freq.get()),
            "price_field": str(self.price_field.get()),
            "log_return": bool(self.log_return.get()),
            "annualize": bool(self.annualize.get()),
            "rf_annual": float(self.rf_annual.get() or 0.0),
            "capm_excess": bool(self.capm_excess.get()),
            "enable_ttest": bool(self.enable_ttest.get()),
        }

    def run(self):
        self._set_running(True)
        if not self.controller.ensure_packages(["pandas", "numpy", "matplotlib", "openpyxl"]):
            self._set_running(False)
            return
        params = self._build_params()
        asset_file = str(self.asset_csv.get()).strip() or None
        market_file = str(self.market_csv.get()).strip() or None
        if params.get("data_mode") == "CSV 上传" and (not asset_file or not market_file):
            self._set_running(False)
            messagebox.showwarning("提示", "CSV 模式需要选择标的与基准 CSV")
            return
        if params.get("data_mode") == "Tushare (CN)":
            if not self.controller.ensure_packages(["tushare"]):
                self._set_running(False)
                return
        if params.get("data_mode") == "yfinance":
            if not self.controller.ensure_packages(["yfinance"]):
                self._set_running(False)
                return

        def job():
            logging.getLogger("ui.stock").info("开始 Stock Analytics")
            return stock_analytics.run_stock_analytics(params=params, asset_csv_file=asset_file, market_csv_file=market_file, dirs=self.controller.dirs)

        self.controller.run_task("stock_run", job, self.on_result)

    def export(self):
        if not isinstance(self.last_result, dict) or not self.last_result.get("ok"):
            messagebox.showwarning("提示", "请先运行分析")
            return
        tag = str(self.tag.get()).strip()
        export_format = str(self.export_format.get())

        def job():
            paths = stock_analytics.export_stock_analytics(result=self.last_result, dirs=self.controller.dirs, tag=tag, export_format=export_format)
            return {"ok": True, "paths": [str(p) for p in paths]}

        def done(res):
            if not isinstance(res, dict) or not res.get("ok"):
                messagebox.showerror("导出失败", str((res or {}).get("error") or "导出失败"))
                return
            paths = res.get("paths") or []
            self.controller.log_info("导出完成: " + str(len(paths)) + " 个文件")
            self._append_summary("\n\n导出文件:\n" + "\n".join(paths))

        self.controller.run_task("stock_export", job, done)

    def save_figures(self):
        if not isinstance(self.last_result, dict) or not self.last_result.get("ok"):
            messagebox.showwarning("提示", "请先运行分析")
            return
        tag = str(self.tag.get()).strip()

        def job():
            paths = stock_analytics.save_stock_figures(result=self.last_result, dirs=self.controller.dirs, tag=tag)
            return {"ok": True, "paths": [str(p) for p in paths]}

        def done(res):
            if not isinstance(res, dict) or not res.get("ok"):
                messagebox.showerror("保存失败", str((res or {}).get("error") or "保存失败"))
                return
            paths = res.get("paths") or []
            self.controller.log_info("图像保存完成: " + str(len(paths)) + " 张")
            self._append_summary("\n\n图像文件:\n" + "\n".join(paths))

        self.controller.run_task("stock_fig", job, done)

    def on_result(self, result):
        self._set_running(False)
        self.last_result = result
        if not isinstance(result, dict) or not result.get("ok"):
            err = str((result or {}).get("error") or "运行失败")
            self._set_summary("运行失败:\n" + err)
            try:
                self.fig_grid.clear()
            except Exception:
                pass
            messagebox.showerror("运行失败", err)
            return
        metrics = result.get("metrics") or {}
        data_df = as_df(result.get("data"))
        txt = "回归结果:\n" + pretty_json(metrics.get("regression") or {}) + "\n\n差分统计:\n" + pretty_json(metrics.get("diff_stats") or {})
        if isinstance(data_df, pd.DataFrame) and not data_df.empty:
            txt += "\n\n对齐数据预览(前 20 行):\n" + data_df.head(20).to_string(index=False)
        self._set_summary(txt)
        figs = result.get("figures") or {}
        try:
            self.fig_grid.show_figures(figs, keys=["price", "returns", "diff", "regression"], max_figs=4)
        except Exception:
            pass

    def _append_summary(self, text):
        if not hasattr(self, "summary"):
            return
        self.summary.configure(state="normal")
        self.summary.insert("end", str(text))
        self.summary.see("end")
        self.summary.configure(state="disabled")

    def _set_running(self, running):
        state = "disabled" if running else "normal"
        for b in [getattr(self, "run_btn", None), getattr(self, "export_btn", None), getattr(self, "fig_btn", None)]:
            if b is not None:
                try:
                    b.configure(state=state)
                except Exception:
                    pass

    def _sync_mode_state(self):
        mode = str(self.data_mode.get())
        csv_state = "normal" if mode == "CSV 上传" else "disabled"
        tushare_state = "readonly" if mode == "Tushare (CN)" else "disabled"
        for w in [getattr(self, "asset_entry", None), getattr(self, "market_entry", None), getattr(self, "asset_btn", None), getattr(self, "market_btn", None)]:
            if w is not None:
                try:
                    w.configure(state=csv_state)
                except Exception:
                    pass
        for w in [getattr(self, "asset_type_cb", None), getattr(self, "benchmark_type_cb", None)]:
            if w is not None:
                try:
                    w.configure(state=tushare_state)
                except Exception:
                    pass


class TextTab(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.last_result = None
        self._vars()
        self._ui()

    def _vars(self):
        self.source_mode = StringVar(value="自动抓取")
        self.keyword = StringVar(value="600519")
        self.pages = IntVar(value=1)
        self.max_items = IntVar(value=50)
        self.start_time = StringVar(value="")
        self.end_time = StringVar(value="")
        self.use_cache = BooleanVar(value=True)
        self.sleep_seconds = DoubleVar(value=1.0)
        self.top_n = IntVar(value=30)
        self.n_topics = IntVar(value=6)
        self.agg = StringVar(value="日")
        self.upload_path = StringVar(value="")
        self.tag = StringVar(value="")

    def _ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        left.grid_columnconfigure(1, weight=1)
        r = 0

        ttk.Label(left, text="数据源").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.source_cb = ttk.Combobox(left, textvariable=self.source_mode, values=["自动抓取", "CSV 上传", "HTML 上传"], state="readonly", width=16)
        self.source_cb.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="股票代码/名称关键词").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.keyword).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="抓取页数").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.pages).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="抓取条数上限").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.max_items).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="开始时间(可选)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.start_time).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="结束时间(可选)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.end_time).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Checkbutton(left, text="使用本地缓存", variable=self.use_cache).grid(row=r, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(left, text="请求间隔(s)").grid(row=r, column=1, sticky="w", padx=6, pady=2)
        r += 1

        ttk.Entry(left, textvariable=self.sleep_seconds).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="Cookie/Headers(可选)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.headers_box = ScrolledText(left, height=5)
        self.headers_box.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="TopN").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.top_n).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="主题数 n_topics").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.n_topics).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="聚合").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(left, textvariable=self.agg, values=["日", "周"], state="readonly", width=16).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="上传文件(可选)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        up = ttk.Frame(left)
        up.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        up.grid_columnconfigure(0, weight=1)
        self.upload_entry = ttk.Entry(up, textvariable=self.upload_path)
        self.upload_entry.grid(row=0, column=0, sticky="ew")
        self.upload_btn = ttk.Button(up, text="选择", command=self._select_upload)
        self.upload_btn.grid(row=0, column=1, padx=4)
        r += 1

        ttk.Label(left, text="输出 tag(可选)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.tag).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        btn_row = ttk.Frame(left)
        btn_row.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=8)
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        self.run_btn = ttk.Button(btn_row, text="开始抓取与分析", command=self.run)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=3)
        self.export_btn = ttk.Button(btn_row, text="导出到 outputs", command=self.export)
        self.export_btn.grid(row=0, column=1, sticky="ew", padx=3)
        r += 1

        btn_row2 = ttk.Frame(left)
        btn_row2.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        btn_row2.grid_columnconfigure(0, weight=1)
        btn_row2.grid_columnconfigure(1, weight=1)
        ttk.Button(btn_row2, text="打开 outputs", command=self.controller.open_outputs).grid(row=0, column=0, sticky="ew", padx=3)
        ttk.Button(btn_row2, text="打开 figures", command=lambda: self.controller.open_folder(self.controller.dirs["figures"])).grid(row=0, column=1, sticky="ew", padx=3)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        vpaned = ttk.Panedwindow(right, orient="vertical")
        vpaned.grid(row=0, column=0, sticky="nsew")
        chart_frame = ttk.Frame(vpaned)
        summary_frame = ttk.Frame(vpaned)
        vpaned.add(chart_frame, weight=3)
        vpaned.add(summary_frame, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.fig_grid = FigureGrid(chart_frame, rows=2, cols=2)
        self.summary = ScrolledText(summary_frame, height=12)
        self.summary.pack(fill="both", expand=True)
        self._set_summary("等待运行...")

        self.source_mode.trace_add("write", lambda *args: self._sync_source_state())
        self._sync_source_state()

    def _select_upload(self):
        mode = str(self.source_mode.get())
        if mode == "HTML 上传":
            types = [("HTML", "*.html;*.htm"), ("All", "*.*")]
        else:
            types = [("CSV", "*.csv"), ("All", "*.*")]
        p = filedialog.askopenfilename(title="选择文件", filetypes=types)
        if p:
            self.upload_path.set(p)

    def _set_summary(self, text):
        if not hasattr(self, "summary"):
            return
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", str(text))
        self.summary.configure(state="disabled")

    def _append_summary(self, text):
        if not hasattr(self, "summary"):
            return
        self.summary.configure(state="normal")
        self.summary.insert("end", str(text))
        self.summary.see("end")
        self.summary.configure(state="disabled")

    def _sync_source_state(self):
        mode = str(self.source_mode.get())
        upload_state = "normal" if mode in ["CSV 上传", "HTML 上传"] else "disabled"
        for w in [getattr(self, "upload_entry", None), getattr(self, "upload_btn", None)]:
            if w is not None:
                try:
                    w.configure(state=upload_state)
                except Exception:
                    pass

    def _build_params(self):
        headers_text = ""
        if hasattr(self, "headers_box"):
            headers_text = self.headers_box.get("1.0", "end").strip()
        return {
            "source_mode": str(self.source_mode.get()),
            "keyword": str(self.keyword.get()).strip(),
            "pages": int(self.pages.get() or 1),
            "max_items": int(self.max_items.get() or 50),
            "start_time": str(self.start_time.get()).strip() or None,
            "end_time": str(self.end_time.get()).strip() or None,
            "use_cache": bool(self.use_cache.get()),
            "sleep_seconds": max(1.0, float(self.sleep_seconds.get() or 1.0)),
            "headers_text": headers_text,
            "top_n": int(self.top_n.get() or 30),
            "n_topics": int(self.n_topics.get() or 6),
            "agg": str(self.agg.get()),
        }

    def _set_running(self, running):
        state = "disabled" if running else "normal"
        for b in [getattr(self, "run_btn", None), getattr(self, "export_btn", None)]:
            if b is not None:
                try:
                    b.configure(state=state)
                except Exception:
                    pass

    def run(self):
        self._set_running(True)
        if not self.controller.ensure_packages(["pandas", "numpy", "requests"]):
            self._set_running(False)
            return
        params = self._build_params()
        params["_progress"] = self.controller.make_progress_cb("text")
        mode = params.get("source_mode")
        upload = str(self.upload_path.get()).strip() or None
        if mode in ["CSV 上传", "HTML 上传"] and not upload:
            self._set_running(False)
            messagebox.showwarning("提示", "上传模式需要选择文件")
            return
        if mode == "自动抓取" and not params.get("keyword"):
            self._set_running(False)
            messagebox.showwarning("提示", "请输入关键词")
            return
        need = ["beautifulsoup4", "lxml", "jieba", "wordcloud", "scikit-learn"]
        if not self.controller.ensure_packages(need):
            self._set_running(False)
            return

        def job():
            logging.getLogger("ui.text").info("开始 Text Mining")
            return text_mining.run_text_mining(params=params, uploaded_file=upload, dirs=self.controller.dirs)

        self.controller.run_task("text_run", job, self.on_result)

    def on_result(self, result):
        self._set_running(False)
        self.last_result = result
        if not isinstance(result, dict) or not result.get("ok"):
            err = str((result or {}).get("error") or "运行失败")
            self._set_summary("运行失败:\n" + err)
            try:
                self.fig_grid.clear()
            except Exception:
                pass
            messagebox.showerror("运行失败", err)
            return
        raw_df = as_df(result.get("raw"))
        clean_df = as_df(result.get("clean"))
        analytics = result.get("analytics") or {}
        txt = "原始条数: " + str(int(getattr(raw_df, "shape", [0])[0])) + "\n清洗后条数: " + str(int(getattr(clean_df, "shape", [0])[0]))
        txt += "\n\n词频 Top10:\n" + pretty_json((analytics.get("freq") or [])[:10])
        txt += "\n\nTF-IDF Top10:\n" + pretty_json((analytics.get("tfidf") or [])[:10])
        txt += "\n\n主题摘要:\n" + pretty_json((analytics.get("topics") or [])[:6])
        self._set_summary(txt)
        figs = result.get("figures") or {}
        try:
            self.fig_grid.show_figures(figs, keys=["wordcloud", "freq", "trend", "topics"], max_figs=4)
        except Exception:
            pass

    def export(self):
        if not isinstance(self.last_result, dict) or not self.last_result.get("ok"):
            messagebox.showwarning("提示", "请先运行分析")
            return
        tag = str(self.tag.get()).strip()

        def job():
            paths = text_mining.export_text_mining(result=self.last_result, dirs=self.controller.dirs, tag=tag)
            return {"ok": True, "paths": [str(p) for p in paths]}

        def done(res):
            if not isinstance(res, dict) or not res.get("ok"):
                messagebox.showerror("导出失败", str((res or {}).get("error") or "导出失败"))
                return
            paths = res.get("paths") or []
            self.controller.log_info("导出完成: " + str(len(paths)) + " 个文件")
            self._append_summary("\n\n导出文件:\n" + "\n".join(paths))

        self.controller.run_task("text_export", job, done)


class DLTab(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.last_result = None
        self._vars()
        self._ui()

    def _vars(self):
        self.data_mode = StringVar(value="Tushare (CN)")
        self.tickers_text = StringVar(value="600519.SH")
        self.start_date = StringVar(value="2018-01-01")
        self.end_date = StringVar(value=pd.Timestamp.today().strftime("%Y-%m-%d"))
        self.freq = StringVar(value="日")
        self.task = StringVar(value="回归 预测下一期收益率")
        self.model_name = StringVar(value="LSTM")
        self.lookback = IntVar(value=30)
        self.use_ohlcv = BooleanVar(value=True)
        self.use_ma = BooleanVar(value=True)
        self.use_rsi = BooleanVar(value=True)
        self.use_macd = BooleanVar(value=True)
        self.use_volatility = BooleanVar(value=True)
        self.quick_mode = BooleanVar(value=True)
        self.epochs = IntVar(value=10)
        self.batch_size = IntVar(value=64)
        self.lr = DoubleVar(value=1e-3)
        self.hidden_size = IntVar(value=64)
        self.layers = IntVar(value=1)
        self.dropout = DoubleVar(value=0.1)
        self.upload_path = StringVar(value="")
        self.tag = StringVar(value="")

    def _ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        left.grid_columnconfigure(1, weight=1)
        r = 0

        ttk.Label(left, text="数据源").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.mode_cb = ttk.Combobox(left, textvariable=self.data_mode, values=["Tushare (CN)", "yfinance", "CSV 上传"], state="readonly", width=16)
        self.mode_cb.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="tickers(逗号分隔)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.tickers_text).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="开始日期").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.start_date).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="结束日期").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.end_date).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="频率").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(left, textvariable=self.freq, values=["日", "周"], state="readonly", width=16).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="任务").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(left, textvariable=self.task, values=["回归 预测下一期收益率", "分类 预测涨跌"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="模型").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(left, textvariable=self.model_name, values=["MLP", "LSTM", "1D-CNN", "TransformerEncoder"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="lookback").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.lookback).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Checkbutton(left, text="使用 OHLCV", variable=self.use_ohlcv).grid(row=r, column=0, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(left, text="快速演示模式", variable=self.quick_mode).grid(row=r, column=1, sticky="w", padx=6, pady=2)
        r += 1

        ind = ttk.LabelFrame(left, text="技术指标")
        ind.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        ind.grid_columnconfigure(0, weight=1)
        ind.grid_columnconfigure(1, weight=1)
        ttk.Checkbutton(ind, text="MA", variable=self.use_ma).grid(row=0, column=0, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(ind, text="RSI", variable=self.use_rsi).grid(row=0, column=1, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(ind, text="MACD", variable=self.use_macd).grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(ind, text="Volatility", variable=self.use_volatility).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        r += 1

        ttk.Label(left, text="epochs").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.epochs).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="batch size").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.batch_size).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="lr").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.lr).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="hidden size").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.hidden_size).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="layers").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.layers).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="dropout").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.dropout).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        ttk.Label(left, text="上传 OHLCV CSV").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        up = ttk.Frame(left)
        up.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        up.grid_columnconfigure(0, weight=1)
        self.upload_entry = ttk.Entry(up, textvariable=self.upload_path)
        self.upload_entry.grid(row=0, column=0, sticky="ew")
        self.upload_btn = ttk.Button(up, text="选择", command=self._select_upload)
        self.upload_btn.grid(row=0, column=1, padx=4)
        r += 1

        ttk.Label(left, text="输出 tag(可选)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(left, textvariable=self.tag).grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        r += 1

        btn_row = ttk.Frame(left)
        btn_row.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=8)
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        self.run_btn = ttk.Button(btn_row, text="开始训练", command=self.run)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=3)
        self.export_btn = ttk.Button(btn_row, text="保存模型与报告", command=self.export)
        self.export_btn.grid(row=0, column=1, sticky="ew", padx=3)
        r += 1

        btn_row2 = ttk.Frame(left)
        btn_row2.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        btn_row2.grid_columnconfigure(0, weight=1)
        btn_row2.grid_columnconfigure(1, weight=1)
        ttk.Button(btn_row2, text="打开 outputs", command=self.controller.open_outputs).grid(row=0, column=0, sticky="ew", padx=3)
        ttk.Button(btn_row2, text="打开 models", command=lambda: self.controller.open_folder(self.controller.dirs["models"])).grid(row=0, column=1, sticky="ew", padx=3)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        vpaned = ttk.Panedwindow(right, orient="vertical")
        vpaned.grid(row=0, column=0, sticky="nsew")
        chart_frame = ttk.Frame(vpaned)
        summary_frame = ttk.Frame(vpaned)
        vpaned.add(chart_frame, weight=3)
        vpaned.add(summary_frame, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.fig_grid = FigureGrid(chart_frame, rows=2, cols=2)
        self.summary = ScrolledText(summary_frame, height=12)
        self.summary.pack(fill="both", expand=True)
        self._set_summary("等待运行...")

        self.data_mode.trace_add("write", lambda *args: self._sync_mode_state())
        self._sync_mode_state()

    def _select_upload(self):
        p = filedialog.askopenfilename(title="选择 OHLCV CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.upload_path.set(p)

    def _sync_mode_state(self):
        mode = str(self.data_mode.get())
        upload_state = "normal" if mode == "CSV 上传" else "disabled"
        for w in [getattr(self, "upload_entry", None), getattr(self, "upload_btn", None)]:
            if w is not None:
                try:
                    w.configure(state=upload_state)
                except Exception:
                    pass

    def _set_summary(self, text):
        if not hasattr(self, "summary"):
            return
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("end", str(text))
        self.summary.configure(state="disabled")

    def _append_summary(self, text):
        if not hasattr(self, "summary"):
            return
        self.summary.configure(state="normal")
        self.summary.insert("end", str(text))
        self.summary.see("end")
        self.summary.configure(state="disabled")

    def _set_running(self, running):
        state = "disabled" if running else "normal"
        for b in [getattr(self, "run_btn", None), getattr(self, "export_btn", None)]:
            if b is not None:
                try:
                    b.configure(state=state)
                except Exception:
                    pass

    def _build_params(self):
        tickers = [t.strip() for t in str(self.tickers_text.get()).split(",") if t.strip()]
        return {
            "data_mode": str(self.data_mode.get()),
            "tickers": tickers,
            "start_date": str(self.start_date.get()).strip(),
            "end_date": str(self.end_date.get()).strip(),
            "freq": str(self.freq.get()),
            "task": str(self.task.get()),
            "model_name": str(self.model_name.get()),
            "lookback": int(self.lookback.get() or 30),
            "use_ohlcv": bool(self.use_ohlcv.get()),
            "use_ma": bool(self.use_ma.get()),
            "use_rsi": bool(self.use_rsi.get()),
            "use_macd": bool(self.use_macd.get()),
            "use_volatility": bool(self.use_volatility.get()),
            "quick_mode": bool(self.quick_mode.get()),
            "epochs": int(self.epochs.get() or 10),
            "batch_size": int(self.batch_size.get() or 64),
            "lr": float(self.lr.get() or 1e-3),
            "hidden_size": int(self.hidden_size.get() or 64),
            "layers": int(self.layers.get() or 1),
            "dropout": float(self.dropout.get() or 0.0),
        }

    def run(self):
        self._set_running(True)
        if not self.controller.ensure_packages(["pandas", "numpy", "matplotlib"]):
            self._set_running(False)
            return
        params = self._build_params()
        mode = params.get("data_mode")
        upload = str(self.upload_path.get()).strip() or None
        if mode == "CSV 上传" and not upload:
            self._set_running(False)
            messagebox.showwarning("提示", "CSV 模式需要选择文件")
            return
        if not params.get("tickers") and mode != "CSV 上传":
            self._set_running(False)
            messagebox.showwarning("提示", "请输入至少一个 ticker/ts_code")
            return
        if mode == "Tushare (CN)":
            if not self.controller.ensure_packages(["tushare"]):
                self._set_running(False)
                return
        if mode == "yfinance":
            if not self.controller.ensure_packages(["yfinance"]):
                self._set_running(False)
                return
        if not self.controller.ensure_packages(["torch"]):
            self._set_running(False)
            return

        def job():
            logging.getLogger("ui.dl").info("开始 Deep Learning Lab")
            return dl_lab.run_dl_lab(params=params, uploaded_file=upload, dirs=self.controller.dirs)

        self.controller.run_task("dl_run", job, self.on_result)

    def on_result(self, result):
        self._set_running(False)
        self.last_result = result
        if not isinstance(result, dict) or not result.get("ok"):
            err = str((result or {}).get("error") or "运行失败")
            self._set_summary("运行失败:\n" + err)
            try:
                self.fig_grid.clear()
            except Exception:
                pass
            messagebox.showerror("运行失败", err)
            return
        report = result.get("report") or {}
        txt = "评估报告:\n" + pretty_json(report)
        self._set_summary(txt)
        figs = result.get("figures") or {}
        keys = ["loss", "pred", "confusion", "feature"]
        try:
            self.fig_grid.show_figures(figs, keys=keys, max_figs=4)
        except Exception:
            pass

    def export(self):
        if not isinstance(self.last_result, dict) or not self.last_result.get("ok"):
            messagebox.showwarning("提示", "请先训练或运行")
            return
        tag = str(self.tag.get()).strip()

        def job():
            paths = dl_lab.export_dl_lab(result=self.last_result, dirs=self.controller.dirs, tag=tag)
            return {"ok": True, "paths": [str(p) for p in paths]}

        def done(res):
            if not isinstance(res, dict) or not res.get("ok"):
                messagebox.showerror("保存失败", str((res or {}).get("error") or "保存失败"))
                return
            paths = res.get("paths") or []
            self.controller.log_info("保存完成: " + str(len(paths)) + " 个文件")
            self._append_summary("\n\n保存文件:\n" + "\n".join(paths))

        self.controller.run_task("dl_export", job, done)


class App(Tk):
    def __init__(self):
        super().__init__()
        self.event_queue = queue.Queue()
        self.logger = setup_logging(self.event_queue)
        self.dirs = DIRS
        self._active_tasks = 0
        self.title("作业集成")
        self.geometry("1200x900")
        self.minsize(1000, 700)
        self._ui()
        self.after(120, self._poll_events)

    def _ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.stock_tab = StockTab(self.notebook, self)
        self.text_tab = TextTab(self.notebook, self)
        self.dl_tab = DLTab(self.notebook, self)
        self.notebook.add(self.stock_tab, text="股票数据与回归")
        self.notebook.add(self.text_tab, text="爬虫 + 文本分析")
        self.notebook.add(self.dl_tab, text="深度学习示例")

        bottom = ttk.Frame(root)
        bottom.grid(row=1, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(1, weight=1)

        bar = ttk.Frame(bottom)
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        bar.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(bar, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="打开 outputs", command=self.open_outputs).grid(row=0, column=1, padx=6)
        ttk.Button(bar, text="安装依赖(清华源)", command=self.install_deps).grid(row=0, column=2, padx=6)
        ttk.Button(bar, text="使用手册", command=self.open_manual).grid(row=0, column=3)

        self.log_text = ScrolledText(bottom, height=10)
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.log_text.configure(state="disabled")
        self.progress.configure(mode="indeterminate", value=0)

    def open_outputs(self):
        self.open_folder(self.dirs["outputs"])

    def open_folder(self, path):
        ok = open_path(path)
        if not ok:
            messagebox.showwarning("提示", "无法打开: " + str(path))

    def open_manual(self):
        p = ROOT / "docs" / "USER_MANUAL.md"
        if not p.exists():
            messagebox.showwarning("提示", "未找到使用手册: " + str(p))
            return
        ok = open_path(p)
        if not ok:
            messagebox.showwarning("提示", "无法打开使用手册: " + str(p))

    def log_info(self, msg):
        logging.getLogger("ui").info(str(msg))

    def ensure_packages(self, packages, title="缺少依赖"):
        miss = missing_packages(packages)
        if not miss:
            return True
        msg = "检测到缺少依赖:\n" + "\n".join(miss) + "\n\n是否安装?\n是: 安装 requirements.txt 全部\n否: 仅安装缺失\n取消: 不安装"
        choice = messagebox.askyesnocancel(title, msg)
        if choice is None:
            return False
        install_all = bool(choice)
        self.install_deps(install_all=install_all, packages=miss if not install_all else None)
        return False

    def _pip_install(self, install_all=False, packages=None):
        log = logging.getLogger("pip")
        args = [sys.executable, "-m", "pip", "install", "--upgrade"]
        if not in_venv():
            args.append("--user")
        args.extend(["-i", PIP_INDEX_URL, "--trusted-host", PIP_TRUSTED_HOST, "--timeout", "30"])
        if install_all:
            req = ROOT / "requirements.txt"
            args.extend(["-r", str(req)])
            log.info("开始安装: requirements.txt")
        else:
            pkgs = [str(p) for p in (packages or []) if str(p).strip()]
            if not pkgs:
                log.info("无需安装")
                return {"ok": True}
            args.extend(pkgs)
            log.info("开始安装: " + " ".join(pkgs))
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    s = line.rstrip("\r\n")
                    if s:
                        log.info(s)
        finally:
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError("pip 安装失败, exit=" + str(rc))
        log.info("安装完成")
        return {"ok": True}

    def install_deps(self, install_all=False, packages=None):
        reqs = parse_requirements(ROOT / "requirements.txt")
        miss = missing_packages(reqs)
        if not miss and not install_all:
            messagebox.showinfo("依赖检查", "未检测到缺失依赖")
            return

        if packages is None and not install_all:
            packages = miss

        def job():
            return self._pip_install(install_all=install_all, packages=packages)

        def done(res):
            ok = isinstance(res, dict) and res.get("ok")
            if ok:
                messagebox.showinfo("安装完成", "依赖安装完成，请重新运行对应功能")
            else:
                messagebox.showerror("安装失败", str((res or {}).get("error") or "安装失败"))

        self.run_task("pip_install", job, done)

    def run_task(self, name, func, callback):
        self._active_tasks += 1
        if self._active_tasks == 1:
            try:
                self.progress.configure(mode="indeterminate", value=0)
                self.progress.start(12)
            except Exception:
                pass

        def worker():
            try:
                res = func()
            except Exception as e:
                res = {"ok": False, "error": str(e)}
            self.event_queue.put({"type": "task_done", "name": name, "result": res, "callback": callback})

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _task_finished(self):
        self._active_tasks = max(0, int(self._active_tasks) - 1)
        if self._active_tasks == 0:
            try:
                self.progress.stop()
                self.progress.configure(mode="indeterminate", value=0)
            except Exception:
                pass

    def _append_log(self, msg):
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(msg) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll_events(self):
        try:
            while True:
                ev = self.event_queue.get_nowait()
                if not isinstance(ev, dict):
                    continue
                t = ev.get("type")
                if t == "log":
                    self._append_log(ev.get("message") or "")
                    continue
                if t == "progress":
                    self._handle_progress(ev)
                    continue
                if t == "task_done":
                    cb = ev.get("callback")
                    res = ev.get("result")
                    try:
                        if cb is not None:
                            cb(res)
                    except Exception as e:
                        self._append_log("callback error: " + str(e))
                    finally:
                        self._task_finished()
        except queue.Empty:
            pass
        self.after(120, self._poll_events)

    def make_progress_cb(self, name):
        def cb(phase, current, total):
            self.event_queue.put({"type": "progress", "name": str(name), "phase": str(phase), "current": int(current), "total": int(total)})
        return cb

    def _handle_progress(self, ev):
        try:
            total = int(ev.get("total") or 0)
            cur = int(ev.get("current") or 0)
        except Exception:
            return
        if total <= 0:
            return
        try:
            if str(self.progress.cget("mode")) != "determinate":
                try:
                    self.progress.stop()
                except Exception:
                    pass
                self.progress.configure(mode="determinate", maximum=total, value=0)
            self.progress.configure(maximum=total)
            self.progress["value"] = min(max(cur, 0), total)
        except Exception:
            pass


if __name__ == "__main__":
    app = None
    try:
        app = App()
        app.mainloop()
    except KeyboardInterrupt:
        try:
            if app is not None:
                app.destroy()
        except Exception:
            pass
