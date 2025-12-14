import logging
from pathlib import Path

from src import io_utils, text_mining


def main():
    root = Path(__file__).resolve().parent
    dirs = io_utils.ensure_output_dirs(root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    params = {
        "source_mode": "自动抓取",
        "keyword": "600519",
        "pages": 1,
        "max_items": 20,
        "start_time": None,
        "end_time": None,
        "use_cache": True,
        "sleep_seconds": 1.0,
        "headers_text": "workers=8\nfetch_detail=1\nmax_detail=10\n",
        "top_n": 10,
        "n_topics": 4,
        "agg": "日",
    }
    res = text_mining.run_text_mining(params=params, uploaded_file=None, dirs=dirs)
    if not isinstance(res, dict) or not res.get("ok"):
        logging.getLogger("self_check").error("失败: " + str((res or {}).get("error")))
        return 1
    raw = res.get("raw")
    if raw is None or getattr(raw, "empty", True):
        logging.getLogger("self_check").warning("抓取结果为空")
        return 0
    logging.getLogger("self_check").info("抓取行数: " + str(int(raw.shape[0])))
    titles = []
    if "title" in raw.columns:
        titles = [str(x) for x in raw["title"].head(10).tolist()]
    for i, t in enumerate(titles, 1):
        logging.getLogger("self_check").info(str(i) + ". " + t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
