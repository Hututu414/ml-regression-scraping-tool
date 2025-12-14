import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from modules import io_utils


def _mpl_available():
    try:
        import matplotlib.pyplot as plt

        return plt, None
    except Exception as e:
        return None, str(e)


def _bs4_available():
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup, None
    except Exception as e:
        return None, str(e)


def _normalize_time_string(s):
    text = str(s or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{2}-\d{2}\s+\d{2}:\d{2}", text):
        year = pd.Timestamp.now().year
        text = str(year) + "-" + text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        text = text + " 00:00"
    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _clean_text(s):
    text = str(s or "")
    text = re.sub(r"https?:\s*[/][/]\\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"www\\.[a-zA-Z0-9\\-\\.]+", " ", text)
    text = re.sub(r"@[\\w\\-]{1,40}", " ", text)
    text = re.sub(r"[\\r\\n\\t]+", " ", text)
    text = re.sub(r"[^\\u4e00-\\u9fff0-9a-zA-Z\\s]+", " ", text)
    text = re.sub(r"\\s{2,}", " ", text).strip()
    return text


def _tokenize(text):
    t = _clean_text(text)
    if not t:
        return []
    try:
        import jieba

        parts = [p.strip() for p in jieba.lcut(t) if p.strip()]
    except Exception:
        parts = re.findall(r"[\\u4e00-\\u9fff]{2,}|[a-zA-Z0-9]{2,}", t)
    stop = {
        "的",
        "了",
        "是",
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "她们",
        "它们",
        "这个",
        "那个",
        "一个",
        "不会",
        "没有",
        "就是",
        "还是",
        "但是",
        "因为",
        "所以",
        "以及",
        "如果",
        "然后",
        "可以",
        "今天",
        "明天",
        "昨天",
        "现在",
        "一起",
        "大家",
        "自己",
        "可能",
        "已经",
        "一下",
        "不会",
        "不是",
        "什么",
        "怎么",
        "这样",
        "那样",
        "一个",
        "一些",
        "一样",
        "一样的",
        "都",
        "很",
        "更",
        "最",
    }
    out = []
    for p in parts:
        if len(p) < 2:
            continue
        if p in stop:
            continue
        out.append(p)
    return out


def _word_freq(docs_tokens, top_n):
    c = Counter()
    for tokens in docs_tokens:
        c.update(tokens)
    items = c.most_common(int(top_n))
    return [{"word": w, "count": int(n)} for w, n in items]


def _tfidf_keywords(docs_tokens, top_n):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return []
    docs = [" ".join(toks) for toks in docs_tokens]
    if not any(docs):
        return []
    vectorizer = TfidfVectorizer(tokenizer=str.split, lowercase=False)
    X = vectorizer.fit_transform(docs)
    scores = np.asarray(X.mean(axis=0)).reshape(-1)
    terms = vectorizer.get_feature_names_out()
    idx = np.argsort(scores)[::-1][: int(top_n)]
    out = []
    for i in idx:
        out.append({"word": str(terms[i]), "score": float(scores[i])})
    return out


def _sentiment_score(text):
    t = _clean_text(text)
    if not t:
        return float("nan")
    try:
        from snownlp import SnowNLP

        return float(SnowNLP(t).sentiments)
    except Exception:
        pos = ["利好", "上涨", "看好", "突破", "反弹", "增持", "强势", "赚钱", "牛", "涨停", "超预期"]
        neg = ["利空", "下跌", "看空", "破位", "回调", "减持", "弱势", "亏", "熊", "跌停", "暴雷"]
        p = sum(1 for w in pos if w in t)
        n = sum(1 for w in neg if w in t)
        raw = (p - n) / (p + n + 1e-9)
        return float(0.5 + 0.5 * math.tanh(raw))


def _topic_model(docs_tokens, n_topics):
    texts = [" ".join(t) for t in docs_tokens]
    texts = [t for t in texts if t.strip()]
    if len(texts) < 5:
        return {"topics": [], "doc_topics": []}
    try:
        from bertopic import BERTopic

        model = BERTopic(language="chinese")
        topics, probs = model.fit_transform(texts)
        summary = []
        info = model.get_topic_info()
        for _, row in info.iterrows():
            topic_id = int(row.get("Topic"))
            if topic_id == -1:
                continue
            words = model.get_topic(topic_id) or []
            top_words = [w for w, _ in words[:10]]
            summary.append({"topic": topic_id, "count": int(row.get("Count")), "words": " ".join(top_words)})
        doc_topics = [{"topic": int(t)} for t in topics]
        return {"topics": summary, "doc_topics": doc_topics}
    except Exception:
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.feature_extraction.text import CountVectorizer
        except Exception:
            return {"topics": [], "doc_topics": []}
        vectorizer = CountVectorizer(tokenizer=str.split, lowercase=False, min_df=2, max_df=0.95)
        X = vectorizer.fit_transform(texts)
        if X.shape[1] < 5:
            return {"topics": [], "doc_topics": []}
        lda = LatentDirichletAllocation(n_components=int(n_topics), random_state=0, learning_method="batch", max_iter=30)
        doc_topic = lda.fit_transform(X)
        topics = doc_topic.argmax(axis=1)
        terms = vectorizer.get_feature_names_out()
        summary = []
        for k in range(int(n_topics)):
            comp = lda.components_[k]
            idx = np.argsort(comp)[::-1][:10]
            words = [str(terms[i]) for i in idx]
            summary.append({"topic": int(k), "count": int(np.sum(topics == k)), "words": " ".join(words)})
        doc_topics = [{"topic": int(t)} for t in topics]
        return {"topics": summary, "doc_topics": doc_topics}


def _trend(df, time_col, score_col, agg):
    if time_col not in df.columns:
        return []
    x = df.copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col])
    if x.empty:
        return []
    rule = "W-FRI" if str(agg) == "周" else "D"
    g = x.set_index(time_col).resample(rule)
    out = pd.DataFrame(
        {
            "count": g.size(),
            "avg_sentiment": g[score_col].mean() if score_col in x.columns else np.nan,
        }
    ).reset_index()
    out = out.rename(columns={time_col: "time"})
    out["time"] = out["time"].astype(str)
    return out.to_dict(orient="records")


def _plot_wordcloud(freq_items):
    plt, err = _mpl_available()
    if plt is None:
        return None
    try:
        from wordcloud import WordCloud
    except Exception:
        return None
    font_path = _find_cn_font()
    words = {it["word"]: float(it["count"]) for it in freq_items}
    if not words:
        return None
    wc = WordCloud(width=900, height=450, background_color="white", collocations=False, font_path=font_path)
    img = wc.generate_from_frequencies(words)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(img, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _plot_freq(freq_items):
    plt, err = _mpl_available()
    if plt is None:
        return None
    if not freq_items:
        return None
    top = freq_items[:30]
    words = [it["word"] for it in top][::-1]
    counts = [it["count"] for it in top][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(words, counts, color="steelblue")
    ax.set_title("词频 Top")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def _plot_trend(trend_items):
    plt, err = _mpl_available()
    if plt is None:
        return None
    if not trend_items:
        return None
    df = pd.DataFrame(trend_items)
    if df.empty:
        return None
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(df["time"].values, df["count"].values, color="blue", linewidth=1.4)
    ax1.set_title("发帖量趋势")
    ax1.set_ylabel("count")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.2)
    if "avg_sentiment" in df.columns and df["avg_sentiment"].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(df["time"].values, df["avg_sentiment"].values, color="orange", linewidth=1.2)
        ax2.set_ylabel("avg sentiment")
    fig.tight_layout()
    return fig


def _plot_topics(topic_items):
    plt, err = _mpl_available()
    if plt is None:
        return None
    if not topic_items:
        return None
    df = pd.DataFrame(topic_items).sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["topic"].astype(str).values, df["count"].values, color="purple")
    ax.set_title("主题分布")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def _parse_uploaded_html(uploaded_file):
    b = io_utils.read_uploaded_bytes(uploaded_file)
    if b is None:
        return []
    try:
        text = b.decode("utf-8", errors="ignore")
    except Exception:
        text = str(b)
    BeautifulSoup, err = _bs4_available()
    if BeautifulSoup is None:
        return []
    try:
        soup = BeautifulSoup(text, "lxml")
    except Exception:
        soup = BeautifulSoup(text, "html.parser")
    links = soup.find_all("a")
    items = []
    for a in links:
        title = a.get_text(strip=True)
        href = a.get("href")
        if not title or not href:
            continue
        if len(title) < 2:
            continue
        items.append({"title": title, "url": href})
    return items


def _find_cn_font():
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arialuni.ttf"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return str(p)
        except Exception:
            continue
    return None


def _extract_json_posts(obj, depth=4):
    if depth <= 0:
        return []
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj[:5]):
            return obj
        out = []
        for x in obj[:50]:
            out.extend(_extract_json_posts(x, depth=depth - 1))
        return out
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_extract_json_posts(v, depth=depth - 1))
        return out
    return []


def _pick_field(d, keys):
    for k in keys:
        if k in d and d.get(k) not in [None, ""]:
            return d.get(k)
    for k, v in d.items():
        lk = str(k).lower()
        for want in keys:
            if str(want).lower() in lk and v not in [None, ""]:
                return v
    return None


def _map_json_item(it, keyword):
    if not isinstance(it, dict):
        return None
    title = _pick_field(it, ["title", "post_title", "subject", "TopicTitle"])
    content = _pick_field(it, ["content", "post_content", "body", "summary", "abstract"])
    t = _pick_field(it, ["time", "post_time", "publish_time", "create_time", "date"])
    url = _pick_field(it, ["url", "post_url", "href", "share_url"])
    author = _pick_field(it, ["author", "user", "nick", "user_nickname", "nickname"])
    ts = _normalize_time_string(t) if t is not None else None
    return {
        "keyword": str(keyword),
        "title": "" if title is None else str(title),
        "content": "" if content is None else str(content),
        "author": "" if author is None else str(author),
        "time": "" if ts is None else str(ts),
        "url": "" if url is None else str(url),
        "source": "eastmoney_json",
    }


def _json_endpoint_urls(keyword, page, page_size):
    k = str(keyword)
    base = "https:" + "/" + "/" + "gbapi.eastmoney.com" + "/" + "webarticlelist" + "/" + "api" + "/" + "Article" + "/"
    urls = []
    if re.fullmatch(r"\\d{6}", k):
        urls.append(base + "ArticleList?code=" + k + "&pageIndex=" + str(page) + "&pageSize=" + str(page_size) + "&sort=1")
    urls.append(base + "Search?keyword=" + k + "&pageIndex=" + str(page) + "&pageSize=" + str(page_size) + "&sort=1")
    return urls


def _fetch_posts_json(keyword, pages, max_items, sleep_seconds, headers):
    posts = []
    for p in range(1, int(pages) + 1):
        for url in _json_endpoint_urls(keyword, p, min(200, int(max_items))):
            try:
                resp = io_utils.request_get(url, headers=headers, timeout=12, retries=2, backoff=1.6, session=None)
                obj = resp.json()
                raw_items = _extract_json_posts(obj)
                mapped = []
                for it in raw_items:
                    m = _map_json_item(it, keyword)
                    if m is not None and (m.get("title") or m.get("content")):
                        mapped.append(m)
                if mapped:
                    posts.extend(mapped)
                    break
            except Exception:
                continue
        time.sleep(float(sleep_seconds))
        if len(posts) >= int(max_items):
            break
    df = pd.DataFrame(posts)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["url", "title"]).reset_index(drop=True)
    return df.head(int(max_items)).reset_index(drop=True)


def _eastmoney_candidates(keyword, page):
    k = str(keyword)
    base = "https:" + "/" + "/" + "guba.eastmoney.com" + "/"
    if re.fullmatch(r"\\d{6}", k):
        return [
            base + "list," + k + ",f_" + str(page) + ".html",
            base + "list," + k + "_" + str(page) + ".html",
            base + "list," + k + ".html",
        ]
    return [
        base + "search?keyword=" + k + "&page=" + str(page),
        base + "search?keyword=" + k,
    ]


def _fetch_list_pages(keyword, pages, sleep_seconds, headers):
    BeautifulSoup, err = _bs4_available()
    if BeautifulSoup is None:
        raise RuntimeError("缺少 bs4: " + str(err))
    rows = []
    sess = None
    for p in range(1, int(pages) + 1):
        urls = _eastmoney_candidates(keyword, p)
        got = False
        for url in urls:
            try:
                resp = io_utils.request_get(url, headers=headers, timeout=12, retries=2, backoff=1.6, session=sess)
                html = resp.text
                if not html or len(html) < 500:
                    continue
                soup = BeautifulSoup(html, "html.parser")
                anchors = soup.find_all("a")
                for a in anchors:
                    title = a.get_text(strip=True)
                    href = a.get("href")
                    if not title or not href:
                        continue
                    if len(title) < 3:
                        continue
                    if "javascript" in href:
                        continue
                    rows.append({"title": title, "url": href})
                got = True
                break
            except Exception:
                continue
        time.sleep(float(sleep_seconds))
        if not got:
            continue
    seen = set()
    out = []
    for r in rows:
        u = str(r.get("url") or "")
        t = str(r.get("title") or "")
        key = u if u else t
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _build_absolute_url(u):
    url = str(u or "").strip()
    if not url:
        return ""
    if url.startswith("http"):
        return url
    base = "https:" + "/" + "/" + "guba.eastmoney.com"
    if url.startswith("/"):
        return base + url
    return base + "/" + url


def _fetch_detail(url, headers):
    BeautifulSoup, err = _bs4_available()
    if BeautifulSoup is None:
        return {"content": "", "time": None, "author": ""}
    try:
        resp = io_utils.request_get(url, headers=headers, timeout=12, retries=2, backoff=1.6, session=None)
        html = resp.text
    except Exception:
        return {"content": "", "time": None, "author": ""}
    soup = BeautifulSoup(html, "html.parser")
    content = ""
    candidates = [
        {"name": "div", "attrs": {"class": "newstext"}},
        {"name": "div", "attrs": {"class": "xeditor"}},
        {"name": "div", "attrs": {"id": "post_content"}},
    ]
    for c in candidates:
        el = soup.find(c["name"], attrs=c["attrs"])
        if el is not None:
            content = el.get_text(" ", strip=True)
            if content:
                break
    author = ""
    a = soup.find("a", attrs={"class": "user-name"})
    if a is not None:
        author = a.get_text(strip=True)
    t = None
    time_el = soup.find("div", attrs={"class": "zwfbtime"})
    if time_el is not None:
        t = _normalize_time_string(time_el.get_text(" ", strip=True))
    return {"content": content, "time": t, "author": author}


def _fetch_posts_auto(keyword, pages, max_items, sleep_seconds, headers, use_cache, dirs, start_time, end_time):
    payload = {
        "keyword": keyword,
        "pages": int(pages),
        "max_items": int(max_items),
        "sleep_seconds": float(sleep_seconds),
        "start_time": str(start_time or ""),
        "end_time": str(end_time or ""),
    }
    cache_file = io_utils.cache_path(dirs, "text_raw", payload)
    if bool(use_cache) and cache_file.exists():
        df = pd.read_csv(cache_file, encoding="utf-8-sig")
        return df

    headers = dict(headers or {})
    start_ts = _normalize_time_string(start_time) if start_time else None
    end_ts = _normalize_time_string(end_time) if end_time else None
    json_df = _fetch_posts_json(keyword, pages, max_items, sleep_seconds, headers)
    if json_df is not None and not json_df.empty:
        df = json_df.copy()
    else:
        items = _fetch_list_pages(keyword, pages, sleep_seconds, headers)
        items = items[: int(max_items)]
        posts = []
        for it in items:
            abs_url = _build_absolute_url(it.get("url"))
            detail = _fetch_detail(abs_url, headers=headers)
            t = detail.get("time")
            if start_ts is not None and t is not None and t < start_ts:
                continue
            if end_ts is not None and t is not None and t > end_ts:
                continue
            posts.append(
                {
                    "keyword": keyword,
                    "title": str(it.get("title") or ""),
                    "content": str(detail.get("content") or ""),
                    "author": str(detail.get("author") or ""),
                    "time": "" if t is None else str(t),
                    "url": abs_url,
                    "source": "eastmoney",
                }
            )
            time.sleep(float(sleep_seconds))
            if len(posts) >= int(max_items):
                break
        df = pd.DataFrame(posts)
    if start_ts is not None or end_ts is not None:
        if "time" in df.columns:
            tp = df["time"].map(_normalize_time_string)
        else:
            tp = pd.Series([None] * int(len(df)), index=df.index)
        mask = pd.Series(True, index=df.index)
        if start_ts is not None:
            mask = mask & ((tp.isna()) | (tp >= start_ts))
        if end_ts is not None:
            mask = mask & ((tp.isna()) | (tp <= end_ts))
        df = df.loc[mask].reset_index(drop=True)
    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return df


def _load_uploaded_csv(uploaded_file):
    b = io_utils.read_uploaded_bytes(uploaded_file)
    df = io_utils.read_csv_bytes(b)
    cols = {str(c).strip().lower(): c for c in df.columns}
    def pick(names):
        for n in names:
            if n in cols:
                return cols[n]
        for k, v in cols.items():
            for n in names:
                if n in k:
                    return v
        return None
    title_col = pick(["title", "标题"])
    content_col = pick(["content", "text", "正文", "内容"])
    time_col = pick(["time", "datetime", "date", "时间", "日期"])
    url_col = pick(["url", "链接"])
    author_col = pick(["author", "user", "作者"])
    out = pd.DataFrame()
    out["title"] = df[title_col] if title_col else ""
    out["content"] = df[content_col] if content_col else ""
    out["time"] = df[time_col] if time_col else ""
    out["url"] = df[url_col] if url_col else ""
    out["author"] = df[author_col] if author_col else ""
    out["source"] = "upload_csv"
    return out


def _build_clean_df(raw_df):
    df = raw_df.copy()
    if "title" not in df.columns:
        df["title"] = ""
    if "content" not in df.columns:
        df["content"] = ""
    df["text"] = (df["title"].astype(str) + " " + df["content"].astype(str)).map(_clean_text)
    df["tokens"] = df["text"].map(_tokenize)
    df["sentiment"] = df["text"].map(_sentiment_score)
    df["time_parsed"] = df.get("time", "").map(_normalize_time_string)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df[df["text"].astype(str).str.len() >= 2].reset_index(drop=True)
    return df


def run_text_mining(params, uploaded_file, dirs):
    try:
        headers = io_utils.parse_kv_text(params.get("headers_text"))
        source_mode = params.get("source_mode")
        keyword = params.get("keyword")
        if source_mode == "CSV 上传":
            raw_df = _load_uploaded_csv(uploaded_file)
        elif source_mode == "HTML 上传":
            links = _parse_uploaded_html(uploaded_file)
            raw_df = pd.DataFrame(links)
            if "title" not in raw_df.columns:
                raw_df["title"] = ""
            raw_df["content"] = ""
            raw_df["time"] = ""
            raw_df["author"] = ""
            raw_df["source"] = "upload_html"
        else:
            raw_df = _fetch_posts_auto(
                keyword=keyword,
                pages=params.get("pages"),
                max_items=params.get("max_items"),
                sleep_seconds=max(1.0, float(params.get("sleep_seconds") or 1.0)),
                headers=headers,
                use_cache=params.get("use_cache"),
                dirs=dirs,
                start_time=params.get("start_time"),
                end_time=params.get("end_time"),
            )
        clean_df = _build_clean_df(raw_df)
        docs_tokens = clean_df["tokens"].tolist()
        freq_items = _word_freq(docs_tokens, params.get("top_n"))
        tfidf_items = _tfidf_keywords(docs_tokens, params.get("top_n"))
        topic_out = _topic_model(docs_tokens, params.get("n_topics"))
        if topic_out.get("doc_topics"):
            clean_df["topic"] = [d.get("topic") for d in topic_out["doc_topics"]]
        trend_items = _trend(clean_df, "time_parsed", "sentiment", params.get("agg"))

        figs = {}
        figs["wordcloud"] = _plot_wordcloud(freq_items)
        figs["freq"] = _plot_freq(freq_items)
        figs["trend"] = _plot_trend(trend_items)
        figs["topics"] = _plot_topics(topic_out.get("topics") or [])

        analytics = {
            "freq": freq_items,
            "tfidf": tfidf_items,
            "topics": topic_out.get("topics") or [],
            "trend": trend_items,
        }
        return {"ok": True, "raw": raw_df, "clean": clean_df, "analytics": analytics, "figures": figs, "params": params}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def export_text_mining(result, dirs, tag):
    if not isinstance(result, dict) or not result.get("ok"):
        return []
    params = result.get("params") or {}
    payload = {"kind": "text", "tag": str(tag or ""), "params": params}
    suffix = io_utils.safe_filename(tag) if str(tag or "").strip() else io_utils.stable_hash(payload)
    raw_df = result.get("raw") or pd.DataFrame()
    clean_df = result.get("clean") or pd.DataFrame()
    analytics = result.get("analytics") or {}

    out = []
    raw_path = Path(dirs["data"]) / ("text_raw_" + suffix + ".csv")
    clean_path = Path(dirs["data"]) / ("text_clean_" + suffix + ".csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    out.extend([raw_path, clean_path])

    report_path = Path(dirs["reports"]) / ("text_analysis_" + suffix + ".xlsx")
    sheets = {
        "raw": raw_df,
        "clean": clean_df,
        "freq": pd.DataFrame(analytics.get("freq") or []),
        "tfidf": pd.DataFrame(analytics.get("tfidf") or []),
        "topics": pd.DataFrame(analytics.get("topics") or []),
        "trend": pd.DataFrame(analytics.get("trend") or []),
        "params": pd.DataFrame([params]),
    }
    io_utils.save_excel(sheets, report_path)
    out.append(report_path)
    return out
