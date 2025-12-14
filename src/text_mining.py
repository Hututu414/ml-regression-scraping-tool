import json
import logging
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from src import io_utils


def _logger():
    return logging.getLogger("text.fetch")


def _progress(params, phase, current, total):
    cb = None
    if isinstance(params, dict):
        cb = params.get("_progress")
    if callable(cb):
        try:
            cb(str(phase), int(current), int(total))
        except Exception:
            pass


class RateLimiter:
    def __init__(self, interval_seconds, jitter_ratio=0.25):
        self.interval = max(0.0, float(interval_seconds or 0.0))
        self.jitter_ratio = max(0.0, float(jitter_ratio or 0.0))
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        now = time.time()
        with self._lock:
            target = self._next_time if self._next_time > now else now
            self._next_time = target + self.interval
        delay = target - now
        if delay > 0:
            time.sleep(delay)
        if self.jitter_ratio > 0:
            time.sleep(random.random() * self.jitter_ratio * self.interval)


def _build_session(headers):
    try:
        import requests
        from requests.adapters import HTTPAdapter
    except Exception as e:
        raise RuntimeError("缺少 requests: " + str(e))
    sess = requests.Session()
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    h = dict(headers or {})
    if "User-Agent" not in h:
        h["User-Agent"] = io_utils.build_user_agent()
    sess.headers.update(h)
    return sess


def _session_get(sess, url, timeout=(6, 18), retries=3, backoff=1.7, limiter=None):
    last = None
    for i in range(int(retries)):
        if limiter is not None:
            limiter.wait()
        try:
            resp = sess.get(url, timeout=timeout)
            code = int(getattr(resp, "status_code", 0) or 0)
            if code in [429, 500, 502, 503, 504]:
                last = RuntimeError("HTTP " + str(code))
            elif code >= 400:
                return resp
            else:
                return resp
        except Exception as e:
            last = e
        time.sleep((float(backoff) ** i) + random.random() * 0.2)
    raise last


def _extract_code_from_input(keyword):
    s = str(keyword or "").strip()
    if not s:
        return ""
    m = re.search(r"list,(\d{6})", s)
    if m:
        return m.group(1)
    m = re.search(r"news,(\d{6})", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d{6}", s):
        return s
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)
    return ""


def _seen_cache_file(dirs):
    return Path(dirs["cache"]) / "forum_seen_urls.json"


def _load_seen_urls(dirs):
    p = _seen_cache_file(dirs)
    try:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return set(str(x) for x in obj if str(x))
            if isinstance(obj, dict):
                return set(str(x) for x in obj.get("seen", []) if str(x))
    except Exception:
        return set()
    return set()


def _save_seen_urls(dirs, seen):
    p = _seen_cache_file(dirs)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        data = sorted(set(str(x) for x in (seen or set()) if str(x)))
        p.write_text(json.dumps({"seen": data}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _post_cache_dir(dirs):
    p = Path(dirs["cache"]) / "forum_posts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _post_cache_path(dirs, url):
    h = io_utils.stable_hash({"url": str(url)})
    return _post_cache_dir(dirs) / ("post_" + h + ".json")


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
        "不是",
        "什么",
        "怎么",
        "这样",
        "那样",
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
        if re.search(r"\d", p) and not re.search(r"[A-Za-z\u4e00-\u9fff]", p):
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


def _plot_wordcloud(freq_items):
    Figure, err = _mpl_available()
    if Figure is None:
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
    fig = Figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.imshow(img, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _plot_freq(freq_items):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    if not freq_items:
        return None
    top = freq_items[:30]
    words = [it["word"] for it in top][::-1]
    counts = [it["count"] for it in top][::-1]
    fig = Figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.barh(words, counts, color="steelblue")
    ax.set_title("词频 Top")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def _plot_trend(trend_items):
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    if not trend_items:
        return None
    df = pd.DataFrame(trend_items)
    if df.empty:
        return None
    fig = Figure(figsize=(9, 4))
    ax1 = fig.add_subplot(111)
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
    Figure, err = _mpl_available()
    if Figure is None:
        return None
    if not topic_items:
        return None
    df = pd.DataFrame(topic_items).sort_values("count", ascending=True)
    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.barh(df["topic"].astype(str).values, df["count"].values, color="purple")
    ax.set_title("主题分布")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def _parse_uploaded_html(html_file):
    b = io_utils.read_uploaded_bytes(html_file)
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
    url = _pick_field(it, ["url", "post_url", "href", "share_url", "TopicUrl"])
    author = _pick_field(it, ["author", "user", "nick", "user_nickname", "nickname"])
    read_count = _pick_field(it, ["read", "read_count", "readCount", "click", "click_count"])
    comment_count = _pick_field(it, ["reply", "reply_count", "comment", "comment_count", "replyCount", "commentCount"])
    post_id = _pick_field(it, ["post_id", "id", "TopicId", "postId"])
    ts = _normalize_time_string(t) if t is not None else None
    abs_url = _build_absolute_url(url)
    return {
        "keyword": str(keyword),
        "title": "" if title is None else str(title),
        "content": "" if content is None else str(content),
        "author": "" if author is None else str(author),
        "time": "" if ts is None else str(ts),
        "url": abs_url,
        "read_count": "" if read_count is None else str(read_count),
        "comment_count": "" if comment_count is None else str(comment_count),
        "post_id": "" if post_id is None else str(post_id),
        "source": "eastmoney_json",
    }


def _json_endpoint_urls(keyword, page, page_size):
    k_raw = str(keyword)
    k_code = _extract_code_from_input(k_raw)
    base = "https:" + "/" + "/" + "gbapi.eastmoney.com" + "/" + "webarticlelist" + "/" + "api" + "/" + "Article" + "/"
    urls = []
    if k_code:
        urls.append(base + "ArticleList?code=" + k_code + "&pageIndex=" + str(page) + "&pageSize=" + str(page_size) + "&sort=1")
    urls.append(base + "Search?keyword=" + k_raw + "&pageIndex=" + str(page) + "&pageSize=" + str(page_size) + "&sort=1")
    return urls


def _fetch_posts_json(keyword, pages, max_items, sleep_seconds, headers, dirs, params):
    log = _logger()
    posts = []
    limiter = RateLimiter(sleep_seconds, jitter_ratio=0.2)
    sess = _build_session(headers)
    total_pages = max(1, int(pages))
    for p in range(1, total_pages + 1):
        _progress(params, "list_pages", p - 1, total_pages)
        urls = _json_endpoint_urls(keyword, p, min(200, int(max_items)))
        got = False
        for url in urls:
            try:
                resp = _session_get(sess, url, timeout=(6, 18), retries=3, backoff=1.6, limiter=limiter)
                if int(getattr(resp, "status_code", 0) or 0) >= 400:
                    continue
                obj = resp.json()
                raw_items = _extract_json_posts(obj)
                mapped = []
                for it in raw_items:
                    m = _map_json_item(it, keyword)
                    if m is not None and (m.get("title") or m.get("content")):
                        mapped.append(m)
                if mapped:
                    posts.extend(mapped)
                    got = True
                    break
            except Exception as e:
                log.warning("JSON 列表失败: page=" + str(p) + " " + str(e))
                continue
        _progress(params, "list_pages", p, total_pages)
        if not got:
            continue
        if len(posts) >= int(max_items):
            break
    df = pd.DataFrame(posts)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["url", "title"]).reset_index(drop=True)
    return df.head(int(max_items)).reset_index(drop=True)


def _eastmoney_candidates(keyword, page):
    k = _extract_code_from_input(keyword) or str(keyword)
    base = "https:" + "/" + "/" + "guba.eastmoney.com" + "/"
    if re.fullmatch(r"\d{6}", k):
        return [
            base + "list," + k + ",f_" + str(page) + ".html",
            base + "list," + k + "_" + str(page) + ".html",
            base + "list," + k + ".html",
        ]
    return [
        base + "search?keyword=" + k + "&page=" + str(page),
        base + "search?keyword=" + k,
    ]


def _extract_post_links_from_html(html, keyword):
    BeautifulSoup, err = _bs4_available()
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html, "html.parser")
    code = _extract_code_from_input(keyword)
    links = []
    pat = re.compile(r"(?:/news,|https?://guba\.eastmoney\.com/news,)(\d{6}),(\d+)\.html", flags=re.IGNORECASE)
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        m = pat.search(str(href))
        if not m:
            continue
        c = m.group(1)
        if code and c != code:
            continue
        title = a.get_text(" ", strip=True) or ""
        abs_url = _build_absolute_url(href)
        links.append({"title": title, "url": abs_url, "post_id": m.group(2), "code": c})
    out = []
    seen = set()
    for it in links:
        u = str(it.get("url") or "")
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(it)
    return out


def _fetch_list_pages(keyword, pages, sleep_seconds, headers, dirs, params):
    BeautifulSoup, err = _bs4_available()
    if BeautifulSoup is None:
        raise RuntimeError("缺少 bs4: " + str(err))
    log = _logger()
    rows = []
    limiter = RateLimiter(sleep_seconds, jitter_ratio=0.2)
    sess = _build_session(headers)
    total_pages = max(1, int(pages))
    for p in range(1, total_pages + 1):
        _progress(params, "list_pages", p - 1, total_pages)
        urls = _eastmoney_candidates(keyword, p)
        got = False
        for url in urls:
            try:
                resp = _session_get(sess, url, timeout=(6, 18), retries=3, backoff=1.6, limiter=limiter)
                html = resp.text
                if not html or len(html) < 500:
                    continue
                rows.extend(_extract_post_links_from_html(html, keyword))
                got = True
                break
            except Exception as e:
                log.warning("列表页失败: " + str(url) + " " + str(e))
                continue
        _progress(params, "list_pages", p, total_pages)
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


def _fetch_detail(url, headers, dirs, sess, limiter, use_cache):
    BeautifulSoup, err = _bs4_available()
    if BeautifulSoup is None:
        return {"content": "", "time": None, "author": ""}
    cache_p = _post_cache_path(dirs, url)
    if bool(use_cache) and cache_p.exists():
        try:
            obj = json.loads(cache_p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    try:
        resp = _session_get(sess, url, timeout=(6, 18), retries=3, backoff=1.6, limiter=limiter)
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
    out = {"content": content, "time": t, "author": author}
    if bool(use_cache):
        try:
            cache_p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return out


def _int_from_headers(headers, key, default):
    if not isinstance(headers, dict):
        return int(default)
    v = headers.get(key)
    if v is None:
        v = headers.get(str(key))
    if v is None:
        return int(default)
    try:
        return int(float(str(v).strip()))
    except Exception:
        return int(default)


def _bool_from_headers(headers, key, default):
    if not isinstance(headers, dict):
        return bool(default)
    v = headers.get(key)
    if v is None:
        v = headers.get(str(key))
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ["1", "true", "yes", "y", "on"]:
        return True
    if s in ["0", "false", "no", "n", "off"]:
        return False
    return bool(default)


def _pop_cfg(headers, key):
    if not isinstance(headers, dict):
        return None
    for k in [key, str(key), str(key).lower(), str(key).upper()]:
        if k in headers:
            return headers.pop(k)
    return None


def _fetch_posts_auto(keyword, pages, max_items, sleep_seconds, headers, use_cache, dirs, start_time, end_time, params):
    payload = {
        "keyword": keyword,
        "pages": int(pages),
        "max_items": int(max_items),
        "sleep_seconds": float(sleep_seconds),
        "start_time": str(start_time or ""),
        "end_time": str(end_time or ""),
    }
    cache_file = io_utils.cache_path(dirs, "text_raw", payload, ext=".csv")
    if bool(use_cache) and cache_file.exists():
        return pd.read_csv(cache_file, encoding="utf-8-sig")

    headers = dict(headers or {})
    workers = _int_from_headers(headers, "workers", 8)
    fetch_detail = _bool_from_headers(headers, "fetch_detail", True)
    detail_min_len = _int_from_headers(headers, "detail_min_len", 40)
    max_detail = _int_from_headers(headers, "max_detail", 2000)
    for k in ["workers", "fetch_detail", "detail_min_len", "max_detail"]:
        _pop_cfg(headers, k)

    log = _logger()
    start_ts = _normalize_time_string(start_time) if start_time else None
    end_ts = _normalize_time_string(end_time) if end_time else None

    seen = _load_seen_urls(dirs) if bool(use_cache) else set()

    json_df = _fetch_posts_json(keyword, pages, max_items, sleep_seconds, headers, dirs=dirs, params=params)
    if json_df is not None and not json_df.empty:
        df = json_df.copy()
    else:
        items = _fetch_list_pages(keyword, pages, sleep_seconds, headers, dirs=dirs, params=params)
        items = items[: int(max_items)]
        df = pd.DataFrame(items)
        if not df.empty and "url" in df.columns:
            df["url"] = df["url"].map(_build_absolute_url)
            df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    if df is None or df.empty:
        out = pd.DataFrame(columns=["keyword", "title", "content", "author", "time", "url", "source"])
        out.to_csv(cache_file, index=False, encoding="utf-8-sig")
        return out

    if "url" in df.columns:
        df["url"] = df["url"].map(_build_absolute_url)
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    log.info("列表结果条数: " + str(int(len(df))))

    if "title" not in df.columns:
        df["title"] = ""
    if "content" not in df.columns:
        df["content"] = ""
    if "author" not in df.columns:
        df["author"] = ""
    if "time" not in df.columns:
        df["time"] = ""
    if "keyword" not in df.columns:
        df["keyword"] = str(keyword)
    if "source" not in df.columns:
        df["source"] = "eastmoney"

    seen_hit = 0
    if bool(use_cache) and "url" in df.columns and seen:
        try:
            seen_hit = int(df["url"].astype(str).isin(seen).sum())
        except Exception:
            seen_hit = 0
    if bool(use_cache) and seen_hit > 0:
        log.info("已抓取 URL 命中: " + str(seen_hit))

    if bool(fetch_detail) and "url" in df.columns and len(df) > 0:
        tls = threading.local()
        need_idx = []
        for i, row in df.iterrows():
            try:
                c = str(row.get("content") or "")
                if len(c.strip()) < int(detail_min_len):
                    need_idx.append(int(i))
            except Exception:
                need_idx.append(int(i))
        if max_detail > 0:
            need_idx = need_idx[: int(max_detail)]
        total_need = int(len(need_idx))
        if total_need > 0:
            log.info("详情页并发抓取: " + str(total_need) + " 条, workers=" + str(int(workers)))
            _progress(params, "detail_pages", 0, total_need)

            def fetch_one(i):
                u = str(df.at[i, "url"])
                sess = getattr(tls, "sess", None)
                if sess is None:
                    sess = _build_session(headers)
                    tls.sess = sess
                limiter = getattr(tls, "limiter", None)
                if limiter is None:
                    limiter = RateLimiter(sleep_seconds, jitter_ratio=0.2)
                    tls.limiter = limiter
                detail = _fetch_detail(u, headers=headers, dirs=dirs, sess=sess, limiter=limiter, use_cache=use_cache)
                return i, detail

            done_n = 0
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
                futures = [ex.submit(fetch_one, i) for i in need_idx]
                for fut in as_completed(futures):
                    try:
                        i, detail = fut.result()
                        if isinstance(detail, dict):
                            if detail.get("content"):
                                df.at[i, "content"] = str(detail.get("content") or "")
                            if detail.get("author"):
                                df.at[i, "author"] = str(detail.get("author") or "")
                            if detail.get("time") is not None:
                                df.at[i, "time"] = "" if detail.get("time") is None else str(detail.get("time"))
                    except Exception as e:
                        log.warning("详情抓取失败: " + str(e))
                    done_n += 1
                    _progress(params, "detail_pages", done_n, total_need)

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

    if bool(use_cache) and "url" in df.columns:
        for u in df["url"].astype(str).tolist():
            if u:
                seen.add(u)
        _save_seen_urls(dirs, seen)

    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return df


def _load_uploaded_csv(csv_file):
    b = io_utils.read_uploaded_bytes(csv_file)
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
    if "time" not in df.columns:
        df["time"] = ""
    df["text"] = (df["title"].astype(str) + " " + df["content"].astype(str)).map(_clean_text)
    df["tokens"] = df["text"].map(_tokenize)
    df["sentiment"] = df["text"].map(_sentiment_score)
    df["time_parsed"] = df["time"].map(_normalize_time_string)
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
            try:
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
                    params=params,
                )
            except Exception as e:
                _logger().warning("抓取失败，返回空结果: " + str(e))
                raw_df = pd.DataFrame(columns=["keyword", "title", "content", "author", "time", "url", "source"])
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
    raw_df = result.get("raw")
    if not isinstance(raw_df, pd.DataFrame):
        raw_df = pd.DataFrame()
    clean_df = result.get("clean")
    if not isinstance(clean_df, pd.DataFrame):
        clean_df = pd.DataFrame()
    analytics = result.get("analytics") or {}

    out = []
    raw_path = Path(dirs["forum_db"]) / ("text_raw_" + suffix + ".csv")
    clean_path = Path(dirs["forum_db"]) / ("text_clean_" + suffix + ".csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    out.extend([raw_path, clean_path])

    report_path = Path(dirs["tables"]) / ("text_analysis_" + suffix + ".xlsx")
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
