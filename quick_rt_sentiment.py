# quick_rt_sentiment.py
# Minimal real-time financial news sentiment ↔ market correlation + (optional) GenAI brief
# - NewsAPI: fetch headlines
# - VADER: headline sentiment ([-1,1])
# - yfinance: index returns (handles Adj Close / Close & 1/Many tickers)
# - Optional OpenAI: generate short brief

import os
from urllib.parse import urlparse, parse_qs, unquote
from typing import List, Dict, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------ Config ------------------
st.set_page_config(page_title="RT Financial Sentiment (Minimal)", layout="wide")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY", ""))
OPENAI_KEY  = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# 指数别名 → yfinance 代码
INDEX_TO_YF = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq Composite": "^IXIC",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225",
}
# 反向映射（用于单 ticker 情况）
YF_TO_INDEX = {v: k for k, v in INDEX_TO_YF.items()}

# 标题关键词 → 指数映射（够用即可）
KEY2INDEX = [
    (["s&p 500", "spx", "gspc"], "S&P 500"),
    (["dow jones", "djia"], "Dow Jones"),
    (["nasdaq", "ixic"], "Nasdaq Composite"),
    (["ftse"], "FTSE 100"),
    (["dax"], "DAX"),
    (["nikkei", "n225"], "Nikkei 225"),
]

# ------------------ Helpers ------------------
def to_day(ts) -> pd.Timestamp:
    """UTC → tz-naive → 00:00 当天"""
    return pd.to_datetime(ts, errors="coerce", utc=True).tz_convert(None).normalize()

def _domain(url: str) -> str:
    try:
        h = urlparse(url).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""

def _unwrap(url: str) -> str:
    """解包 Google News / MSN 聚合链接，拿真实原文 URL"""
    try:
        h = _domain(url)
        if h in {"news.google.com", "www.google.com", "google.com", "www.msn.com", "msn.com"}:
            qs = parse_qs(urlparse(url).query)
            for k in ("url", "u"):
                if k in qs and qs[k]:
                    return unquote(qs[k][0])
    except Exception:
        pass
    return url

def map_index(title: str) -> str:
    t = (title or "").lower()
    for kws, idx in KEY2INDEX:
        if any(k in t for k in kws):
            return idx
    return "S&P 500"  # 兜底一个主指数，确保相关性能算

# ------------------ 1) Fetch news ------------------
def fetch_news(mode: str = "top", query: str = "", from_hours: int = 12,
               page_size: int = 40, country: str = "us") -> Tuple[pd.DataFrame, Dict]:
    """
    mode = "top" -> top-headlines (country=us, category=business)
    mode = "everything" -> everything (时间窗)
    """
    if not NEWSAPI_KEY:
        return pd.DataFrame(), {"error": "Missing NEWSAPI_KEY"}

    if mode == "everything":
        url = "https://newsapi.org/v2/everything"
        frm = (pd.Timestamp.utcnow() - pd.Timedelta(hours=from_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "q": query or "stocks OR markets OR economy OR index",
            "from": frm, "sortBy": "publishedAt",
            "language": "en",
            "pageSize": min(page_size, 100),
            "apiKey": NEWSAPI_KEY,
        }
    else:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "category": "business",
            "country": country,
            "language": "en",
            "pageSize": min(page_size, 100),
            "apiKey": NEWSAPI_KEY,
        }
        if query:
            params["q"] = query

    r = requests.get(url, params=params, timeout=20)
    meta = {"http": r.status_code, "mode": mode}
    try:
        js = r.json()
    except Exception:
        return pd.DataFrame(), meta
    meta["api_status"] = js.get("status")
    meta["message"] = js.get("message")

    if js.get("status") != "ok":
        return pd.DataFrame(), meta

    arts = js.get("articles", []) or []
    rows = [{
        "date": a.get("publishedAt"),
        "headline": (a.get("title") or "").strip(),
        "news_url": a.get("url"),
        "source": (a.get("source") or {}).get("name"),
    } for a in arts]

    if not rows:
        return pd.DataFrame(), meta

    df = pd.DataFrame(rows)
    df["news_url"] = df["news_url"].astype(str)
    df["news_url_unwrapped"] = df["news_url"].map(_unwrap)
    df["date"] = df["date"].map(to_day)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=False).drop_duplicates(subset=["news_url_unwrapped"])
    df["market_index"] = df["headline"].map(map_index)
    return df, meta

# ------------------ 2) Sentiment (VADER) ------------------
_analyzer = SentimentIntensityAnalyzer()

def vader_label_score(text: str, pos=0.05, neg=-0.05) -> Tuple[str, float]:
    s = _analyzer.polarity_scores(text or "")
    c = s["compound"]
    if c >= pos:   return "Positive", c
    if c <= neg:   return "Negative", c
    return "Neutral", c

def score_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    labs, scs = [], []
    for t in df["headline"].astype(str).tolist():
        lab, c = vader_label_score(t)
        labs.append(lab); scs.append(c)
    out = df.copy()
    out["sent_label"] = labs
    out["sent_score"] = scs  # [-1,1]
    return out

# ------------------ 3) Daily aggregate ------------------
def daily_sentiment(df_scored: pd.DataFrame) -> pd.DataFrame:
    if df_scored.empty:
        return pd.DataFrame(columns=["date", "market_index", "sent_mean", "n"])
    d = df_scored.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    # 权重：置信度=abs(compound)
    d["w"] = d["sent_score"].abs().clip(0.2, 1.0)
    d["score_w"] = d["sent_score"] * d["w"]
    g = (d.groupby(["date", "market_index"])
           .agg(sent_mean=("score_w", "mean"), n=("headline", "size"))
           .reset_index()
           .sort_values(["market_index", "date"]))
    return g

# ------------------ 4) Market data & correlation ------------------
def fetch_index_returns(indices: List[str], days: int = 60) -> pd.DataFrame:
    """兼容 MultiIndex/单列；优先 Adj Close，缺失回退 Close。"""
    tickers = [INDEX_TO_YF[i] for i in indices if i in INDEX_TO_YF]
    if not tickers:
        return pd.DataFrame()

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days + 5)

    data = yf.download(
        tickers=tickers, start=start, end=end,
        progress=False, auto_adjust=False
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()

    rows = []

    if isinstance(data.columns, pd.MultiIndex):
        # 多 ticker：列是 (ticker, field)
        for yf_tic in tickers:
            if yf_tic not in data.columns.get_level_values(0):
                continue
            sub = data[yf_tic]
            for col in ("Adj Close", "Close", "adjclose", "close"):
                if col in sub.columns:
                    ser = sub[col].dropna()
                    if ser.empty:
                        break
                    tmp = ser.pct_change().reset_index().rename(columns={"Date": "date", col: "ret"})
                    tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
                    tmp["market_index"] = YF_TO_INDEX.get(yf_tic, yf_tic)
                    rows.append(tmp)
                    break
    else:
        # 单 ticker：列是单层
        price_col = None
        for col in ("Adj Close", "Close", "adjclose", "close"):
            if col in data.columns:
                price_col = col
                break
        if price_col is None:
            return pd.DataFrame()
        ser = data[price_col].dropna()
        tmp = ser.pct_change().reset_index().rename(columns={"Date": "date", price_col: "ret"})
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
        tmp["market_index"] = YF_TO_INDEX.get(tickers[0], tickers[0])
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def corr_same_and_t1(daily: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    if daily.empty or rets.empty:
        return pd.DataFrame()
    m0 = pd.merge(daily, rets, on=["date", "market_index"], how="inner")

    def _pearson(a, b):
        a = pd.Series(a).astype(float)
        b = pd.Series(b).astype(float)
        if a.notna().sum() < 5 or b.notna().sum() < 5:
            return np.nan
        return float(a.corr(b))

    rows = []
    for idx, g in m0.groupby("market_index"):
        r_t = _pearson(g["sent_mean"], g["ret"])
        rows.append({"index": idx, "corr_t": r_t})

    # T+1：让收益后移一位（明日收益）
    rets_t1 = rets.copy()
    rets_t1["ret_t1"] = rets_t1.groupby("market_index")["ret"].shift(-1)
    m1 = pd.merge(daily, rets_t1[["date", "market_index", "ret_t1"]],
                  on=["date", "market_index"], how="inner")
    for idx, g in m1.groupby("market_index"):
        r_t1 = _pearson(g["sent_mean"], g["ret_t1"])
        for row in rows:
            if row["index"] == idx:
                row["corr_t1"] = r_t1
                break
    return pd.DataFrame(rows).sort_values("corr_t1", ascending=False)

# ------------------ 5) (Optional) GenAI brief ------------------
def genai_brief(daily: pd.DataFrame, corr: pd.DataFrame) -> str:
    """健壮：corr 为空或没有 corr_t1 列也不报错，改走模板。"""
    # latest day（若 daily 为空则用今天）
    latest = daily["date"].max() if not daily.empty else pd.Timestamp.today().normalize()
    latest_day = str(pd.to_datetime(latest).date())

    # 取当天/最近一天的情感
    today = pd.DataFrame()
    if not daily.empty:
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        today = d[d["date"] == pd.to_datetime(latest).normalize()]
        if today.empty:
            # 没有今天就取最后一日
            today = d.sort_values("date").groupby("market_index").tail(1)

    pos = today.sort_values("sent_mean", ascending=False).head(3) if not today.empty else pd.DataFrame()
    neg = today.sort_values("sent_mean", ascending=True).head(3)  if not today.empty else pd.DataFrame()

    # 相关性（容错）
    top_corr = pd.DataFrame()
    if isinstance(corr, pd.DataFrame) and not corr.empty and ("corr_t1" in corr.columns):
        top_corr = corr.dropna(subset=["corr_t1"]).sort_values("corr_t1", ascending=False).head(3)

    template = []
    template.append(f"Daily brief for {latest_day}.")
    if not pos.empty:
        template.append("Most positive sentiment: " + ", ".join(
            f"{r.market_index} ({r.sent_mean:+.2f})" for r in pos.itertuples()))
    if not neg.empty:
        template.append("Most negative sentiment: " + ", ".join(
            f"{r.market_index} ({r.sent_mean:+.2f})" for r in neg.itertuples()))
    if not top_corr.empty:
        template.append("Highest T+1 correlation: " + ", ".join(
            f"{r.index} ({r.corr_t1:+.2f})" for r in top_corr.itertuples()))
    else:
        template.append("Not enough overlapping data to compute stable T+1 correlations yet.")

    base_text = " ".join(template)

    # 没有 OPENAI_KEY 就返回模板
    if not OPENAI_KEY:
        return base_text

    try:
        import openai
        # 新版 SDK
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=OPENAI_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": (
                    "Rewrite the following bullet points into a crisp 3-4 sentence analyst brief. "
                    "Neutral tone, no hype, 1 actionable takeaway if reasonable.\n\n" + base_text
                )}],
                temperature=0.2, max_tokens=180
            )
            return resp.choices[0].message.content.strip()
        # 旧版 SDK
        openai.api_key = OPENAI_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": (
                "Rewrite the following bullet points into a crisp 3-4 sentence analyst brief. "
                "Neutral tone, no hype, 1 actionable takeaway if reasonable.\n\n" + base_text
            )}],
            temperature=0.2, max_tokens=180
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return base_text + f"\n(GenAI disabled: {e})"

# ------------------ UI ------------------
st.title("Real-time Financial News — Minimal Pipeline")

with st.sidebar:
    st.subheader("Fetch settings")
    mode = st.radio("Endpoint", ["top-headlines (US/business)", "everything (time window)"], index=0)
    mode = "top" if mode.startswith("top") else "everything"
    query = st.text_input("Query (optional)", "")
    from_hours = st.slider("Look-back hours (everything)", 3, 48, 12, 1)
    page_size = st.slider("Max headlines", 10, 100, 40, 5)
    indices = st.multiselect("Indices", list(INDEX_TO_YF.keys()),
                             default=["S&P 500", "Dow Jones", "Nasdaq Composite"])
    days = st.slider("Market history days", 30, 180, 60, 10)
    st.markdown("---")
    st.caption(f"NEWSAPI_KEY: {'set' if NEWSAPI_KEY else 'missing'} • OPENAI_KEY: {'set' if OPENAI_KEY else 'missing'}")

col1, col2, col3 = st.columns([1, 1, 1])
btn_fetch = col1.button("Fetch now", type="primary")
btn_brief = col2.button("Generate brief", disabled=("daily" not in st.session_state or st.session_state.get("daily", pd.DataFrame()).empty))
btn_clear = col3.button("Clear cache")

if btn_clear:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.experimental_rerun()

if "news_df" not in st.session_state: st.session_state["news_df"] = pd.DataFrame()
if "daily"   not in st.session_state: st.session_state["daily"] = pd.DataFrame()
if "corr"    not in st.session_state: st.session_state["corr"] = pd.DataFrame()

if btn_fetch:
    df, meta = fetch_news(mode=mode, query=query, from_hours=from_hours, page_size=page_size)
    st.session_state["raw_meta"] = meta
    if df.empty:
        st.warning(f"No headlines. meta={meta}")
    else:
        df = score_news(df)
        daily = daily_sentiment(df)
        rets = fetch_index_returns(indices, days=days)
        corr = corr_same_and_t1(daily, rets)

        st.session_state["news_df"] = df
        st.session_state["daily"]   = daily
        st.session_state["corr"]    = corr
        st.success(f"Fetched {len(df)} headlines. Daily rows={len(daily)}, corr rows={len(corr)}.")

st.write("**Fetch report:**")
st.json(st.session_state.get("raw_meta", {}))

st.subheader("Fetched headlines (scored)")
df_show = st.session_state["news_df"]
if df_show.empty:
    st.info("No data yet.")
else:
    st.dataframe(df_show[["date", "headline", "sent_label", "sent_score",
                          "market_index", "news_url_unwrapped", "source"]],
                 use_container_width=True, height=320)

st.subheader("Daily sentiment vs returns (T, T+1 corr)")
corr = st.session_state["corr"]
if corr.empty:
    st.info("Fetch to compute correlations.")
else:
    st.dataframe(corr, use_container_width=True)
    if "corr_t1" in corr.columns:
        st.bar_chart(corr.set_index("index")["corr_t1"])

st.subheader("GenAI / Template Brief")
if btn_brief:
    brief = genai_brief(st.session_state["daily"], st.session_state["corr"])
    st.write(brief)
