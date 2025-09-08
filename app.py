# app.py — Financial News Sentiment & Market Trends (safe & realtime)
import os, re, io, json, requests, math, shutil, datetime as dt
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ------------------- Streamlit & Matplotlib setup -------------------
st.set_page_config(page_title="Financial News Sentiment & Market Trends",
                   layout="wide", page_icon="📈")
mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": True})

# ------------------- Default Paths (can be overridden in sidebar) -------------------
PATH_NEWS  = "financial_news_events_with_sentiment.csv"   # 主表(新闻+情绪)
PATH_DAILY = "sentiment_daily_infile.csv"
PATH_CORR  = "correlation_summary.csv"
PATH_TREND = "trend_summary.csv"

# ------------------- Small helpers -------------------
def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s))

def star(p):
    if pd.isna(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "·" if p < 0.10 else ""

def to_tznaive(s):
    """统一 tz-naive 日期"""
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def pick_sentiment_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    # prefer ensemble -> llm -> slm -> baseline -> original label
    if "predicted_sentiment_ens" in df.columns:
        return "predicted_sentiment_ens", "conf_ens" if "conf_ens" in df.columns else None
    if "predicted_sentiment_llm" in df.columns:
        return "predicted_sentiment_llm", "conf_llm" if "conf_llm" in df.columns else None
    if "predicted_sentiment_slm" in df.columns:
        return "predicted_sentiment_slm", "conf_slm" if "conf_slm" in df.columns else None
    if "predicted_sentiment_baseline" in df.columns:
        return "predicted_sentiment_baseline", "conf_baseline" if "conf_baseline" in df.columns else None
    return "sentiment", None

# ------------------- Prepare daily from news -------------------
def prepare_daily_from_news(df_news: pd.DataFrame, smooth_win: int = 3,
                            default_conf: float = 0.6) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame(columns=["market_index","date","sent_index","sent_n","mkt_ret","sent_smooth","mkt_ret_fwd","cum_index"])

    df = df_news.copy()

    # time — tz-naive days
    df["date"] = to_tznaive(df["date"])
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.normalize()

    # index column
    if "market_index" not in df.columns:
        df["market_index"] = "ALL"

    # pick sentiment
    sent_col, conf_col = pick_sentiment_columns(df)
    SENT_VAL = {"Negative":-1.0, "Neutral":0.0, "Positive":1.0}
    IMPACT_W = {"High":1.0, "Medium":0.6, "Low":0.3}

    df["label"] = df[sent_col].astype(str).str.title()
    df["val"]   = df["label"].map(SENT_VAL).fillna(0.0)
    df["w"]     = df.get("impact_level", "Medium").map(IMPACT_W).fillna(0.6)
    df["conf"]  = pd.to_numeric(df.get(conf_col, default_conf), errors="coerce").fillna(default_conf).clip(0,1)
    df["score"] = df["val"] * df["w"] * df["conf"]

    # market return (percent → decimal if needed)
    if "index_change_percent" in df.columns:
        df["ret"] = pd.to_numeric(df["index_change_percent"], errors="coerce")
        if df["ret"].abs().max() > 1.0:
            df["ret"] = df["ret"]/100.0
    else:
        df["ret"] = np.nan

    daily = (df.groupby(["market_index","date"])
               .agg(sent_index=("score","mean"),
                    sent_n=("score","size"),
                    mkt_ret=("ret","mean"))
               .reset_index()
               .sort_values(["market_index","date"])
            )

    # smoothing & t+1
    daily["sent_smooth"] = daily.groupby("market_index")["sent_index"]\
                                .transform(lambda s: s.rolling(smooth_win, min_periods=1).mean())
    daily["mkt_ret_fwd"] = daily.groupby("market_index")["mkt_ret"].shift(-1)

    # cumulative (=100)
    daily["cum_index"] = (daily.groupby("market_index")["mkt_ret"]
                          .apply(lambda s: (1.0 + s.fillna(0)).cumprod()*100.0)
                          .reset_index(level=0, drop=True))
    return daily

# ------------------- Correlation & trend summaries -------------------
def compute_corr_summary(daily: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=[
            "Index","N (t→t+1)","Pearson (t→t+1)","p (t→t+1)",
            "Spearman (t→t+1)","p_s (t→t+1)","N (t→t)","Pearson (t→t)","p (t→t)",
            "Spearman (t→t)","p_s (t→t)"
        ])
    rows = []
    for idx, g in daily.groupby("market_index"):
        g0 = g.dropna(subset=["sent_smooth","mkt_ret"])
        g1 = g.dropna(subset=["sent_smooth","mkt_ret_fwd"])

        if len(g0) >= min_n:
            r0, p0 = pearsonr(g0["sent_smooth"], g0["mkt_ret"])
            s0, q0 = spearmanr(g0["sent_smooth"], g0["mkt_ret"])
        else:
            r0=p0=s0=q0=np.nan

        if len(g1) >= min_n:
            r1, p1 = pearsonr(g1["sent_smooth"], g1["mkt_ret_fwd"])
            s1, q1 = spearmanr(g1["sent_smooth"], g1["mkt_ret_fwd"])
        else:
            r1=p1=s1=q1=np.nan

        rows.append({
            "Index": idx,
            "N (t→t+1)": len(g1), "Pearson (t→t+1)": r1, "p (t→t+1)": p1,
            "Spearman (t→t+1)": s1, "p_s (t→t+1)": q1,
            "N (t→t)": len(g0), "Pearson (t→t)": r0, "p (t→t)": p0,
            "Spearman (t→t)": s0, "p_s (t→t)": q0,
        })
    corr_df = pd.DataFrame(rows).sort_values("Pearson (t→t+1)", ascending=False)
    return corr_df

def compute_trend_summary(daily: pd.DataFrame, roll_win: int = 21, min_n: int = 10) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["Index","N_total","Pearson_all","p_all","RollWin","r_roll_mean","r_roll_median","r_roll_pos_share"])
    rows=[]
    for idx, g in daily.groupby("market_index"):
        gg = g.dropna(subset=["sent_smooth","mkt_ret_fwd"]).copy()
        if len(gg) >= min_n:
            r_all, p_all = pearsonr(gg["sent_smooth"], gg["mkt_ret_fwd"])
            gg["r_roll"] = gg["sent_smooth"].rolling(roll_win, min_periods=roll_win).corr(gg["mkt_ret_fwd"])
            rows.append({
                "Index": idx,
                "N_total": len(gg),
                "Pearson_all": round(r_all,3),
                "p_all": float(f"{p_all:.3g}"),
                "RollWin": roll_win,
                "r_roll_mean": round(float(np.nanmean(gg["r_roll"])),3) if gg["r_roll"].notna().any() else np.nan,
                "r_roll_median": round(float(np.nanmedian(gg["r_roll"])),3) if gg["r_roll"].notna().any() else np.nan,
                "r_roll_pos_share": round(float(np.nanmean((gg["r_roll"]>0).astype(float))),3) if gg["r_roll"].notna().any() else np.nan
            })
        else:
            rows.append({"Index":idx,"N_total":len(gg),"Pearson_all":np.nan,"p_all":np.nan,
                         "RollWin":roll_win,"r_roll_mean":np.nan,"r_roll_median":np.nan,"r_roll_pos_share":np.nan})
    return pd.DataFrame(rows).sort_values("Pearson_all", ascending=False)

# ------------------- Plots -------------------
def plot_corr_bar(corr_df: pd.DataFrame) -> plt.Figure:
    if corr_df.empty:
        return plt.figure(figsize=(6,3))
    d = corr_df[["Index","N (t→t+1)","Pearson (t→t+1)","p (t→t+1)"]].dropna(subset=["Pearson (t→t+1)"]).copy()
    d = d.sort_values("Pearson (t→t+1)", ascending=True)
    d["star"] = d["p (t→t+1)"].map(star)
    pos_color, neg_color = "#1f77b4", "#d62728"
    colors = [pos_color if r>=0 else neg_color for r in d["Pearson (t→t+1)"]]
    fig_h = max(3, 0.35*len(d)+1.2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    bars = ax.barh(d["Index"], d["Pearson (t→t+1)"], color=colors)
    ax.axvline(0, ls="--", lw=1, color="#666", alpha=0.7)
    ax.set_xlabel("Pearson r (Sent_t → Ret_(t+1))")
    ax.set_title("Indices — Next-day correlation (with significance)\nNote: * p<0.05, ** p<0.01, *** p<0.001, · p<0.10")
    for rect, pval in zip(bars, d["p (t→t+1)"].values):
        if pd.notna(pval) and pval < 0.05:
            rect.set_edgecolor("#333"); rect.set_linewidth(1.2)
    for yi, (rv, st_sym, n_sample) in enumerate(zip(d["Pearson (t→t+1)"].values,
                                                   d["star"].values,
                                                   d["N (t→t+1)"].astype(int).values)):
        txt = f"{rv:+.3f}{st_sym}   N={n_sample}"
        off = 0.01
        if rv >= 0:
            ax.text(rv + off, yi, txt, ha="left", va="center", fontsize=9)
        else:
            ax.text(rv - off, yi, f"N={n_sample}   {rv:+.3f}{st_sym}",
                    ha="right", va="center", fontsize=9)
    plt.tight_layout()
    return fig

def plot_dual_lines(g: pd.DataFrame, smooth_win: int = 3, title_prefix: str = ""):
    g = g.sort_values("date").copy()
    gg = g.dropna(subset=["sent_smooth","mkt_ret_fwd"])
    title = title_prefix
    if len(gg) >= 10:
        r, p = pearsonr(gg["sent_smooth"], gg["mkt_ret_fwd"])
        title = f"{title_prefix}: Sentiment vs Market | Pearson(t→t+1)={r:+.3f}, p={p:.3g}, n={len(gg)}"
    fig, ax1 = plt.subplots(figsize=(9.5, 4.6))
    ax1.plot(g["date"], g["sent_smooth"], lw=2, label=f"Sentiment ({smooth_win}D)")
    ax1.axhline(0, ls="--", lw=1, alpha=0.6)
    ax1.set_ylabel("Sentiment (weighted)")
    ax2 = ax1.twinx()
    ax2.plot(g["date"], g["cum_index"], lw=1.6, alpha=0.9, label="Cumulative (=100)")
    ax2.set_ylabel("Cumulative")
    ax1.set_title(title)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lab1+lab2, loc="upper left")
    plt.tight_layout()
    return fig

def plot_rolling_corr(g: pd.DataFrame, roll_win: int = 21) -> Optional[plt.Figure]:
    gg = g.dropna(subset=["sent_smooth","mkt_ret_fwd"]).copy()
    if len(gg) < roll_win:
        return None
    gg["r_roll"] = gg["sent_smooth"].rolling(roll_win, min_periods=roll_win).corr(gg["mkt_ret_fwd"])
    if gg["r_roll"].dropna().empty:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(gg["date"], gg["r_roll"], lw=1.8)
    ax.axhline(0, ls="--", lw=1, alpha=0.6)
    ax.set_ylabel(f"Rolling r ({roll_win}D)\nSent_t vs Ret_(t+1)")
    ax.set_title(f"Rolling correlation ({roll_win}D)")
    plt.tight_layout()
    return fig

# ------------------- Daily brief (template) -------------------
def build_daily_brief(corr_df: pd.DataFrame, daily_df: pd.DataFrame) -> str:
    if daily_df.empty:
        return "No data available."
    d = daily_df.copy()
    d["date"] = to_tznaive(d["date"])
    as_of_ts = d["date"].max()
    as_of_date = as_of_ts.date()

    pos = corr_df[(corr_df["p (t→t+1)"] < 0.10) & (corr_df["Pearson (t→t+1)"] > 0)]\
              .sort_values("Pearson (t→t+1)", ascending=False).head(3)
    neg = corr_df[(corr_df["p (t→t+1)"] < 0.10) & (corr_df["Pearson (t→t+1)"] < 0)]\
              .sort_values("Pearson (t→t+1)").head(3)

    latest = (d.sort_values("date")
                .groupby("market_index").tail(1)
                .sort_values("sent_index", ascending=False).head(3))

    lb = d[d["date"] >= (as_of_ts - pd.Timedelta(days=3))]
    movers = (lb.groupby("market_index")["sent_index"]
                .agg(["first","last"])
                .assign(delta=lambda x: x["last"]-x["first"])
                .sort_values("delta", ascending=False).head(3))

    def _fmt_rows(df):
        if df.empty: return "none"
        return "; ".join(f"{r.Index} r={r['Pearson (t→t+1)']:+.3f}{star(r['p (t→t+1)'])}"
                         for _, r in df.iterrows())

    lines = []
    lines.append(f"As of {as_of_date}, the news-sentiment index shows mixed conditions across major markets.")
    lines.append("Statistically significant next-day correlations include: "
                 f"positive — {_fmt_rows(pos)}; negative — {_fmt_rows(neg)}.")
    if not latest.empty:
        lines.append("Today’s highest sentiment levels: " + "; ".join(
            f"{r.market_index} {r.sent_index:+.3f}" for _, r in latest.iterrows()) + ".")
    if not movers.empty:
        lines.append("Three-day sentiment momentum (top movers): " + "; ".join(
            f"{idx} {row['delta']:+.3f}" for idx, row in movers.iterrows()) + ".")
    lines.append("Correlations are modest and do not imply causality. This brief is auto-generated from the latest metrics.")
    return " ".join(lines)

# ------------------- Real-time ingestion (NewsAPI / RSS) -------------------
YF_INDEX = {
    "S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq Composite": "^IXIC",
    "FTSE 100": "^FTSE", "DAX": "^GDAXI", "Euro Stoxx 50": "^STOXX50E",
    "CAC 40": "^FCHI", "Nikkei 225": "^N225", "Hang Seng": "^HSI",
    "ASX 200": "^AXJO", "S&P/TSX Composite": "^GSPTSE", "KOSPI": "^KS11",
    "IBOVESPA": "^BVSP", "NSE Nifty": "^NSEI", "Shanghai Composite": "000001.SS",
}
KEY2INDEX = [
    (["s&p 500","s&p500","spx"], "S&P 500"),
    (["dow jones","djia"], "Dow Jones"),
    (["nasdaq"], "Nasdaq Composite"),
    (["ftse"], "FTSE 100"),
    (["dax"], "DAX"),
    (["stoxx"], "Euro Stoxx 50"),
    (["cac 40","cac40"], "CAC 40"),
    (["nikkei"], "Nikkei 225"),
    (["hang seng","hsi"], "Hang Seng"),
    (["asx"], "ASX 200"),
    (["tsx","s&p/tsx"], "S&P/TSX Composite"),
    (["kospi"], "KOSPI"),
    (["ibovespa","bovespa"], "IBOVESPA"),
    (["nifty","nse"], "NSE Nifty"),
    (["shanghai","sse","上证","沪指"], "Shanghai Composite"),
]
def map_index_from_headline(title: str) -> str:
    t = (title or "").lower()
    for kws, idx in KEY2INDEX:
        if any(k in t for k in kws):
            return idx
    return "ALL"

@st.cache_data(show_spinner=False)
def _finance_domains(level: int = 2) -> List[str]:
    core = [
        "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "marketwatch.com",
        "investing.com", "yahoo.com", "cnbc.com", "seekingalpha.com", "morningstar.com",
        "thestreet.com", "nasdaq.com", "barrons.com", "investopedia.com", "apnews.com",
        "cnn.com", "bbc.com", "forbes.com", "economist.com", "finance.yahoo.com",
    ]
    if level <= 1:
        return core
    if level == 2:
        return core + ["pymnts.com", "businessinsider.com", "fool.com", "fortune.com"]
    # 最严格
    return core + ["pymnts.com", "businessinsider.com", "fool.com", "fortune.com", "markets.businessinsider.com"]

def _host(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", re.split(r"/+", url)[2])
    except Exception:
        return ""

def filter_finance(df: pd.DataFrame, enabled: bool, level: int) -> pd.DataFrame:
    if not enabled or df.empty:
        return df
    wl = set(_finance_domains(level))
    out = df.copy()
    out["__host"] = out["news_url"].astype(str).map(_host)
    out = out[out["__host"].isin(wl)]
    out = out.drop(columns=["__host"])
    return out

def fetch_newsapi(api_key: str, query: str="", from_hours: int=12,
                  page_size: int=100, language: str="en", source_mode: str="top-headlines",
                  max_pages: int = 1) -> Tuple[pd.DataFrame, dict]:
    if not api_key:
        # 返回带标准列名的空表，避免下游 KeyError
        return pd.DataFrame(columns=["date","headline","news_url","source","market_index"]), {"code":0, "raw":{}}

    if source_mode not in ("top-headlines","everything"):
        source_mode = "top-headlines"

    if source_mode == "everything":
        tfrom = (pd.Timestamp.utcnow() - pd.Timedelta(hours=from_hours)).isoformat(timespec="seconds").replace("+00:00","Z")
        base_params = {
            "q": query or "stocks OR markets OR index OR economy",
            "from": tfrom, "language": language, "sortBy": "publishedAt",
            "pageSize": min(page_size, 100), "apiKey": api_key,
        }
        url = "https://newsapi.org/v2/everything"
    else:
        base_params = {
            "language": language, "pageSize": min(page_size, 100), "apiKey": api_key,
        }
        if not query:
            base_params["category"] = "business"
        else:
            base_params["q"] = query
        url = "https://newsapi.org/v2/top-headlines"

    rows = []
    raw_any = {}
    for page in range(1, max_pages+1):
        params = dict(base_params); params["page"] = page
        try:
            r = requests.get(url, params=params, timeout=20)
            raw_any = r.json()
            r.raise_for_status()
            arts = raw_any.get("articles", []) or []
            for a in arts:
                rows.append({
                    "date": a.get("publishedAt"),
                    "headline": (a.get("title") or "").strip(),
                    "news_url": a.get("url"),
                    "source": (a.get("source") or {}).get("name"),
                })
            if len(arts) < params["pageSize"]:
                break
        except Exception:
            break

    meta = {"code": r.status_code if 'r' in locals() else 0, "raw": raw_any, "source": source_mode, "count": len(rows)}

    # —— 关键：空结果保护 + 统一列名 ——
    if not rows:
        return pd.DataFrame(columns=["date","headline","news_url","source","market_index"]), meta

    df = pd.DataFrame(rows)
    # 统一为 tz-naive
    df["date"] = to_tznaive(df["date"])
    # 即便全是 NaT，也不会 KeyError
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=["date","headline","news_url","source","market_index"]), meta

    df = df.sort_values("date", ascending=False).drop_duplicates(subset=["news_url"])
    df["market_index"] = df["headline"].apply(map_index_from_headline)
    return df, meta


def fetch_rss(n_limit: int = 120) -> pd.DataFrame:
    try:
        import feedparser
    except Exception:
        return pd.DataFrame(columns=["date","headline","news_url","source","market_index"])

    FEEDS = [
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.ft.com/companies?format=rss",
        "https://www.reuters.com/markets/rss",
        "https://www.cnbc.com/id/15839135/device/rss/rss.html",
    ]
    rows=[]
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:max(1, n_limit//len(FEEDS))]:
                rows.append({
                    "date": getattr(e, "published", None) or getattr(e, "updated", None),
                    "headline": getattr(e, "title", ""),
                    "news_url": getattr(e, "link", ""),
                    "source": feed.feed.get("title", ""),
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["date","headline","news_url","source","market_index"])

    df = pd.DataFrame(rows)
    df["date"] = to_tznaive(df["date"])
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=["date","headline","news_url","source","market_index"])

    df = df.sort_values("date", ascending=False).drop_duplicates(subset=["news_url"])
    df["market_index"] = df["headline"].apply(map_index_from_headline)
    return df


def fetch_returns_for_indices(indices: List[str], days: int=7) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()
    tickers = [YF_INDEX[i] for i in indices if i in YF_INDEX]
    if not tickers: return pd.DataFrame()
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days+5)
    df = yf.download(tickers=tickers, start=start.date(), end=end.date(), progress=False)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.fillna(method="ffill")
    out=[]
    for idx, tic in YF_INDEX.items():
        if tic in df.columns:
            px = df[tic].dropna()
            ret = px.pct_change()
            tmp = ret.reset_index().rename(columns={"Date":"date", tic:"ret"})
            tmp["date"] = to_tznaive(tmp["date"])
            tmp["market_index"] = idx
            out.append(tmp)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# ------------------- Lightweight classifiers for real-time -------------------
@st.cache_resource(show_spinner=False)
def get_slm():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() \
             else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    net = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device).eval()
    return tok, net, device

def slm_predict(texts: List[str], neutral_band=(0.45,0.55), max_len=128, bs=64) -> List[str]:
    import torch
    tok, net, device = get_slm()
    outs=[]
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = net(**enc).logits
            ppos = torch.softmax(logits, dim=-1)[:,1].detach().cpu().tolist()
            for p in ppos:
                if neutral_band[0] <= p <= neutral_band[1]:
                    outs.append("Neutral")
                else:
                    outs.append("Positive" if p>neutral_band[1] else "Negative")
    return outs

@st.cache_resource(show_spinner=False)
def load_baseline_model():
    try:
        from joblib import load
        model = load("models/tfidf_lr.joblib")
        return model
    except Exception:
        return None

def baseline_predict(texts: List[str]) -> Optional[List[str]]:
    model = load_baseline_model()
    if model is None:
        return None
    try:
        return model.predict(texts).tolist()
    except Exception:
        return None

def rt_classify_and_aggregate(df_news: pd.DataFrame, smooth_win: int=3, engine: str="SLM") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_news.empty:
        return df_news, pd.DataFrame()
    df = df_news.copy()

    texts = df["headline"].fillna("").astype(str).tolist()
    if engine == "Baseline + SLM":
        bl = baseline_predict(texts)
        if bl is not None:
            df["predicted_sentiment_slm"] = bl
        else:
            df["predicted_sentiment_slm"] = slm_predict(texts)
    else:
        df["predicted_sentiment_slm"] = slm_predict(texts)

    val_map = {"Negative":-1, "Neutral":0, "Positive":1}
    df["score"] = df["predicted_sentiment_slm"].map(val_map).fillna(0) * 0.6 * 0.6  # 简单权重
    # 统一 date 为 tz-naive
    df["date"] = to_tznaive(df["date"])

    # D 聚合
    daily = (df.groupby([pd.Grouper(key="date", freq="D"), "market_index"])
               .agg(sent_index=("score","mean"), sent_n=("score","size"))
               .reset_index()
             ).sort_values(["market_index","date"])
    daily["sent_smooth"] = daily.groupby("market_index")["sent_index"]\
                                .transform(lambda s: s.rolling(smooth_win, min_periods=1).mean())
    return df, daily

# ------------------- Sidebar -------------------
st.sidebar.header("Data")
smooth_win = st.sidebar.slider("Sentiment smoothing window (days)", 1, 7, 3, 1)
min_n      = st.sidebar.slider("Min samples for correlation", 5, 40, 10, 1)
roll_win   = st.sidebar.slider("Rolling window for correlation (days)", 10, 60, 21, 1)

# --- Data source: choose main CSV path and show row count ---
st.sidebar.markdown("### Data source")
default_news_path = PATH_NEWS
news_path_input = st.sidebar.text_input(
    "Main CSV path (news + sentiments)",
    value=default_news_path,
    help="确保这是你想要追加的数据文件。支持相对或绝对路径。"
)
PATH_NEWS = news_path_input.strip() or default_news_path

def _try_read_rows(fp):
    try:
        return len(pd.read_csv(fp))
    except Exception:
        return 0

st.sidebar.caption(f"Using: **{os.path.abspath(PATH_NEWS)}**")
st.sidebar.write(f"Current rows: **{_try_read_rows(PATH_NEWS):,}**")

# --- Lightweight Real-Time controls ---
st.sidebar.markdown("### Lightweight Real-Time")
use_newsapi   = st.sidebar.checkbox("Fetch via NewsAPI (recommended)", value=True)
api_key_default = st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY", ""))
api_key_input   = st.sidebar.text_input("NewsAPI key (local only)", value=api_key_default if use_newsapi else "", type="password")
news_query      = st.sidebar.text_input("NewsAPI query (optional)", value="", disabled=not use_newsapi)
from_hours      = st.sidebar.slider("Lookback hours", 3, 48, 12, 1)
rss_max         = st.sidebar.number_input("RSS max headlines (fallback)", min_value=20, max_value=500, value=120, step=10, disabled=use_newsapi)

st.sidebar.markdown("### Speed controls")
engine = st.sidebar.selectbox("RT classifier", ["SLM", "Baseline + SLM"])
enable_extractive = st.sidebar.checkbox("Enable GenAI summarization (extractive)", value=False)
max_rt = st.sidebar.slider("Max headlines per fetch", 5, 50, 20, 1)

st.sidebar.markdown("### Finance filter")
restrict_fin = st.sidebar.checkbox("Restrict to finance domains", value=True)
fin_level = st.sidebar.slider("Finance filter level (higher = stricter)", 0, 4, 2, 1)

st.sidebar.markdown("### Files expected in this folder:")
st.sidebar.code("\n".join([os.path.basename(PATH_NEWS), PATH_DAILY, PATH_CORR, PATH_TREND]))

# ------------------- Data loading / ensuring -------------------
news_df  = load_csv(PATH_NEWS)
daily_df = load_csv(PATH_DAILY)
corr_df  = load_csv(PATH_CORR)
trend_df = load_csv(PATH_TREND)

if daily_df.empty and not news_df.empty:
    daily_df = prepare_daily_from_news(news_df, smooth_win=smooth_win)
    daily_df.to_csv(PATH_DAILY, index=False, encoding="utf-8-sig")
if corr_df.empty and not daily_df.empty:
    corr_df = compute_corr_summary(daily_df, min_n=min_n)
    corr_df.to_csv(PATH_CORR, index=False, encoding="utf-8-sig")
if trend_df.empty and not daily_df.empty:
    trend_df = compute_trend_summary(daily_df, roll_win=roll_win, min_n=min_n)
    trend_df.to_csv(PATH_TREND, index=False, encoding="utf-8-sig")

# ------------------- UI -------------------
st.title("Financial News Sentiment & Market Trends")

tabs = st.tabs(["Overview", "Explore Index", "Correlation", "Rolling Corr", "Daily Brief", "Real-Time (News)"])

# -------- Overview --------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("News rows", f"{len(news_df):,}")
    c2.metric("Daily rows", f"{len(daily_df):,}")
    c3.metric("Indices", f"{daily_df['market_index'].nunique() if not daily_df.empty else 0:,}")
    st.markdown("---")
    if not corr_df.empty:
        fig = plot_corr_bar(corr_df)
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Correlation table is empty.")

# -------- Explore Index --------
with tabs[1]:
    if daily_df.empty:
        st.warning("Daily file is empty.")
    else:
        idx_list = sorted(daily_df["market_index"].unique().tolist())
        idx_pick = st.selectbox("Choose index", idx_list, index=0)
        g = daily_df[daily_df["market_index"]==idx_pick]
        st.dataframe(g.tail(20), use_container_width=True)
        fig = plot_dual_lines(g, smooth_win, title_prefix=idx_pick)
        st.pyplot(fig); plt.close(fig)

# -------- Correlation --------
with tabs[2]:
    if corr_df.empty:
        st.warning("Correlation table is empty.")
    else:
        st.dataframe(corr_df, use_container_width=True)
        fig = plot_corr_bar(corr_df); st.pyplot(fig); plt.close(fig)
        buf = io.BytesIO()
        corr_df.to_csv(buf, index=False); buf.seek(0)
        st.download_button("Download correlation (.csv)", buf, file_name="correlation_summary.csv", mime="text/csv")

# -------- Rolling Corr --------
with tabs[3]:
    if daily_df.empty:
        st.warning("Daily file is empty.")
    else:
        idx_list = sorted(daily_df["market_index"].unique().tolist())
        idx_pick = st.selectbox("Choose index for rolling corr", idx_list, index=0, key="roll_idx")
        g = daily_df[daily_df["market_index"]==idx_pick]
        fig = plot_rolling_corr(g, roll_win=roll_win)
        if fig:
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("Not enough data for rolling correlation.")

# -------- Daily Brief --------
with tabs[4]:
    if corr_df.empty or daily_df.empty:
        st.info("Need correlation & daily data.")
    else:
        brief = build_daily_brief(corr_df, daily_df)
        st.subheader("Auto-generated English Daily Brief")
        st.text_area("Daily brief", value=brief, height=220)
        st.download_button("Download brief (.txt)",
                           brief.encode("utf-8"), file_name="ai_brief.txt", mime="text/plain")
    st.caption("© Your Name — For academic research only. No investment advice.")

# -------- Real-Time --------
with tabs[5]:
    st.subheader("Real-time ingestion")
    colA, colB, colC = st.columns([1,1,1])
    fetch_btn  = colA.button("Fetch now")
    append_btn = colB.button("Append to main CSV & rebuild daily/corr")
    test_btn   = colC.button("Test NewsAPI key")

    if "rt_df" not in st.session_state:    st.session_state["rt_df"] = pd.DataFrame()
    if "rt_daily" not in st.session_state: st.session_state["rt_daily"] = pd.DataFrame()

    # --- Test key ---
    if test_btn and use_newsapi:
        try:
            r = requests.get("https://newsapi.org/v2/top-headlines",
                             params={"language":"en","pageSize":1,"apiKey":api_key_input}, timeout=15)
            st.info(f"HTTP={r.status_code} → {r.text[:250]}")
        except Exception as e:
            st.exception(e)

    # --- Fetch ---
    if fetch_btn:
        if use_newsapi:
            df_rt, meta = fetch_newsapi(api_key_input, query=news_query, from_hours=int(from_hours),
                                        page_size=max_rt, source_mode="top-headlines", max_pages=1)
            st.info(f"NewsAPI fetched: {meta.get('count',0)} headlines. (source={meta.get('source')} code={meta.get('code')})")
        else:
            df_rt = fetch_rss(n_limit=int(rss_max))
            st.info(f"RSS fetched: {len(df_rt)} headlines.")

        # 金融域过滤
        df_rt = filter_finance(df_rt, restrict_fin, fin_level)
        # 限制数量
        if len(df_rt) > max_rt:
            df_rt = df_rt.head(max_rt)

        if df_rt.empty:
            st.warning("No data fetched.")
        else:
            # classify & aggregate
            df_rt, daily_rt = rt_classify_and_aggregate(df_rt, smooth_win=smooth_win, engine=engine)
            st.session_state["rt_df"] = df_rt.copy()
            st.session_state["rt_daily"] = daily_rt.copy()
            st.success(f"Fetched {len(df_rt)} headlines.")

            # 简单“抽取式要点”
            if enable_extractive:
                st.markdown("#### Extractive summary (top lines):")
                # 规则：近期+包含关键词优先
                key_terms = ["rate","inflation","jobs","fed","market","stocks","index","earnings","growth","trade","tariff","oil","bond","yields","tech","chip"]
                df_sc = df_rt.copy()
                score = []
                for t in df_sc["headline"].fillna("").astype(str):
                    s = 0
                    tl = t.lower()
                    for k in key_terms:
                        if k in tl: s += 1
                    s += min(3, len(t)//50)  # 长标题少许加分
                    score.append(s)
                df_sc["__score"] = score
                for line in df_sc.sort_values(["__score","date"], ascending=[False, False]).head(min(12, len(df_sc)))["headline"].tolist():
                    st.markdown(f"- {line}")

    # --- Show RT table & chart ---
    if not st.session_state["rt_df"].empty:
        st.markdown("#### Latest headlines (classified):")
        st.dataframe(
            st.session_state["rt_df"][["date","headline","predicted_sentiment_slm","market_index","news_url"]]
            .reset_index(drop=True),
            use_container_width=True, height=320
        )

        idx_list_rt = sorted(st.session_state["rt_daily"]["market_index"].unique().tolist())
        idx_pick_rt = st.selectbox("Select index for RT chart", idx_list_rt, index=0, key="rt_idx")

        g = st.session_state["rt_daily"][st.session_state["rt_daily"]["market_index"]==idx_pick_rt].copy()
        g = g.sort_values("date")
        use_yf = st.checkbox("Also plot recent market cumulative via yfinance", value=False)
        if use_yf:
            df_ret = fetch_returns_for_indices([idx_pick_rt], days=7)
            if not df_ret.empty:
                g = g.merge(df_ret, on=["date","market_index"], how="left")
                g["cum_index"] = (1 + g["ret"].fillna(0)).cumprod() * 100.0

        fig, ax1 = plt.subplots(figsize=(9.5, 4.4))
        ax1.plot(g["date"], g["sent_smooth"], lw=2, label=f"RT Sentiment ({smooth_win}D)")
        ax1.axhline(0, ls="--", lw=1, alpha=0.6)
        ax1.set_ylabel("Sentiment (weighted)")
        title = f"Real-time: {idx_pick_rt} — Sentiment"
        if "cum_index" in g.columns and g["cum_index"].notna().any():
            ax2 = ax1.twinx()
            ax2.plot(g["date"], g["cum_index"], lw=1.6, alpha=0.9, label="Cumulative (=100)")
            ax2.set_ylabel("Cumulative")
            title += " vs Market"
            l1, lab1 = ax1.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(l1+l2, lab1+lab2, loc="upper left")
        else:
            ax1.legend(loc="upper left")
        ax1.set_title(title)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # --- Append ---
    if append_btn:
        add = st.session_state.get("rt_df", pd.DataFrame()).copy()
        if add.empty:
            st.warning("Nothing to append. Please click **Fetch now** first.")
        else:
            # 读取当前主表（不经 cache）
            try:
                base = pd.read_csv(PATH_NEWS)
            except FileNotFoundError:
                base = pd.DataFrame()

            before = len(base)

            def _normalize(df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                # URL 标准化（去重更稳）
                if "news_url" in df.columns:
                    df["news_url"] = df["news_url"].astype(str).str.strip().str.lower()
                else:
                    df["news_url"] = np.nan
                # 日期 tz-naive
                if "date" in df.columns:
                    df["date"] = to_tznaive(df["date"])
                # 市场索引兜底
                if "market_index" not in df.columns:
                    df["market_index"] = "ALL"
                return df

            base = _normalize(base)
            add  = _normalize(add)

            merged = pd.concat([base, add], ignore_index=True)
            if "news_url" in merged.columns:
                merged = merged.drop_duplicates(subset=["news_url"])
            merged = merged.sort_values("date", na_position="last")

            after = len(merged)
            added = after - before

            # Sanity：禁止把大表写小
            if before > 0 and after < before:
                st.error(
                    f"Sanity check failed: merged rows ({after:,}) < original ({before:,}). "
                    "Abort writing to avoid losing data. Please check PATH_NEWS."
                )
            else:
                # 备份 + 写回
                try:
                    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    bak = f"{PATH_NEWS}.bak_{ts}"
                    if os.path.exists(PATH_NEWS):
                        shutil.copyfile(PATH_NEWS, bak)
                    merged.to_csv(PATH_NEWS, index=False, encoding="utf-8-sig")
                    st.success(f"Appended {max(0, added):,} rows (before={before:,} → after={after:,}). Backup: {os.path.abspath(bak)}")
                except Exception as e:
                    st.exception(e)

                # 重建日频/相关/趋势
                try:
                    daily_new = prepare_daily_from_news(merged, smooth_win=smooth_win)
                    daily_new.to_csv(PATH_DAILY, index=False, encoding="utf-8-sig")
                    corr_new  = compute_corr_summary(daily_new, min_n=min_n)
                    corr_new.to_csv(PATH_CORR, index=False, encoding="utf-8-sig")
                    trend_new = compute_trend_summary(daily_new, roll_win=roll_win, min_n=min_n)
                    trend_new.to_csv(PATH_TREND, index=False, encoding="utf-8-sig")
                    st.success("Daily/correlation/trend rebuilt.")
                except Exception as e:
                    st.exception(e)

                # 清缓存并刷新
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                if hasattr(st, "rerun"):
                    st.rerun()
                elif hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
                else:
                    st.info("Append 成功。当前 Streamlit 无自动刷新，请手动 Rerun。")
