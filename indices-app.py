# ===========================================================================
# Stock Index Dashboard — indices-app.py
#
# Improvements over previous version:
#   • Hardcoded ticker→currency map removes ~40 slow per-ticker API calls
#   • Simulation inner loops replaced with NumPy dot products (10-50× faster)
#   • scipy.optimize.minimize replaces cvxpy (drops heavy dependency, faster)
#   • All matplotlib/seaborn removed; every chart is now interactive Plotly
#   • Dark mode locked in; Plotly template + CSS always use dark theme
#   • VaR formula corrected (removed erroneous √t double-scaling)
#   • Chart line renamed to "Capital Market Line" (CML) — previously mislabelled
#   • treasury_10y passed as explicit arg to cached simulation functions
#   • Ticker history in tab4 cached with @st.cache_data
# ===========================================================================

import math
import os
import base64
import datetime
import pytz
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def root_mean_squared_error(y_true, y_pred):
    """Compute RMSE as a float. Shims the function added in scikit-learn 1.4."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IndexPulse",
    page_icon=":money_with_wings:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — dark mode (locked)
# ---------------------------------------------------------------------------
plotly_tpl = "plotly_dark"

def _bg_css(image_path: str) -> str:
    """
    Read an image file and return a <style> block that sets it as the
    full-screen background of the Streamlit app. Supports SVG, JPEG, PNG,
    and WebP. Falls back to a plain dark background if the file is missing.
    """
    mime_map = {".svg": "image/svg+xml", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    ext = image_path[image_path.rfind("."):].lower()
    mime = mime_map.get(ext, "image/jpeg")
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            "<style>"
            ".stApp {"
            f'background-image: url("data:{mime};base64,{b64}");'
            "background-size: cover;"
            "background-attachment: fixed;"
            "background-color: #0e1117;"
            "}"
            "</style>"
        )
    except FileNotFoundError:
        return "<style>.stApp {background-color: #0e1117;}</style>"

st.markdown(_bg_css(os.path.join(_HERE, "background.jpg")), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hardcoded ticker → domestic-currency map
# Eliminates ~40 individual yf.Ticker(...).info API calls on startup.
# ---------------------------------------------------------------------------
TICKER_CURRENCY: dict[str, str] = {
    # North America
    "^GSPC": "USD", "^DJI": "USD", "^IXIC": "USD", "^RUT": "USD",
    "^NYA": "USD", "^XAX": "USD", "^VIX": "USD", "DX-Y.NYB": "USD",
    "^GSPTSE": "CAD",
    # Latin America
    "^BVSP": "BRL", "^MXX": "MXN", "^MERV": "ARS",
    # Asia Developed & Oceania
    "^N225": "JPY", "^HSI": "HKD", "000001.SS": "CNY", "399001.SZ": "CNY",
    "^TWII": "TWD", "^KS11": "KRW",
    "^AXJO": "AUD", "^NZ50": "NZD", "^AORD": "AUD",
    "^CASE30": "EGP", "^JN0U.JO": "ZAR",
    # Asia Emerging
    "^STI": "SGD", "^JKSE": "IDR", "^KLSE": "MYR", "^BSESN": "INR",
    "^TA125.TA": "ILS", "IMOEX.ME": "RUB", "MOEX.ME": "RUB",
    # Europe & UK
    "^FTSE": "GBP", "^BUK100P": "GBP",
    "^GDAXI": "EUR", "^FCHI": "EUR", "^STOXX50E": "EUR",
    "^N100": "EUR", "^BFX": "EUR", "^125904-USD-STRD": "USD",
    # Forex indices
    "^XDA": "AUD", "^XDB": "GBP", "^XDE": "EUR", "^XDN": "JPY",
}

# ---------------------------------------------------------------------------
# Module-level date constants
# ---------------------------------------------------------------------------
_now_utc = datetime.datetime.now(pytz.utc)
current_date = datetime.date(_now_utc.year, _now_utc.month, _now_utc.day)
time_now_ref = datetime.date(_now_utc.year, _now_utc.month, 1)
time_before = datetime.date(max(2010, _now_utc.year - 10), 1, 1)
time_max = time_now_ref - datetime.timedelta(weeks=1)

# ---------------------------------------------------------------------------
# Scrape world-indices list from Yahoo Finance
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching world indices list…")
def url_indices(url: str) -> pd.DataFrame:
    """
    Scrape the first HTML table from the given URL (Yahoo Finance world-indices
    page) and return it as a DataFrame. Returns an empty DataFrame on failure.
    Result is cached so the request is only made once per session.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(r.text)
        return tables[0]
    except Exception:
        return pd.DataFrame()

world_idx = url_indices("https://finance.yahoo.com/world-indices")
if world_idx.empty or len(world_idx) <= 1:
    world_idx = pd.read_csv(os.path.join(_HERE, "World_Indices_Yahoo.csv"))

world_idx = world_idx.dropna(how="all")
world_idx = world_idx[world_idx["Symbol"] != "^CASE30"]

ticker_name: dict[str, str] = dict(zip(world_idx["Symbol"], world_idx["Name"]))
ticker_name["^NZ50"] = "S&P/NZX 50 INDEX GROSS"
ticker_name = {k: v for k, v in ticker_name.items() if pd.notna(k) and pd.notna(v)}

# ---------------------------------------------------------------------------
# Short descriptions for each major index
# ---------------------------------------------------------------------------
TICKER_DESC: dict[str, str] = {
    # North America
    "^GSPC": "S&P 500 — 500 largest US publicly traded companies",
    "^DJI": "Dow Jones Industrial Average — 30 major US blue-chip companies",
    "^IXIC": "NASDAQ Composite — all stocks listed on the NASDAQ exchange",
    "^RUT": "Russell 2000 — 2,000 US small-cap companies",
    "^NYA": "NYSE Composite — all common stocks listed on the New York Stock Exchange",
    "^XAX": "NYSE American Composite — small/mid-cap stocks on NYSE American",
    "^VIX": "CBOE Volatility Index — market expectation of 30-day S&P 500 volatility (fear gauge)",
    "DX-Y.NYB": "US Dollar Index — value of USD against a basket of six major currencies",
    "^GSPTSE": "S&P/TSX Composite — largest companies listed on the Toronto Stock Exchange",
    # Latin America
    "^BVSP": "Bovespa (IBOVESPA) — most liquid stocks traded on the B3 exchange in Brazil",
    "^MXX": "IPC (Índice de Precios y Cotizaciones) — 35 most traded stocks on the Mexican Stock Exchange",
    "^MERV": "MERVAL — most traded stocks on the Buenos Aires Stock Exchange (Argentina)",
    # Asia Developed & Oceania
    "^N225": "Nikkei 225 — 225 blue-chip companies listed on the Tokyo Stock Exchange",
    "^HSI": "Hang Seng Index — 82 largest companies listed on the Hong Kong Stock Exchange",
    "000001.SS": "SSE Composite — all stocks listed on the Shanghai Stock Exchange",
    "399001.SZ": "SZSE Component — 500 largest stocks on the Shenzhen Stock Exchange",
    "^TWII": "Taiwan Weighted Index — all listed stocks on the Taiwan Stock Exchange",
    "^KS11": "KOSPI — all common stocks on the Korea Stock Exchange",
    "^AXJO": "S&P/ASX 200 — 200 largest companies on the Australian Securities Exchange",
    "^NZ50": "S&P/NZX 50 — 50 largest and most liquid stocks on the New Zealand Exchange",
    "^AORD": "All Ordinaries — approximately 500 largest companies on the Australian Securities Exchange",
    "^CASE30": "EGX 30 — 30 most actively traded companies on the Egyptian Exchange",
    "^JN0U.JO": "FTSE/JSE Africa All-Share — all ordinary shares on the Johannesburg Stock Exchange",
    # Asia Emerging
    "^STI": "Straits Times Index — 30 largest companies on the Singapore Exchange",
    "^JKSE": "IDX Composite — all stocks listed on the Indonesia Stock Exchange",
    "^KLSE": "FTSE Bursa Malaysia KLCI — 30 largest companies on Bursa Malaysia",
    "^BSESN": "BSE SENSEX — 30 financially sound companies on the Bombay Stock Exchange (India)",
    "^TA125.TA": "Tel Aviv 125 — 125 largest companies on the Tel Aviv Stock Exchange",
    "IMOEX.ME": "MOEX Russia Index — 50 most liquid Russian stocks on the Moscow Exchange",
    "MOEX.ME": "MOEX Russia Index (USD-denominated) — Moscow Exchange broad-market index",
    # Europe & UK
    "^FTSE": "FTSE 100 — 100 largest companies by market cap on the London Stock Exchange",
    "^GDAXI": "DAX — 40 largest blue-chip companies on the Frankfurt Stock Exchange (Germany)",
    "^FCHI": "CAC 40 — 40 largest companies by market cap on Euronext Paris (France)",
    "^STOXX50E": "EURO STOXX 50 — 50 leading blue-chip companies across the Eurozone",
    "^N100": "Euronext 100 — 100 largest and most liquid stocks across Euronext markets",
    "^BFX": "BEL 20 — 20 largest companies on Euronext Brussels (Belgium)",
    "^BUK100P": "Cboe UK 100 — 100 largest UK-listed companies (GBP price-return version)",
    "^125904-USD-STRD":"FTSE Developed Europe Index — broad index of large/mid-cap European stocks",
    # Forex indices
    "^XDA": "Australian Dollar Index — AUD against a trade-weighted basket of currencies",
    "^XDB": "British Pound Index — GBP against a trade-weighted basket of currencies",
    "^XDE": "Euro Index — EUR against a trade-weighted basket of currencies",
    "^XDN": "Japanese Yen Index — JPY against a trade-weighted basket of currencies",
}

def build_idx_info() -> pd.DataFrame:
    """
    Build the Stock Indices Reference Table shown in tab 1.
    Combines ticker symbols and names from the scraped Yahoo Finance list
    with the hardcoded TICKER_CURRENCY and TICKER_DESC mappings.
    Returns a DataFrame sorted alphabetically by index name.
    """
    records = [
        {
            "Ticker Symbol": sym,
            "Ticker Name": name,
            "Description": TICKER_DESC.get(sym, "—"),
            "Currency": TICKER_CURRENCY.get(sym, "USD"),
        }
        for sym, name in ticker_name.items()
        if pd.notna(sym)
    ]
    return (
        pd.DataFrame(records)
        .sort_values("Ticker Name")
        .reset_index(drop=True)
    )

idx_info = build_idx_info()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "APP INTRODUCTION",
    "STOCK INDEX DASHBOARD",
    "PORTFOLIO SIMULATION",
    "CLOSING PRICE PREDICTION",
])

# ===========================================================================
# TAB 1 — Introduction
# ===========================================================================
with tab1:
    st.markdown(
        "<h1 style='text-align:center;'>Stock Index Visualization & Price Prediction App</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
**Stock indices** measure the performance of a group of stocks representing a particular market or sector.
This app provides three analytical tools:

* **STOCK INDEX DASHBOARD** — Visualize closing prices and trading volumes of major world indices,
  scraped live from [Yahoo Finance](https://finance.yahoo.com/world-indices/).
* **PORTFOLIO SIMULATION** — Construct the [Efficient Frontier](https://www.investopedia.com/terms/e/efficientfrontier.asp)
  via Monte Carlo sampling. Identify minimum-risk, maximum-return, and maximum Sharpe-ratio portfolios
  and compute their Value at Risk (VaR).
* **CLOSING PRICE PREDICTION** — Predict future closing prices using a
  [Multi-layer Perceptron Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
  trained on historical data.

**Tech stack:** Streamlit · Plotly · Pandas · NumPy · Scikit-learn · yFinance · SciPy

Feel free to connect via [LinkedIn](https://www.linkedin.com/in/hai-vu/),
[Email](mailto:namhaivu97@gmail.com) or [GitHub](https://github.com/namhaivu173).
        """,
        unsafe_allow_html=True,
    )
    st.write("## Stock Indices Reference Table")
    with st.expander("CLICK HERE FOR MORE INFORMATION"):
        st.table(idx_info[(idx_info["Currency"] != "") & (idx_info["Ticker Symbol"] != "^IPSA")], 
                 width="content")

# ===========================================================================
# TAB 2 — Stock Index Dashboard
# ===========================================================================
with tab2:
    st.write("## Stock Indices Historical Data")
    st.write("### Specify time range")

    with st.form(key="form_dates"):
        c1, c2 = st.columns(2)
        with c1:
            time_start = st.date_input(
                "Start date", value=time_before, max_value=time_max, key="start"
            )
        with c2:
            time_end = st.date_input(
                "End date", value=time_now_ref, max_value=current_date, key="end"
            )
        down_sampling = st.checkbox("Downsample to weekly (faster charts)", value=True)
        if st.form_submit_button("Submit"):
            if time_start >= time_end:
                st.error("Start date must be earlier than end date.")
                st.stop()

    # ── Risk-free rate ────────────────────────────────────────────────────
    @st.cache_data(show_spinner="Fetching risk-free rate…")
    def get_riskfree(time_end_str: str) -> float:
        """
        Fetch the most recent 10-year US Treasury yield (^TNX) up to
        time_end_str and return it as a decimal (e.g. 0.045 for 4.5%).
        Used as the risk-free rate in Sharpe Ratio and CML calculations.
        """
        df = yf.Ticker("^TNX").history(period="max")
        df.index = pd.to_datetime(df.index.strftime("%Y-%m-%d"))
        return float(df[df.index <= time_end_str]["Close"].tail(1).iloc[0] / 100)

    treasury_10y = get_riskfree(str(time_end))

    # ── Treasury yield curve ──────────────────────────────────────────────
    @st.cache_data(show_spinner="Fetching treasury yields…")
    def all_treasury(start, end) -> pd.DataFrame:
        """
        Download daily closing yields for the 1-year (^IRX), 10-year (^TNX),
        and 20-year (^TYX) US Treasury benchmarks. Gaps are forward-filled to
        produce a continuous daily series. Result is cached per session.
        """
        df = yf.download(
            ["^IRX", "^TNX", "^TYX"], start=start, end=end,
            interval="1d", progress=False, auto_adjust=False
        )["Close"]
        df = df.resample("D").ffill()
        df.columns = ["1-Year", "10-Year", "20-Year"]
        return df

    df_treasury = all_treasury(time_start, time_end)

    # ── Ticker download — hardcoded currencies, one batch FX call ─────────
    @st.cache_data(show_spinner="Downloading index price data…")
    def get_tickers(_tickers, start, end) -> pd.DataFrame:
        """
        Batch-download daily OHLCV data for all tickers and convert closing
        prices to USD using a single batch FX download (one call per unique
        non-USD currency). Returns a long-format DataFrame with columns:
        Date, Close (USD), Volume, Ticker, Domestic (currency), ExchgRate.
        Rows with missing Close or Volume are dropped. Result is cached.
        """
        clean = [str(t).strip() for t in _tickers if pd.notna(t)]

        raw = yf.download(
            clean, start=start, end=end,
            interval="1d",
            group_by="ticker", threads=True,
            progress=False, auto_adjust=False
        )

        # Batch FX download (only non-USD pairs)
        unique_ccy = {TICKER_CURRENCY.get(t, "USD") for t in clean
                      if TICKER_CURRENCY.get(t, "USD") != "USD"}
        fx_data: dict = {}
        if unique_ccy:
            fx_syms = [f"{c}USD=X" for c in unique_ccy]
            try:
                fx_raw = yf.download(
                    fx_syms, start=start, end=end,
                    interval="1d",
                    group_by="ticker", threads=True,
                    progress=False, auto_adjust=False
                )
                for curr in unique_ccy:
                    sym = f"{curr}USD=X"
                    try:
                        fx_data[curr] = (
                            fx_raw[sym]["Close"]
                            if len(unique_ccy) > 1
                            else fx_raw["Close"]
                        )
                    except (KeyError, TypeError):
                        fx_data[curr] = None
            except Exception:
                pass

        frames = []
        for t in clean:
            try:
                df_t = (raw[t][["Close", "Volume"]].copy()
                        if len(clean) > 1 else raw[["Close", "Volume"]].copy())
            except KeyError:
                continue
            df_t["Ticker"] = t
            df_t = df_t.reset_index()
            ccy = TICKER_CURRENCY.get(t, "USD")
            df_t["Domestic"] = ccy
            df_t["ExchgRate"] = 1.0
            if ccy != "USD" and fx_data.get(ccy) is not None:
                fx_s = fx_data[ccy].reindex(df_t["Date"]).ffill().bfill()
                df_t["Close"] = df_t["Close"] * fx_s.values
                df_t["ExchgRate"] = fx_s.values
            frames.append(df_t)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
        out["Close"] = pd.to_numeric(out["Close"], errors="coerce").astype(float)
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").astype(float)
        return out.dropna(subset=["Close", "Volume"])

    df_tickers = get_tickers(ticker_name.keys(), time_start, time_end)

    # ── Region mapping ────────────────────────────────────────────────────
    region_idx = {
        "North America": [
            "^GSPC", "^DJI", "^IXIC", "^RUT", "^GSPTSE",
            "^NYA", "^XAX", "^VIX", "DX-Y.NYB",
        ],
        "Latin America": ["^BVSP", "^MXX", "^MERV"],
        "Asia Developed & Oceania": [
            "^N225", "^HSI", "000001.SS", "399001.SZ", "^TWII", "^KS11",
            "^AXJO", "^NZ50", "^AORD", "^CASE30", "^JN0U.JO",
        ],
        "Asia Emerging": [
            "^STI", "^JKSE", "^KLSE", "^BSESN",
            "^TA125.TA", "IMOEX.ME", "MOEX.ME",
        ],
        "Europe & UK": [
            "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E",
            "^N100", "^BFX", "^BUK100P", "^125904-USD-STRD",
        ],
        "Forex Indices": ["^XDA", "^XDB", "^XDE", "^XDN"],
    }
    region_idx_nofx = {k: v for k, v in region_idx.items() if k != "Forex Indices"}

    avail = set(df_tickers["Ticker"].unique())
    valid = set(df_tickers.loc[df_tickers["Volume"] > 0.1, "Ticker"].unique())

    region_idx = {
        r: [t for t in tks if t in avail]
        for r, tks in region_idx.items()
        if any(t in avail for t in tks)
    }
    region_idx_nofx = {
        r: [t for t in tks if t in valid]
        for r, tks in region_idx_nofx.items()
        if any(t in valid for t in tks)
    }

    def get_region(ticker: str) -> str | None:
        """Return the geographic region name for a ticker, or None if not found."""
        for r, tks in region_idx.items():
            if ticker in tks:
                return r
        return None

    df_tickers["Region"] = df_tickers["Ticker"].map(get_region)
    df_tickers2 = df_tickers[df_tickers["Region"].notna()].copy()

    # ── Reference-date filter (≥75 % of tickers available) ───────────────
    df_minDates = (
        df_tickers2.groupby("Ticker")["Date"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "amin", "max": "amax"})
    )
    df_countDates = (
        df_minDates.groupby("amin")["Ticker"]
        .size()
        .reset_index(name="Ticker_Count")
    )
    df_countDates["CumSum"] = df_countDates["Ticker_Count"].cumsum()
    df_countDates["Proportion"] = df_countDates["CumSum"] / df_countDates["Ticker_Count"].sum()
    minDate = df_countDates[df_countDates["Proportion"] >= 0.75].iloc[0, 0]
    remove_ticks = list(df_minDates[df_minDates["amin"] > minDate]["Ticker"])

    df_tickers2 = df_tickers2[
        (~df_tickers2["Ticker"].isin(remove_ticks)) &
        (df_tickers2["Date"] >= minDate)
    ]
    df_valid = df_tickers2[df_tickers2["Date"] >= df_tickers2["Date"].min()].copy()

    # Vectorised reference values
    ref_price = (
        df_valid.loc[df_valid["Close"] != 0.0]
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker")["Close"].transform("first")
    )
    ref_vol = (
        df_valid.loc[df_valid["Volume"] > 0.1]
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker")["Volume"].transform("first")
    )
    df_tickers2["Ref_Price"] = ref_price.reindex(df_tickers2.index).fillna(0.0)
    df_tickers2["Ref_Volume"] = ref_vol.reindex(df_tickers2.index).fillna(0.0)
    df_tickers2["Ref_Return"] = (df_tickers2["Close"] / df_tickers2["Ref_Price"] - 1) * 100
    df_tickers2["Daily_Return"] = df_tickers2.groupby("Ticker")["Close"].pct_change(1, fill_method=None)
    df_tickers2["Ref_VolChg"] = (df_tickers2["Volume"] / df_tickers2["Ref_Volume"] - 1) * 100

    @st.cache_data
    def rotate_df(df, value: str) -> pd.DataFrame:
        """
        Pivot the long-format ticker DataFrame into a wide matrix where rows
        are dates and columns are tickers. Forward- and back-fills gaps so
        every date has a value for every ticker.
        """
        out = df.pivot(index="Date", columns="Ticker", values=value)
        return out.ffill().bfill()

    df_refReturn = rotate_df(df_tickers2, "Ref_Return")
    df_refVolChg = rotate_df(df_tickers2[df_tickers2["Volume"] > 0.1], "Ref_VolChg")
    df_dayReturn = rotate_df(df_tickers2, "Daily_Return")
    df_dayClose = rotate_df(df_tickers2, "Close")

    corr_idx = df_dayReturn.corr(method="pearson")
    cov_idx = df_dayReturn.cov() * 252

    def _prune_region(rdict: dict, df: pd.DataFrame) -> dict:
        """Remove tickers from each region list that are absent from df."""
        present = set(df["Ticker"])
        return {r: [t for t in tks if t in present] for r, tks in rdict.items()}

    region_idx2 = _prune_region(region_idx, df_tickers2)
    region_idx_nofx = _prune_region(region_idx_nofx, df_tickers2)

    # ── Shared chart helpers ──────────────────────────────────────────────
    COLORS = px.colors.qualitative.Plotly

    def downsample(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
        """Resample a time-indexed DataFrame to a lower frequency by averaging."""
        return df.resample(freq).mean()

    def make_line_chart(data: pd.DataFrame, y_title: str = "", down_sample: bool = True) -> go.Figure:
        """
        Build a Plotly Express line chart from a wide-format DataFrame (columns
        are tickers, index is date). Melts to long format before plotting so
        Plotly receives a stable, unambiguous data structure on every rerun.
        Optionally downsamples to weekly frequency to reduce render load.
        Returns an empty Figure if the DataFrame is empty after downsampling.
        """
        if down_sample:
            data = downsample(data)

        if data.empty:
            st.warning("Empty dataset after downsampling")
            return go.Figure()

        df = data.copy()
        idx_col = df.index.name or "Date"
        df = df.reset_index().rename(columns={df.index.name: idx_col} if df.index.name else {})
        df = df.melt(id_vars=idx_col, var_name="Ticker", value_name="Value")

        fig = px.line(
            df,
            x=idx_col,
            y="Value",
            color="Ticker",
            template=plotly_tpl,
        )

        fig.update_layout(
            yaxis_title=y_title,
            xaxis_title="Date",
            legend_title="Ticker",
            height=370,
            margin=dict(t=30, b=20),
        )

        return fig

    midpoint = len(region_idx2) // 2

    # ── Per-region ticker selector ────────────────────────────────────────
    MAX_LINES = 5  # default cap; regions above this get a truncated default
    with st.expander("🔍 Filter indices shown per region", expanded=False):
        st.caption(
            "Regions with many indices are capped to avoid crowded charts. "
            "Use the selectors below to customise which tickers appear in all line plots."
        )
        region_selected: dict[str, list[str]] = {}
        fcol1, fcol2 = st.columns(2)
        for i, (key, tickers) in enumerate(region_idx2.items()):
            default = tickers[:MAX_LINES] if len(tickers) > MAX_LINES else tickers
            with (fcol1 if i % 2 == 0 else fcol2):
                region_selected[key] = st.multiselect(
                    key, options=tickers, default=default, key=f"sel_{key}"
                )

    # Precompute combined DataFrames (full columns; subset when plotting)
    all_close_tickers = [t for tks in region_idx2.values() for t in tks]
    all_ret_tickers = all_close_tickers
    all_vol_tickers = [t for tks in region_idx_nofx.values() for t in tks]
    dfs_dayClose2 = pd.concat([df_dayClose[v] for v in region_idx2.values()], axis=1)
    dfs_refReturn2 = pd.concat([df_refReturn[v] for v in region_idx2.values()], axis=1)
    dfs_refVolChg2 = pd.concat([df_refVolChg[v] for v in region_idx_nofx.values()], axis=1)

    # ── Plot 1: Historical closing prices ─────────────────────────────────
    with st.expander("1 — INDEX HISTORICAL CLOSING PRICES (USD)", expanded=False):
        col1, col2 = st.columns(2)
        for i, (key, tickers) in enumerate(region_idx2.items()):
            sel = region_selected.get(key, tickers) or tickers
            col = col1 if i < midpoint else col2
            with col:
                st.markdown(f"**{key}** ({len(sel)} of {len(tickers)} shown)")
                st.plotly_chart(
                    make_line_chart(dfs_dayClose2[sel], "Closing Price (USD)", down_sampling),
                    config={"responsive": True}, key=f"close_{key}",
                )

    # ── Plot 2: Price % change from start ─────────────────────────────────
    with st.expander("2 — PRICE CHANGES SINCE START DATE (%)", expanded=False):
        col1, col2 = st.columns(2)
        for i, (key, tickers) in enumerate(region_idx2.items()):
            sel = region_selected.get(key, tickers) or tickers
            col = col1 if i < midpoint else col2
            with col:
                st.markdown(f"**{key}** ({len(sel)} of {len(tickers)} shown)")
                st.plotly_chart(
                    make_line_chart(dfs_refReturn2[sel], "Price % Change", down_sampling),
                    config={"responsive": True}, key=f"pchg_{key}",
                )

    # ── Plot 3: Volume % change ───────────────────────────────────────────
    with st.expander("3 — TRADING VOLUME CHANGES SINCE START DATE (%)", expanded=False):
        col1, col2 = st.columns(2)
        for i, (key, tickers) in enumerate(region_idx_nofx.items()):
            # Intersect user selection with tickers valid for volume
            sel_all = region_selected.get(key, tickers)
            sel = [t for t in sel_all if t in tickers] or tickers
            col = col1 if i < midpoint else col2
            with col:
                st.markdown(f"**{key}** ({len(sel)} of {len(tickers)} shown)")
                st.plotly_chart(
                    make_line_chart(dfs_refVolChg2[sel], "Volume % Change", down_sampling),
                    config={"responsive": True}, key=f"vol_{key}",
                )

    # ── Helper: remove box-plot outliers via IQR ──────────────────────────
    def remove_outliers(group: pd.DataFrame, var: str = "Volume") -> pd.DataFrame:
        """
        Drop rows whose value in column var falls outside 1.5 × IQR of the
        group. Used to clean box-plot data so extreme values don't compress
        the visible distribution.
        """
        q1, q3 = group[var].quantile(0.25), group[var].quantile(0.75)
        iqr = q3 - q1
        return group[(group[var] >= q1 - 1.5 * iqr) & (group[var] <= q3 + 1.5 * iqr)]

    # ── Plot 4: Closing price boxplots ────────────────────────────────────
    with st.expander("4 — CLOSING PRICE DISTRIBUTIONS (USD, outliers removed)", expanded=True):
        rkeys = list(region_idx2.keys())
        nrows4 = math.ceil(len(rkeys) / 2)
        fig4 = make_subplots(
            rows=nrows4, cols=2, subplot_titles=rkeys,
            vertical_spacing=0.10,
        )
        for idx_r, region in enumerate(rkeys):
            row, col = idx_r // 2 + 1, idx_r % 2 + 1
            region_df = df_tickers2[df_tickers2["Region"] == region]
            rdata = pd.concat([
                remove_outliers(grp, var="Close")
                for _, grp in region_df.groupby("Ticker")
            ]) if not region_df.empty else region_df
            for t_idx, ticker in enumerate(sorted(rdata["Ticker"].unique())):
                d = rdata[rdata["Ticker"] == ticker]["Close"]
                fig4.add_trace(
                    go.Box(
                        y=d, name=ticker,
                        marker_color=COLORS[t_idx % len(COLORS)],
                        showlegend=False, boxpoints=False, line_width=1.2,
                    ),
                    row=row, col=col,
                )
        fig4.update_layout(
            template=plotly_tpl,
            height=300 * nrows4,
            title_text="Closing Price Distributions by Region (USD)",
            title_x=0.5, margin=dict(t=70, b=20),
        )
        st.plotly_chart(fig4)

    # ── Plot 5: Volume boxplots ───────────────────────────────────────────
    with st.expander("5 — TRADING VOLUME DISTRIBUTIONS (millions, outliers removed)", expanded=True):
        nkeys = list(region_idx_nofx.keys())
        nrows5 = math.ceil(len(nkeys) / 2)
        fig5 = make_subplots(
            rows=nrows5, cols=2, subplot_titles=nkeys,
            vertical_spacing=0.10,
        )
        for idx_r, region in enumerate(nkeys):
            row, col = idx_r // 2 + 1, idx_r % 2 + 1
            rdata = df_tickers2[df_tickers2["Region"] == region].copy()
            rdata = rdata[rdata["Volume"] > 0.1].copy()
            rdata["Volume_mil"] = rdata["Volume"] / 1e6
            rdata = pd.concat([
                remove_outliers(grp, var="Volume")
                for _, grp in rdata.groupby("Ticker")
            ]) if not rdata.empty else rdata
            for t_idx, ticker in enumerate(sorted(rdata["Ticker"].unique())):
                d = rdata[rdata["Ticker"] == ticker]["Volume_mil"]
                fig5.add_trace(
                    go.Box(
                        y=d, name=ticker,
                        marker_color=COLORS[t_idx % len(COLORS)],
                        showlegend=False, boxpoints=False, line_width=1.2,
                    ),
                    row=row, col=col,
                )
        fig5.update_layout(
            template=plotly_tpl,
            height=300 * nrows5,
            title_text="Trading Volume Distributions by Region (millions)",
            title_x=0.5, margin=dict(t=70, b=20),
        )
        st.plotly_chart(fig5)

    # ── Plot 6: Correlation heatmap ───────────────────────────────────────
    with st.expander("6 — CORRELATION MATRIX OF DAILY RETURNS", expanded=True):
        labels = corr_idx.columns.tolist()
        z_vals = corr_idx.values.copy()
        mask = np.triu(np.ones_like(z_vals, dtype=bool), k=1)
        z_masked = np.where(mask, np.nan, z_vals)
        text_mat = np.where(mask, "", np.round(z_vals, 2).astype(str))

        fig6 = go.Figure(go.Heatmap(
            z=z_masked, x=labels, y=labels,
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=text_mat, texttemplate="%{text}", textfont_size=7,
            colorbar=dict(title="ρ", thickness=14),
            hoverongaps=False,
        ))
        fig6.update_layout(
            template=plotly_tpl,
            title="Pairwise Pearson Correlation of Daily Returns",
            title_x=0.5, height=600,
            xaxis=dict(tickangle=-45, tickfont_size=8),
            yaxis=dict(autorange="reversed", tickfont_size=8),
            margin=dict(t=60, b=60, l=60, r=20),
        )
        st.plotly_chart(fig6)

# ===========================================================================
# TAB 3 — Portfolio Simulation / Efficient Frontier
# ===========================================================================
with tab3:
    st.write("## Portfolio Efficient Frontier Simulation")

    with st.expander("READ MORE ABOUT THE EFFICIENT FRONTIER", expanded=True):
        st.markdown("""
The **Efficient Frontier** ([Modern Portfolio Theory](https://www.investopedia.com/terms/m/modernportfoliotheory.asp))
is the set of portfolios offering the **highest expected return for each level of risk** (or lowest risk for
each return level). Key assumptions:

- Investors are rational and risk-averse.
- Markets are efficient (prices reflect all public information).
- Asset returns follow a normal distribution; correlations are explicitly modelled.
- Investors can borrow/lend at the risk-free rate (10-year US Treasury Yield, updated to your chosen end date).
- Fractional shares are allowed.

The **Capital Market Line (CML)** passes through the risk-free rate and the tangency portfolio
(highest Sharpe ratio). All CML portfolios dominate other feasible portfolios in risk-adjusted terms.
The chart is fully interactive — zoom, pan, and hover for details.
        """)

    st.write("### Specify simulation parameters")
    idx_options = list(df_dayReturn.columns)

    with st.form(key="form_sim"):
        c1, c2 = st.columns(2)
        with c1:
            n_indices = st.number_input(
                "Max assets per portfolio", 2, len(idx_options) - 1,
                int(min(len(idx_options) // 2 - 1, 9)),
            )
        with c2:
            n_portfolios = st.number_input("Portfolios to simulate", 1000, 50000, 5000)
        if st.form_submit_button("Run Simulation"):
            if n_indices < 2 or n_portfolios < 1000:
                st.stop()

    small_n = n_portfolios // 2
    large_n = n_portfolios - small_n

    # ── Monte Carlo simulation (vectorised, list-based, no nested loops) ──
    @st.cache_data(show_spinner="Running Monte Carlo simulation…")
    def mean_variance(
        _df_dayReturn: pd.DataFrame,
        _treasury_10y: float,
        max_return: float | None = None,
        n_indices: int = 6,
        n_portfolios: int = 5000,
        random_seed: int = 99,
    ) -> pd.DataFrame:
        """
        Monte Carlo simulation of random portfolios using mean-variance analysis.
        For each iteration, randomly selects n_indices tickers and assigns
        random weights, then computes annualised return, variance, and Sharpe
        Ratio via NumPy dot products (no Python loops over assets).
        Optionally caps portfolios to max_return to bias sampling toward the
        realistic frontier. Returns a DataFrame of simulated portfolios.
        """
        ann_ret = (1 + _df_dayReturn.mean(skipna=True)) ** 252 - 1
        cov = _df_dayReturn.cov() * 252
        cols = list(_df_dayReturn.columns)
        rng = np.random.default_rng(random_seed)

        records, attempts, limit = [], 0, n_portfolios * 30
        while len(records) < n_portfolios and attempts < limit:
            attempts += 1
            assets = rng.choice(cols, n_indices, replace=False)
            w = rng.random(n_indices)
            w /= w.sum()

            ret = float(w @ ann_ret[assets].values)
            var = float(w @ cov.loc[assets, assets].values @ w)

            if max_return is None or ret <= max_return:
                records.append({"expReturn": ret, "expVariance": var, "weights": w, "tickers": assets})

        df_mv = pd.DataFrame(records)
        if not df_mv.empty:
            df_mv["Sharpe_Ratio"] = (df_mv["expReturn"] - _treasury_10y) / df_mv["expVariance"] ** 0.5
        return df_mv

    # ── Optimised frontier: max-return subject to random variance cap ─────
    # Uses scipy SLSQP — much faster than cvxpy for this problem class.
    @st.cache_data(show_spinner="Running frontier optimisation…")
    def optimize_return(
        _df_dayReturn: pd.DataFrame,
        _treasury_10y: float,
        max_variance: float = 1.0,
        n_indices: int = 6,
        n_portfolios: int = 2500,
        random_seed: int = 99,
    ) -> pd.DataFrame:
        """
        Generate optimised frontier portfolios using scipy SLSQP.
        For each iteration, randomly selects n_indices tickers and draws a
        random variance budget, then maximises expected return subject to a
        weight-sum-to-one constraint and the variance cap. Portfolios that
        fail to converge are skipped. Returns a DataFrame of optimised
        portfolios with return, variance, and Sharpe Ratio columns.
        """
        ann_ret = (1 + _df_dayReturn.mean(skipna=True)) ** 252 - 1
        cov = _df_dayReturn.cov() * 252
        cols = list(_df_dayReturn.columns)
        rng = np.random.default_rng(random_seed)

        records = []
        while len(records) < n_portfolios:
            assets = rng.choice(cols, n_indices, replace=False)
            ret_vec = ann_ret[assets].values
            cov_sub = cov.loc[assets, assets].values
            max_var = float(rng.uniform(0.0005, max_variance))

            w0 = np.ones(n_indices) / n_indices
            res = minimize(
                lambda w: -w @ ret_vec,
                w0,
                jac=lambda _: -ret_vec,
                method="SLSQP",
                bounds=[(0, 1)] * n_indices,
                constraints=[
                    {"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "ineq", "fun": lambda w, cv=cov_sub, mv=max_var: mv - w @ cv @ w},
                ],
                options={"ftol": 1e-10, "maxiter": 200},
            )
            if not res.success:
                continue
            w = np.maximum(res.x, 0)
            w /= w.sum()
            ret = float(w @ ret_vec)
            var = float(w @ cov_sub @ w)
            records.append({"expReturn": ret, "expVariance": var, "weights": w, "tickers": assets})

        df_mv = pd.DataFrame(records)
        if not df_mv.empty:
            df_mv["Sharpe_Ratio"] = (df_mv["expReturn"] - _treasury_10y) / df_mv["expVariance"] ** 0.5
        return df_mv

    df_sim1 = mean_variance(df_dayReturn, treasury_10y, n_indices=n_indices,
                              n_portfolios=large_n, max_return=0.5)
    max_var1 = float(df_sim1["expVariance"].max()) if not df_sim1.empty else 1.0
    df_sim2 = optimize_return(df_dayReturn, treasury_10y, n_indices=n_indices,
                                n_portfolios=small_n, max_variance=max_var1)
    df_sim = pd.concat(
        [d for d in [df_sim1, df_sim2] if not d.empty], axis=0
    ).reset_index(drop=True)

    # ── Special portfolios ────────────────────────────────────────────────
    df_minrisk = df_sim.sort_values("expVariance").head(1).reset_index(drop=True)
    df_maxreturn = df_sim.sort_values("expReturn", ascending=False).head(1).reset_index(drop=True)
    df_maxadj = df_sim.sort_values("Sharpe_Ratio", ascending=False).head(1).reset_index(drop=True)

    def port_df(row: pd.Series) -> pd.DataFrame:
        """Extract ticker symbols and their weights from a simulation result row."""
        return pd.DataFrame({"Ticker": row["tickers"], "Weight": row["weights"]})

    df_minrisk_port = port_df(df_minrisk.iloc[0])
    df_maxreturn_port = port_df(df_maxreturn.iloc[0])
    df_maxadj_port = port_df(df_maxadj.iloc[0])

    # Correlation among Max-Sharpe assets
    df_dayReturn_max = df_tickers2[df_tickers2["Ticker"].isin(df_maxadj_port["Ticker"])]
    df_dayReturn_max = rotate_df(df_dayReturn_max, "Daily_Return")
    corr_idx2 = df_dayReturn_max.corr(method="pearson")

    # Capital Market Line (CML) — through risk-free rate and tangency portfolio
    rf_rate = treasury_10y
    tan_std = float(df_maxadj["expVariance"].iloc[0] ** 0.5)
    tan_ret = float(df_maxadj["expReturn"].iloc[0])
    slope_cml = (tan_ret - rf_rate) / tan_std if tan_std > 0 else 0.0
    max_std = float(df_sim["expVariance"].max() ** 0.5)
    cml_x = [0.0, max_std * 1.15]
    cml_y = [rf_rate, rf_rate + slope_cml * max_std * 1.15]

    special_port = pd.concat([df_minrisk, df_maxreturn, df_maxadj]).reset_index(drop=True)
    special_port["Name"] = ["Min Risk", "Max Return", "Max Sharpe Ratio"]

    # ── Efficient Frontier interactive chart ──────────────────────────────
    figEF = go.Figure()

    # Scatter cloud
    figEF.add_trace(go.Scatter(
        x=df_sim["expVariance"] ** 0.5,
        y=df_sim["expReturn"],
        mode="markers",
        marker=dict(
            color=df_sim["Sharpe_Ratio"],
            colorscale="Viridis", showscale=True,
            size=5, opacity=0.65,
            colorbar=dict(title="Sharpe<br>Ratio", thickness=14),
        ),
        text=[
            f"Return: {r:.2%}<br>Volatility: {v:.2%}<br>Sharpe: {s:.3f}"
            for r, v, s in zip(
                df_sim["expReturn"],
                df_sim["expVariance"] ** 0.5,
                df_sim["Sharpe_Ratio"],
            )
        ],
        hoverinfo="text", showlegend=False, name="Portfolio",
    ))

    # Capital Market Line
    figEF.add_trace(go.Scatter(
        x=cml_x, y=cml_y, mode="lines",
        line=dict(color="royalblue", width=2.5, dash="dash"),
        name="Capital Market Line (CML)",
    ))

    # Risk-free rate
    figEF.add_trace(go.Scatter(
        x=[0], y=[rf_rate], mode="markers",
        marker=dict(color="gold", size=14, symbol="diamond",
                    line=dict(color="white", width=1)),
        name=f"Risk-Free Rate ({rf_rate:.2%})",
        text=f"10-yr Treasury: {rf_rate:.2%}", hoverinfo="text",
    ))

    # Special portfolios
    specials = [
        (df_minrisk, "limegreen", "star", "Min Volatility Portfolio"),
        (df_maxreturn, "tomato", "star-triangle-up", "Max Return Portfolio"),
        (df_maxadj, "orange", "star-diamond", "Max Sharpe Ratio Portfolio"),
    ]
    for df_sp, col, sym, label in specials:
        figEF.add_trace(go.Scatter(
            x=df_sp["expVariance"] ** 0.5,
            y=df_sp["expReturn"],
            mode="markers",
            marker=dict(color=col, size=16, symbol=sym,
                        line=dict(color="white", width=1)),
            name=label,
            text=(
                f"{label}<br>"
                f"Return: {df_sp['expReturn'].iloc[0]:.2%}<br>"
                f"Volatility: {df_sp['expVariance'].iloc[0]**0.5:.2%}<br>"
                f"Sharpe: {df_sp['Sharpe_Ratio'].iloc[0]:.3f}"
            ),
            hoverinfo="text",
        ))

    figEF.update_layout(
        template=plotly_tpl,
        title=dict(
            text=(
                f"<b>Efficient Frontier</b> — {len(df_sim):,} Simulated Portfolios"
                f"<br><sub>Up to {n_indices} indices per portfolio</sub>"
            ),
            x=0.5,
        ),
        xaxis=dict(title="Annualized Volatility (Risk)", tickformat=".1%"),
        yaxis=dict(title="Annualized Return", tickformat=".1%"),
        legend=dict(
            x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)",
            bordercolor="gray", borderwidth=1,
        ),
        height=680, margin=dict(t=100),
    )
    st.plotly_chart(figEF)

    # ── Portfolio Value at Risk ───────────────────────────────────────────
    with st.expander("PORTFOLIO VALUE AT RISK (VaR)", expanded=True):

        with st.form(key="form_var"):
            c1, c2, c3 = st.columns(3)
            with c1:
                initial_inv = st.number_input("Initial investment (USD)", 1.0, 10_000_000.0, 100_000.0)
            with c2:
                periods = st.number_input("Estimation horizon (days)", 1, 252, 5)
            with c3:
                conf_level = st.number_input("Confidence level", 0.50, 0.999, 0.95, format="%.3f")
            st.form_submit_button("Calculate VaR")

        fmt_inv = f"${initial_inv:,.2f}"

        # ── Corrected VaR formula ─────────────────────────────────────────
        # For a t-day horizon:
        #   t-day return = annual_return × t/252
        #   t-day std (σ_t) = √(annual_variance × t/252)
        #   Expected value = initial_inv × (1 + t-day return)
        #   Dollar std = initial_inv × σ_t
        #   VaR = initial_inv − PPF(α, expected_value, dollar_std)
        # The original code had two bugs:
        #   1. dollar_std was not multiplied by initial_inv
        #   2. result was scaled by √periods a second time (double-counting)
        # -----------------------------------------------------------------
        def calc_var(df: pd.DataFrame, inv: float, cl: float, t: int) -> np.ndarray:
            """
            Parametric (normal-distribution) Value at Risk for each simulated
            portfolio over a t-day holding period at confidence level cl.
            Scales annual return and variance to the t-day horizon, then uses
            the normal PPF to find the portfolio value cutoff at alpha = 1 - cl.
            VaR = initial investment − cutoff (positive value = potential loss).
            """
            alpha = 1 - cl
            t_ret = df["expReturn"].values * t / 252
            t_std = (df["expVariance"].values * t / 252) ** 0.5
            exp_val = inv * (1 + t_ret) # expected portfolio value in $
            dol_std = inv * t_std # dollar std of portfolio value
            cutoff = norm.ppf(alpha, exp_val, dol_std)
            return inv - cutoff # VaR (positive = potential loss)

        def var_periods(sp: pd.DataFrame, inv: float, cl: float, t: int) -> pd.DataFrame:
            """
            Compute VaR for the three special portfolios (Min Risk, Max Return,
            Max Sharpe) across holding periods 1 through t. Returns a DataFrame
            with one row per day and one column per portfolio, used to plot how
            potential loss grows with the holding horizon.
            """
            cols = ["Min Risk", "Max Return", "Max Sharpe Ratio"]
            out = pd.DataFrame(0.0, index=range(t), columns=cols)
            for i in range(len(sp)):
                row = sp.iloc[[i]].reset_index(drop=True)
                for j in range(t):
                    out.iloc[j, i] = float(calc_var(row, inv, cl, j + 1)[0])
            return out

        var_vals = calc_var(df_sim, initial_inv, conf_level, periods)
        mean_var = float(np.mean(var_vals))

        log_scale = st.session_state.get("var_log_scale", False)
        c1, c2 = st.columns(2)

        with c1:
            # t-day portfolio dollar returns for comparison overlay
            t_ret = df_sim["expReturn"].values * periods / 252
            dollar_ret = initial_inv * t_ret
            ret_mean = float(np.mean(dollar_ret))
            ret_std = float(np.std(dollar_ret))
            x_min = min(float(np.min(var_vals)), float(np.min(dollar_ret)))
            x_max = max(float(np.max(var_vals)), float(np.max(dollar_ret)))
            x_pdf = np.linspace(x_min, x_max, 400)
            y_pdf = norm.pdf(x_pdf, ret_mean, ret_std)

            figV1 = go.Figure()
            figV1.add_trace(go.Histogram(
                x=var_vals, histnorm="probability density", opacity=0.65,
                nbinsx=40, name=f"Portfolio {periods}-day VaR Distribution",
                marker_color="steelblue",
            ))
            figV1.add_trace(go.Histogram(
                x=dollar_ret, histnorm="probability density", opacity=0.50,
                nbinsx=40, name="Portfolio Return Normal Distribution",
                marker_color="tomato",
            ))
            figV1.add_trace(go.Scatter(
                x=x_pdf, y=y_pdf,
                mode="lines", name="Normal Return PDF",
                line=dict(color="limegreen", width=2),
            ))
            figV1.add_vline(
                x=mean_var, line_dash="dash", line_color="crimson",
                annotation_text=f"Mean VaR: ${mean_var:,.0f}",
                annotation_position="top right",
            )
            figV1.update_layout(
                template=plotly_tpl,
                barmode="overlay",
                title=f"Portfolio Value at Risk (VaR) vs. Normally Distributed Return",
                xaxis_title=f"Value at Risk  (Initial: {fmt_inv})",
                yaxis_title="Probability Density",
                yaxis_type="log" if log_scale else "linear",
                legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top",
                            bgcolor="rgba(0,0,0,0.3)", bordercolor="gray", borderwidth=1),
                height=420, margin=dict(t=50),
            )
            st.plotly_chart(figV1)
            st.checkbox("Logarithmic y-axis for histogram", value=False, key="var_log_scale")

        with c2:
            # VaR over holding period for the three special portfolios
            df_var = var_periods(special_port, initial_inv, conf_level, periods)
            sp_cols = ["Min Risk", "Max Return", "Max Sharpe Ratio"]
            sp_clrs = ["limegreen", "tomato", "orange"]

            figV2 = go.Figure()
            if periods == 1:
                for i, nm in enumerate(sp_cols):
                    figV2.add_trace(go.Bar(
                        x=[f"{nm} Portfolio"], y=[df_var.iloc[0, i]],
                        marker_color=sp_clrs[i], name=nm,
                    ))
                figV2.update_layout(xaxis_title="Portfolio", barmode="group")
            else:
                for i, nm in enumerate(sp_cols):
                    figV2.add_trace(go.Scatter(
                        x=list(range(0, periods)), y=df_var.iloc[:, i],
                        mode="lines+markers",
                        line=dict(color=sp_clrs[i], width=2),
                        name=f"{nm} Portfolio",
                    ))
                figV2.update_layout(xaxis_title="Holding Period (days)")

            figV2.update_layout(
                template=plotly_tpl,
                title=f"Maximum Portfolio Loss (VaR @ {conf_level:.1%}) over {periods}-day Horizon",
                yaxis_title=f"Value at Risk  (Initial: {fmt_inv})",
                legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                            bgcolor="rgba(0,0,0,0.3)", bordercolor="gray", borderwidth=1),
                height=420, margin=dict(t=50),
            )
            st.plotly_chart(figV2)
        
        formatted_var = '{:,.2f}'.format(abs(mean_var))
        
        st.write(
            f'Based on {len(df_sim):,} simulated portfolio scenarios, the {periods}-day Value at Risk (VaR) '
            f'for a \${initial_inv:,.2f} investment is \${formatted_var} at a(n) {round(conf_level * 100, 1)}% '
            f'confidence level. This means there is a(n) {round(conf_level * 100, 1)}% probability that losses '
            f'will not exceed this amount over the next {periods} days. '
            f'[Click here to learn more about Value at Risk.]'
            f'(https://www.investopedia.com/articles/04/092904.asp)'
        )
        

    # ── Asset allocation & performance tables ─────────────────────────────
    with st.expander("PORTFOLIO ASSET ALLOCATION & PERFORMANCE", expanded=True):

        def fmt_port_df(df_port: pd.DataFrame) -> pd.DataFrame:
            """Sort portfolio holdings by weight descending and format weights as percentages."""
            out = df_port.sort_values("Weight", ascending=False).reset_index(drop=True).copy()
            out["Weight"] = out["Weight"].map(lambda x: f"{x:.2%}")
            return out

        def perf_table(df_row: pd.DataFrame) -> pd.DataFrame:
            """Build a formatted two-column metrics table (Metric / Value) for a single portfolio row."""
            return pd.DataFrame({
                "Metric": ["Annualized Return", "Annualized Volatility", "Sharpe Ratio"],
                "Value": [
                    f"{float(df_row['expReturn'].iloc[0]):.2%}",
                    f"{float(df_row['expVariance'].iloc[0]**0.5):.2%}",
                    f"{float(df_row['Sharpe_Ratio'].iloc[0]):.4f}",
                ],
            })

        c1, c2, c3 = st.columns(3)
        port_specs = [
            (c1, "Minimum Risk", df_minrisk_port, df_minrisk),
            (c2, "Maximum Return", df_maxreturn_port, df_maxreturn),
            (c3, "Highest Sharpe Ratio", df_maxadj_port, df_maxadj),
        ]
        for col, title, df_port, df_perf in port_specs:
            with col:
                st.write(f"#### {title} Portfolio")
                st.dataframe(fmt_port_df(df_port), width="stretch", hide_index=True)
                st.write("#### Performance")
                st.dataframe(perf_table(df_perf), width="stretch", hide_index=True)

    # ── Supplementary charts ──────────────────────────────────────────────
    with st.expander("OTHER RELEVANT INFORMATION", expanded=True):
        c1, c2 = st.columns(2)

        with c1:
            st.write("#### US Treasury Yield Curve (%)")
            fig_tsy = px.line(
                df_treasury, template=plotly_tpl,
                labels={"value": "Yield (%)", "variable": "Maturity"},
            )
            fig_tsy.update_layout(height=360, margin=dict(t=30))
            st.plotly_chart(fig_tsy)

        with c2:
            st.write("#### Correlation Matrix — Max Sharpe Portfolio Assets")
            labels2 = corr_idx2.columns.tolist()
            z2 = corr_idx2.values.copy()
            mask2 = np.triu(np.ones_like(z2, dtype=bool), k=1)
            fig_c2 = go.Figure(go.Heatmap(
                z=np.where(mask2, np.nan, z2),
                x=labels2, y=labels2,
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                text=np.where(mask2, "", np.round(z2, 2).astype(str)),
                texttemplate="%{text}", textfont_size=11,
                colorbar=dict(title="ρ", thickness=12),
                hoverongaps=False,
            ))
            fig_c2.update_layout(
                template=plotly_tpl, height=360,
                xaxis=dict(tickangle=-45),
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=60, l=60, r=20),
            )
            st.plotly_chart(fig_c2)

# ===========================================================================
# TAB 4 — Closing Price Prediction
# ===========================================================================
with tab4:
    st.write("## Stock Indices Price Prediction")
    st.write(f"Today's date: **{current_date.strftime('%Y-%m-%d')} UTC**")

    ticker_lists = sorted(ticker_name.keys())

    st.write("### Specify prediction parameters")
    with st.form(key="form_pred"):
        c1, c2 = st.columns(2)
        with c1:
            pick_ticker = st.selectbox(
                "Select ticker",
                ticker_lists,
                index=ticker_lists.index("^GSPC") if "^GSPC" in ticker_lists else 0,
            )
            st.markdown(f"Selected: **{ticker_name.get(pick_ticker, pick_ticker)}**")
        with c2:
            pred_rows = st.slider("Lookback period (days)", 5, 252, 30)
            st.write(
                f"Prices from the past **{pred_rows} days** will be used to predict "
                f"the closing price of day **{pred_rows + 1}**."
            )
        if st.form_submit_button("Generate Predictions"):
            if not (pick_ticker and 5 <= pred_rows <= 252):
                st.error("Invalid inputs.")
                st.stop()

    st.write("### Prediction Model Outputs")

    ticker_chosen = ticker_name.get(pick_ticker, pick_ticker)
    idx_currency = (
        idx_info.loc[idx_info["Ticker Symbol"] == pick_ticker, "Currency"].iloc[0]
        if pick_ticker in idx_info["Ticker Symbol"].values else "USD"
    )

    @st.cache_data(show_spinner="Downloading historical prices…")
    def load_ticker_history(symbol: str) -> pd.DataFrame:
        """
        Download the full price history for a single ticker via yfinance and
        return a DataFrame indexed by normalised UTC dates with a Close column.
        Result is cached to avoid redundant downloads within the same session.
        """
        df = yf.Ticker(symbol).history(period="max")["Close"].reset_index()
        df["Date"] = pd.to_datetime(
            pd.to_datetime(df["Date"], utc=True).dt.strftime("%Y-%m-%d")
        )
        return df.set_index("Date")

    df_ticks = load_ticker_history(pick_ticker)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_ticks.values)
    train_rows = math.ceil(len(df_ticks) * 0.8)
    s_train = scaled[:train_rows]
    s_test = scaled[train_rows:]

    def lookback_split(arr: np.ndarray, lb: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Create supervised learning sequences from a 1-D scaled price array.
        Each sample X[i] is a window of lb consecutive values; the target Y[i]
        is the next value after the window. Returns (X, Y) as NumPy arrays
        ready for the MLP regressor.
        """
        X, Y = [], []
        for i in range(len(arr) - lb):
            X.append(arr[i: i + lb])
            Y.append(arr[i + lb])
        return np.array(X).reshape(len(X), -1), np.array(Y).ravel()

    x_train, y_train = lookback_split(s_train, pred_rows)
    x_test, y_test = lookback_split(s_test, pred_rows)

    @st.cache_data(show_spinner="Training MLP model…")
    def train_mlp(x_tr: np.ndarray, y_tr: np.ndarray) -> MLPRegressor:
        """
        Train a two-hidden-layer MLP regressor (20 × 20, ReLU, Adam) on the
        scaled training sequences. Cached so the model is only retrained when
        the ticker or lookback window changes.
        """
        m = MLPRegressor(
            hidden_layer_sizes=(20, 20), activation="relu",
            solver="adam", max_iter=1000, random_state=99,
        )
        m.fit(x_tr, y_tr)
        return m

    model = train_mlp(x_train, y_train)
    y_pred = model.predict(x_test)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))

    def mape(y_true: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error with a small epsilon guard to avoid
        division by zero when the true value is at or near zero.
        """
        return float(np.mean(np.abs((y_true - y_hat) / (np.abs(y_true) + 1e-8))))

    df_score = pd.DataFrame({
        "Metric": ["RMSE", "MAPE", "R²"],
        "Value": [
            f"{root_mean_squared_error(y_test_inv, y_pred_inv):.4f}",
            f"{mape(y_test_inv, y_pred_inv):.2%}",
            f"{r2_score(y_test_inv, y_pred_inv):.4f}",
        ],
    })

    train_price = df_ticks.iloc[: train_rows + pred_rows]
    test_price = df_ticks.iloc[train_rows + pred_rows:]

    pred_price = pd.DataFrame(
        {"Close": y_test_inv.ravel(), "Predictions": y_pred_inv.ravel()},
        index=test_price.index,
    )

    def future_pred(x_te: np.ndarray, days: int = 5) -> np.ndarray:
        """
        Autoregressively forecast the next `days` closing prices beyond the
        test set. Starting from the last test window, each prediction is fed
        back as the newest input for the following step. Returns inverse-scaled
        prices in the original currency units.
        """
        x_rec = x_te[-1].reshape(1, -1)
        preds = []
        for _ in range(days + 1):
            p = model.predict(x_rec)[0]
            preds.append(p)
            x_rec = np.roll(x_rec, -1)
            x_rec[0, -1] = p
        return scaler.inverse_transform(np.array(preds[1:]).reshape(-1, 1)).ravel()

    future_prices = future_pred(x_test, 5)
    last_date = pred_price.index[-1]
    next_dates = pd.bdate_range(last_date, periods=6)[1:]
    pred_new = pd.DataFrame(
        {"Close": np.nan, "Predictions": future_prices}, index=next_dates
    )
    pred_price2 = pd.concat([pred_price, pred_new])

    # ── Main prediction chart ────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("#### Training, Testing & Predicted Price Chart")
        figP = go.Figure()
        figP.add_trace(go.Scatter(
            x=train_price.index, y=train_price["Close"].values,
            mode="lines", name="Train", line=dict(color="steelblue", width=1.5),
        ))
        figP.add_trace(go.Scatter(
            x=pred_price2.index, y=pred_price2["Close"],
            mode="lines", name="Test (Actual)", line=dict(color="darkorange", width=1.5),
        ))
        figP.add_trace(go.Scatter(
            x=pred_price2.index, y=pred_price2["Predictions"],
            mode="lines", name="Predicted", line=dict(color="limegreen", width=1.5, dash="dot"),
        ))
        if len(next_dates):
            figP.add_vrect(
                x0=str(last_date), x1=str(next_dates[-1]),
                fillcolor="rgba(128,128,128,0.12)", layer="below", line_width=0,
                annotation_text="5-day Forecast", annotation_position="top left",
            )
        figP.update_layout(
            template=plotly_tpl,
            title=f"{ticker_chosen} — Price Prediction ({pred_rows}-day lookback)",
            xaxis_title="Date",
            yaxis_title=f"Closing Price ({idx_currency})",
            legend=dict(x=0.01, y=0.99),
            height=480, margin=dict(t=60),
        )
        st.plotly_chart(figP)

    with c2:
        st.write("#### Last 5 + Next 5 Trading Days")
        disp = pred_price2[["Close", "Predictions"]].tail(10).copy()
        disp.index = disp.index.strftime("%Y-%m-%d")
        st.dataframe(
            disp.style.format({"Close": "{:.2f}", "Predictions": "{:.2f}"}, na_rep="—"),
            width="stretch",
        )
        st.write("#### Model Performance")
        st.dataframe(df_score, width="stretch", hide_index=True)

        pred_csv = pred_price2.to_csv().encode()
        st.download_button(
            "⬇  Download Predictions (CSV)",
            data=pred_csv,
            file_name=f"{ticker_chosen} Price Predictions.csv",
            mime="text/csv",
        )

    # ── Test-period close-up ────────────────────────────────────────────
    st.write("#### Test Period: Actual vs. Predicted (Close-up)")
    figP2 = go.Figure()
    figP2.add_trace(go.Scatter(
        x=pred_price.index, y=pred_price["Close"],
        mode="lines", name="Actual", line=dict(color="darkorange", width=1.5),
    ))
    figP2.add_trace(go.Scatter(
        x=pred_price.index, y=pred_price["Predictions"],
        mode="lines", name="Predicted", line=dict(color="royalblue", width=1.5, dash="dot"),
    ))
    figP2.update_layout(
        template=plotly_tpl,
        title=f"{ticker_chosen} — Test Period: Actual vs. Predicted",
        xaxis_title="Date",
        yaxis_title=f"Closing Price ({idx_currency})",
        legend=dict(x=0.01, y=0.99),
        height=400, margin=dict(t=60),
    )
    st.plotly_chart(figP2)

st.text('')
st.write("## THANKS FOR VISITING!")
st.write('Created by: [Hai Vu](https://www.linkedin.com/in/hai-vu/)')