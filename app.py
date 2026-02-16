import base64
import io
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import mplfinance as mpf  # noqa: E402
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

logging.basicConfig()
logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"[^A-Za-z0-9.\-^=]")
_HOST_RE = re.compile(r"^[A-Za-z0-9.\-:]+$")

_OLLAMA_DEFAULT_HOST = "localhost"

STRATEGIC_PERIOD_MAP = {
    "1mo": "1y",
    "3mo": "2y",
    "6mo": "2y",
    "1y": "5y",
    "2y": "5y",
}

ANALYSIS_HISTORY_FILE = Path(__file__).parent / "analysis_history.json"

CHART_BASE_HEIGHT = 600
CHART_SUBCHART_HEIGHT = 200
_DATA_TIMEOUT = 30  # seconds per supplementary data future
_CHART_DPI = 200

_CHART_COLORS = {
    "candle_up": "#26a69a",
    "candle_down": "#ef5350",
    "sma20": "#ff9800",
    "sma50": "#e040fb",
    "ema12": "#00e5ff",
    "ema26": "#ff4081",
    "bb": "#78909c",
    "rsi": "#e040fb",
    "rsi_over": "#ef5350",
    "rsi_under": "#26a69a",
    "macd_line": "#42a5f5",
    "macd_signal": "#ff9800",
    "macd_hist_pos": "#26a69a",
    "macd_hist_neg": "#ef5350",
    "atr": "#ffa726",
    "adx": "#ab47bc",
    "plus_di": "#26a69a",
    "minus_di": "#ef5350",
    "adx_thresh": "#78909c",
    "volume_up": "#26a69a",
    "volume_down": "#ef5350",
}

_MPF_MC = mpf.make_marketcolors(
    up=_CHART_COLORS["candle_up"],
    down=_CHART_COLORS["candle_down"],
    edge="inherit",
    wick="inherit",
    volume={"up": _CHART_COLORS["volume_up"], "down": _CHART_COLORS["volume_down"]},
)
_MPF_STYLE = mpf.make_mpf_style(
    marketcolors=_MPF_MC,
    facecolor="#16213e",
    figcolor="#1a1a2e",
    gridcolor="#2a2a4a",
    gridstyle="--",
    gridaxis="both",
    rc={
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "axes.edgecolor": "#2a2a4a",
    },
)


_LEGEND_KWARGS = dict(
    loc="upper left", fontsize=7,
    facecolor="#1a1a2e", edgecolor="#2a2a4a", labelcolor="white",
)

_ALL_INDICATOR_LABELS = [
    "SMA (20, 50)",
    "EMA (12, 26)",
    "Bollinger Bands (20, 2σ)",
    "RSI (14)",
    "MACD (12, 26, 9)",
    "ATR (14)",
    "ADX (14)",
]


@st.cache_data(show_spinner=False, ttl=300)
def fetch_stock_data(symbol: str, period: str) -> pd.DataFrame | None:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
    except Exception:
        logger.warning("Failed to fetch stock data for %s", symbol, exc_info=True)
        return None
    if df.empty:
        return None
    _compute_indicators(df)
    return df


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_ticker_info(symbol: str) -> dict:
    """Fetch ticker info dict from Yahoo Finance (single network call).

    Both ``fetch_company_name`` and ``fetch_fundamentals`` read from this
    shared cache so that ``yf.Ticker(symbol).info`` is only fetched once
    per symbol per TTL window.
    """
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        logger.warning("Failed to fetch ticker info for %s", symbol, exc_info=True)
        return {}


def fetch_company_name(symbol: str) -> str:
    """Look up the long company name for a ticker symbol."""
    info = _fetch_ticker_info(symbol)
    return info.get("longName") or info.get("shortName") or symbol.upper()


def fetch_fundamentals(symbol: str) -> dict | None:
    """Fetch fundamental data for a ticker symbol."""
    info = _fetch_ticker_info(symbol)
    if not info:
        return None
    result = {
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "marketCap": info.get("marketCap"),
        "dividendYield": info.get("dividendYield"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
    }
    if any(v is not None for v in result.values()):
        return result
    return None


@st.cache_data(show_spinner=False, ttl=300)
def fetch_next_earnings(symbol: str) -> str | None:
    """Find the next upcoming earnings date."""
    try:
        dates = yf.Ticker(symbol).get_earnings_dates(limit=8)
        if dates is None or dates.empty:
            return None
        now = pd.Timestamp.now(tz=dates.index.tz)
        future = dates[dates.index > now]
        if future.empty:
            return None
        nearest = future.index.min()
        days = (nearest - now).days
        if days == 0:
            return f"today ({nearest.strftime('%Y-%m-%d')})"
        return f"in {days} days ({nearest.strftime('%Y-%m-%d')})"
    except Exception:
        logger.warning("Failed to fetch earnings dates for %s", symbol, exc_info=True)
        return None


@st.cache_data(show_spinner=False, ttl=300)
def fetch_news_headlines(symbol: str, limit: int = 5) -> list[dict]:
    """Fetch recent news headlines for a ticker symbol."""
    try:
        news = yf.Ticker(symbol).news
        if not news:
            return []
        results = []
        for item in news[:limit]:
            content = item.get("content", {})
            title = content.get("title", "")
            provider = content.get("provider", {})
            publisher = provider.get("displayName", "")
            pub_date = content.get("pubDate", "")
            date_str = ""
            if pub_date:
                try:
                    date_str = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).strftime("%Y-%m-%d")
                except (ValueError, AttributeError):
                    pass
            if title:
                results.append({"title": title, "publisher": publisher, "date": date_str})
        return results
    except Exception:
        logger.warning("Failed to fetch news for %s", symbol, exc_info=True)
        return []


@st.cache_data(show_spinner=False, ttl=300)
def fetch_market_context(period: str) -> dict | None:
    """Fetch S&P 500 market context for the given period."""
    try:
        df = yf.Ticker("^GSPC").history(period=period)
        if df.empty:
            return None
        first_close = df["Close"].iloc[0]
        latest_close = df["Close"].iloc[-1]
        period_return = ((latest_close - first_close) / first_close) * 100 if first_close != 0 else 0.0
        return {
            "latest_close": float(latest_close),
            "period_return": float(period_return),
            "period_high": float(df["High"].max()),
            "period_low": float(df["Low"].min()),
        }
    except Exception:
        logger.warning("Failed to fetch market context", exc_info=True)
        return None


def _compute_indicators(df: pd.DataFrame) -> None:
    """Compute technical indicator columns on the dataframe in-place."""
    close = df["Close"]

    # SMA
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()

    # EMA
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    df["BB_Mid"] = df["SMA_20"]
    bb_std = close.rolling(window=20).std(ddof=0)
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        df["RSI"] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ATR (14-period)
    prev_close = close.shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    # ADX (14-period)
    raw_plus_dm = df["High"].diff()
    raw_minus_dm = -df["Low"].diff()
    plus_dm = raw_plus_dm.where((raw_plus_dm > raw_minus_dm) & (raw_plus_dm > 0), 0.0)
    minus_dm = raw_minus_dm.where((raw_minus_dm > raw_plus_dm) & (raw_minus_dm > 0), 0.0)
    atr14 = df["ATR"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Plus_DI"] = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14
        df["Minus_DI"] = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14
        dx = (abs(df["Plus_DI"] - df["Minus_DI"]) / (df["Plus_DI"] + df["Minus_DI"])) * 100
    df["ADX"] = dx.ewm(span=14, adjust=False).mean()
    # Blank out warm-up period (need ~28 bars: 14 for ATR/DI smoothing + 14 for ADX smoothing)
    warmup = 28
    df.loc[df.index[:warmup], ["ADX", "Plus_DI", "Minus_DI"]] = np.nan


def compute_support_resistance(
    df: pd.DataFrame, window: int = 5, n_levels: int = 3
) -> tuple[list[float], list[float]]:
    """Compute support and resistance levels from local extrema."""
    highs = df["High"]
    lows = df["Low"]

    resistance_pts: list[float] = []
    support_pts: list[float] = []

    for i in range(window, len(df) - window):
        segment_high = highs.iloc[i - window:i + window + 1]
        if highs.iloc[i] >= segment_high.max():
            resistance_pts.append(float(highs.iloc[i]))
        segment_low = lows.iloc[i - window:i + window + 1]
        if lows.iloc[i] <= segment_low.min():
            support_pts.append(float(lows.iloc[i]))

    def cluster_levels(points: list[float]) -> list[float]:
        if not points:
            return []
        sorted_pts = sorted(points)
        clusters: list[list[float]] = [[sorted_pts[0]]]
        for p in sorted_pts[1:]:
            mean_last = sum(clusters[-1]) / len(clusters[-1])
            if mean_last != 0 and abs(p - mean_last) / abs(mean_last) <= 0.01:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        clusters.sort(key=len, reverse=True)
        return sorted(sum(c) / len(c) for c in clusters[:n_levels])

    return cluster_levels(support_pts), cluster_levels(resistance_pts)


def build_candlestick_chart(df: pd.DataFrame, symbol: str,
                            title: str | None = None) -> bytes:
    """Build a candlestick chart and return PNG bytes."""
    price_title = title or f"{symbol.upper()} Price"

    # Determine which indicator panels have valid data
    show_rsi = df["RSI"].notna().any()
    show_macd = df["MACD"].notna().any()
    show_atr = df["ATR"].notna().any()
    show_adx = df["ADX"].notna().any()

    addplots: list = []

    # --- Overlays (panel 0 = price) ---
    addplots.append(mpf.make_addplot(df["SMA_20"], panel=0, color=_CHART_COLORS["sma20"], width=1))
    addplots.append(mpf.make_addplot(df["SMA_50"], panel=0, color=_CHART_COLORS["sma50"], width=1))
    addplots.append(mpf.make_addplot(df["EMA_12"], panel=0, color=_CHART_COLORS["ema12"], width=1, linestyle="dotted"))
    addplots.append(mpf.make_addplot(df["EMA_26"], panel=0, color=_CHART_COLORS["ema26"], width=1, linestyle="dotted"))
    addplots.append(mpf.make_addplot(df["BB_Upper"], panel=0, color=_CHART_COLORS["bb"], width=1, linestyle="dashed"))
    addplots.append(mpf.make_addplot(
        df["BB_Lower"], panel=0, color=_CHART_COLORS["bb"], width=1, linestyle="dashed",
        fill_between={"y1": df["BB_Upper"].values, "alpha": 0.1, "color": _CHART_COLORS["bb"]},
    ))

    # --- Indicator sub-panels (panel 2+ since volume is panel 1) ---
    next_panel = 2

    rsi_panel = None
    if show_rsi:
        rsi_panel = next_panel
        addplots.append(mpf.make_addplot(df["RSI"], panel=rsi_panel, color=_CHART_COLORS["rsi"], width=1, ylabel="RSI (14)"))
        next_panel += 1

    macd_panel = None
    if show_macd:
        macd_panel = next_panel
        addplots.append(mpf.make_addplot(df["MACD"], panel=macd_panel, color=_CHART_COLORS["macd_line"], width=1, ylabel="MACD"))
        addplots.append(mpf.make_addplot(df["MACD_Signal"], panel=macd_panel, color=_CHART_COLORS["macd_signal"], width=1))
        hist_colors = [_CHART_COLORS["macd_hist_pos"] if v >= 0 else _CHART_COLORS["macd_hist_neg"]
                       for v in df["MACD_Hist"].fillna(0)]
        addplots.append(mpf.make_addplot(df["MACD_Hist"], panel=macd_panel, type="bar", color=hist_colors))
        next_panel += 1

    atr_panel = None
    if show_atr:
        atr_panel = next_panel
        addplots.append(mpf.make_addplot(df["ATR"], panel=atr_panel, color=_CHART_COLORS["atr"], width=1, ylabel="ATR (14)"))
        next_panel += 1

    adx_panel = None
    if show_adx:
        adx_panel = next_panel
        addplots.append(mpf.make_addplot(df["ADX"], panel=adx_panel, color=_CHART_COLORS["adx"], width=2, ylabel="ADX (14)"))
        addplots.append(mpf.make_addplot(df["Plus_DI"], panel=adx_panel, color=_CHART_COLORS["plus_di"], width=1, linestyle="dotted"))
        addplots.append(mpf.make_addplot(df["Minus_DI"], panel=adx_panel, color=_CHART_COLORS["minus_di"], width=1, linestyle="dotted"))
        next_panel += 1

    # Panel ratios: price(5), volume(1.5), then 2 per indicator panel
    num_ind_panels = sum([show_rsi, show_macd, show_atr, show_adx])
    panel_ratios = [5, 1.5] + [2] * num_ind_panels

    chart_height_px = CHART_BASE_HEIGHT + num_ind_panels * CHART_SUBCHART_HEIGHT
    fig_height = chart_height_px / 100  # inches at DPI 200

    kwargs: dict = dict(
        type="candle",
        style=_MPF_STYLE,
        title=price_title,
        volume=True,
        volume_panel=1,
        panel_ratios=panel_ratios,
        figsize=(12, fig_height),
        returnfig=True,
        tight_layout=True,
        scale_padding={"top": 1.5, "bottom": 0.8},
    )
    if addplots:
        kwargs["addplot"] = addplots

    fig, axlist = mpf.plot(df, **kwargs)

    # Style the title
    fig.suptitle(price_title, color="white", fontsize=14, y=0.98)
    # Remove the default mplfinance title (set on ax)
    axlist[0].set_title("")

    # Add legend to price panel
    ax_price = axlist[0]
    legend_entries = [
        Line2D([], [], color=_CHART_COLORS["sma20"], linewidth=1, label="SMA 20"),
        Line2D([], [], color=_CHART_COLORS["sma50"], linewidth=1, label="SMA 50"),
        Line2D([], [], color=_CHART_COLORS["ema12"], linewidth=1, linestyle="dotted", label="EMA 12"),
        Line2D([], [], color=_CHART_COLORS["ema26"], linewidth=1, linestyle="dotted", label="EMA 26"),
        Line2D([], [], color=_CHART_COLORS["bb"], linewidth=1, linestyle="dashed", label="BB"),
    ]
    ax_price.legend(handles=legend_entries, **_LEGEND_KWARGS)

    # Add horizontal threshold lines on indicator panels.
    # mplfinance interleaves real axes with hidden spacer axes, so the
    # actual axis for panel N is at index 2*N in axlist.
    if rsi_panel is not None:
        ax_rsi = axlist[2 * rsi_panel]
        ax_rsi.axhline(70, color=_CHART_COLORS["rsi_over"], linewidth=1, linestyle="--")
        ax_rsi.axhline(30, color=_CHART_COLORS["rsi_under"], linewidth=1, linestyle="--")
        ax_rsi.set_ylim(0, 100)
        rsi_legend = [
            Line2D([], [], color=_CHART_COLORS["rsi"], linewidth=1, label="RSI"),
        ]
        ax_rsi.legend(handles=rsi_legend, **_LEGEND_KWARGS)

    if atr_panel is not None:
        ax_atr = axlist[2 * atr_panel]
        atr_legend = [
            Line2D([], [], color=_CHART_COLORS["atr"], linewidth=1, label="ATR"),
        ]
        ax_atr.legend(handles=atr_legend, **_LEGEND_KWARGS)

    if macd_panel is not None:
        ax_macd = axlist[2 * macd_panel]
        macd_legend = [
            Line2D([], [], color=_CHART_COLORS["macd_line"], linewidth=1, label="MACD"),
            Line2D([], [], color=_CHART_COLORS["macd_signal"], linewidth=1, label="Signal"),
        ]
        ax_macd.legend(handles=macd_legend, **_LEGEND_KWARGS)

    if adx_panel is not None:
        ax_adx = axlist[2 * adx_panel]
        ax_adx.axhline(25, color=_CHART_COLORS["adx_thresh"], linewidth=1, linestyle="--")
        adx_legend = [
            Line2D([], [], color=_CHART_COLORS["adx"], linewidth=2, label="ADX"),
            Line2D([], [], color=_CHART_COLORS["plus_di"], linewidth=1, linestyle="dotted", label="+DI"),
            Line2D([], [], color=_CHART_COLORS["minus_di"], linewidth=1, linestyle="dotted", label="-DI"),
        ]
        ax_adx.legend(handles=adx_legend, **_LEGEND_KWARGS)

    # Export to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_CHART_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _price_change_stats(df: pd.DataFrame) -> tuple[float, float, float, float, float]:
    """Return (prev_close, pct_change, avg_vol, latest_vol, vol_ratio) for the last bar."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    prev_close = float(prev["Close"])
    pct_change = ((latest["Close"] - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0
    avg_vol = df["Volume"].mean()
    latest_vol = latest["Volume"]
    vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 0
    return prev_close, pct_change, avg_vol, latest_vol, vol_ratio


def _build_prompt_data_lines(
    symbol: str, period: str, df: pd.DataFrame,
    fundamentals: dict | None = None,
    earnings_info: str | None = None,
    market_context: dict | None = None,
    support_levels: list[float] | None = None,
    resistance_levels: list[float] | None = None,
    strategic_period: str | None = None,
    news_headlines: list[dict] | None = None,
) -> list[str]:
    """Assemble the shared data lines used by both observation and analysis prompts."""
    latest = df.iloc[-1]
    prev_close, pct_change, avg_vol, latest_vol, vol_ratio = _price_change_stats(df)

    indicator_text = ", ".join(_ALL_INDICATOR_LABELS)

    # --- Image description ---
    lines: list[str] = []
    if strategic_period:
        lines.append(
            f"You are provided two chart images. "
            f"Image 1: {period} chart (primary analysis). "
            f"Image 2: {strategic_period} chart (strategic context).\n"
        )
    else:
        lines.append(f"You are provided one chart image for the {period} timeframe.\n")

    # --- Core data ---
    lines += [
        f"Analyze this candlestick chart for {symbol.upper()}.\n",
        f"Timeframe: {period} (latest bar is the most recent trading day)\n",
        "Key data points:",
        f"- Latest close: ${latest['Close']:.2f}",
        f"- Previous close: ${prev_close:.2f}",
        f"- Change: {pct_change:+.2f}%",
        f"- Period high: ${df['High'].max():.2f}",
        f"- Period low: ${df['Low'].min():.2f}",
        f"- Average volume: {avg_vol:,.0f}",
        f"- Latest volume: {latest_vol:,.0f} ({vol_ratio:.1f}x average)",
    ]

    # --- Actual indicator values ---
    lines.append("\nIndicator readings (latest bar):")
    sma20 = latest.get("SMA_20")
    sma50 = latest.get("SMA_50")
    sma20_str = f"${sma20:.2f}" if pd.notna(sma20) else "N/A (insufficient data)"
    sma50_str = f"${sma50:.2f}" if pd.notna(sma50) else "N/A (insufficient data)"
    lines.append(f"- SMA 20: {sma20_str}, SMA 50: {sma50_str}")
    if pd.notna(sma20) and pd.notna(sma50):
        cross = "above" if sma20 > sma50 else "below"
        lines.append(f"  SMA 20 is {cross} SMA 50 ({'bullish' if cross == 'above' else 'bearish'} alignment)")
    ema12 = latest.get("EMA_12")
    ema26 = latest.get("EMA_26")
    ema12_str = f"${ema12:.2f}" if pd.notna(ema12) else "N/A"
    ema26_str = f"${ema26:.2f}" if pd.notna(ema26) else "N/A"
    lines.append(f"- EMA 12: {ema12_str}, EMA 26: {ema26_str}")
    if pd.notna(ema12) and pd.notna(ema26):
        cross = "above" if ema12 > ema26 else "below"
        lines.append(f"  EMA 12 is {cross} EMA 26 ({'bullish' if cross == 'above' else 'bearish'} alignment)")
    bb_upper = latest.get("BB_Upper")
    bb_lower = latest.get("BB_Lower")
    bb_mid = latest.get("BB_Mid")
    if pd.notna(bb_upper) and pd.notna(bb_lower):
        close_price = latest["Close"]
        lines.append(f"- Bollinger Bands: Upper ${bb_upper:.2f}, Mid ${bb_mid:.2f}, Lower ${bb_lower:.2f}")
        if close_price > bb_upper:
            lines.append("  Price is ABOVE upper band (overbought / breakout)")
        elif close_price < bb_lower:
            lines.append("  Price is BELOW lower band (oversold / breakdown)")
        elif bb_upper != bb_lower:
            pct_bb = (close_price - bb_lower) / (bb_upper - bb_lower) * 100
            lines.append(f"  Price is at {pct_bb:.0f}% of band width")
        else:
            lines.append("  Price is at mid-band (bands are flat)")
    else:
        lines.append("- Bollinger Bands: N/A (insufficient data)")
    rsi = latest.get("RSI")
    if pd.notna(rsi):
        rsi_label = " (overbought)" if rsi > 70 else " (oversold)" if rsi < 30 else ""
        lines.append(f"- RSI (14): {rsi:.1f}{rsi_label}")
    else:
        lines.append("- RSI (14): N/A (insufficient data)")
    macd = latest.get("MACD")
    macd_sig = latest.get("MACD_Signal")
    macd_hist = latest.get("MACD_Hist")
    if pd.notna(macd) and pd.notna(macd_sig):
        cross = "above" if macd > macd_sig else "below"
        lines.append(f"- MACD: {macd:.4f}, Signal: {macd_sig:.4f}, Histogram: {macd_hist:.4f}")
        lines.append(f"  MACD is {cross} signal line ({'bullish' if cross == 'above' else 'bearish'})")
    else:
        lines.append("- MACD: N/A (insufficient data)")
    atr_val = latest.get("ATR")
    if pd.notna(atr_val):
        close_price = latest["Close"]
        if close_price > 0:
            atr_pct = (atr_val / close_price) * 100
            lines.append(f"- ATR (14): ${atr_val:.2f} ({atr_pct:.1f}% of price)")
        else:
            lines.append(f"- ATR (14): ${atr_val:.2f}")
    else:
        lines.append("- ATR (14): N/A (insufficient data)")
    adx_val = latest.get("ADX")
    plus_di = latest.get("Plus_DI")
    minus_di = latest.get("Minus_DI")
    if pd.notna(adx_val) and pd.notna(plus_di) and pd.notna(minus_di):
        if adx_val < 20:
            adx_label = "weak/no trend"
        elif adx_val < 25:
            adx_label = "emerging trend"
        elif adx_val < 50:
            adx_label = "strong trend"
        else:
            adx_label = "very strong trend"
        dominant = "+DI (bullish)" if plus_di > minus_di else "-DI (bearish)"
        lines.append(f"- ADX (14): {adx_val:.1f} ({adx_label})")
        lines.append(f"  +DI: {plus_di:.1f}, -DI: {minus_di:.1f} — {dominant} dominant")
    else:
        lines.append("- ADX (14): N/A (insufficient data)")

    # --- Fundamental context ---
    if fundamentals:
        lines.append("\nFundamental context:")
        if fundamentals.get("sector"):
            lines.append(f"- Sector: {fundamentals['sector']}, Industry: {fundamentals.get('industry', 'N/A')}")
        if fundamentals.get("trailingPE") is not None:
            lines.append(f"- Trailing P/E: {fundamentals['trailingPE']:.1f}")
        if fundamentals.get("forwardPE") is not None:
            lines.append(f"- Forward P/E: {fundamentals['forwardPE']:.1f}")
        if fundamentals.get("marketCap") is not None:
            mc = fundamentals["marketCap"]
            if mc >= 1e12:
                lines.append(f"- Market cap: ${mc/1e12:.2f}T")
            elif mc >= 1e9:
                lines.append(f"- Market cap: ${mc/1e9:.2f}B")
            else:
                lines.append(f"- Market cap: ${mc/1e6:.0f}M")
        if fundamentals.get("dividendYield") is not None:
            lines.append(f"- Dividend yield: {fundamentals['dividendYield']*100:.2f}%")
        if fundamentals.get("fiftyTwoWeekHigh") is not None and fundamentals.get("fiftyTwoWeekLow") is not None:
            lines.append(
                f"- 52-week range: ${fundamentals['fiftyTwoWeekLow']:.2f} "
                f"- ${fundamentals['fiftyTwoWeekHigh']:.2f}"
            )

    # --- Earnings ---
    if earnings_info:
        lines.append(f"\nUpcoming earnings: {earnings_info}")
        lines.append("  Note: Earnings announcements can cause significant price gaps regardless of technical setup.")

    # --- Market context ---
    if market_context:
        lines.append("\nMarket context (S&P 500):")
        lines.append(f"- S&P 500 latest: {market_context['latest_close']:,.2f}")
        lines.append(f"- S&P 500 period return: {market_context['period_return']:+.2f}%")
        lines.append(
            f"- S&P 500 period range: {market_context['period_low']:,.2f} "
            f"- {market_context['period_high']:,.2f}"
        )

    # --- News headlines ---
    if news_headlines:
        lines.append("\nRecent news:")
        for item in news_headlines:
            date_part = f" ({item['date']})" if item.get("date") else ""
            pub_part = f" — {item['publisher']}" if item.get("publisher") else ""
            lines.append(f"- {item['title']}{pub_part}{date_part}")

    # --- Support/Resistance ---
    if support_levels:
        lines.append(
            f"\nAlgorithmic support levels: "
            + ", ".join(f"${lvl:.2f}" for lvl in support_levels)
        )
    if resistance_levels:
        lines.append(
            "Algorithmic resistance levels: "
            + ", ".join(f"${lvl:.2f}" for lvl in resistance_levels)
        )

    # --- Recent price action (last 5 bars) ---
    recent = df.tail(5)
    lines.append("\nRecent price action (last 5 bars):")
    lines.append("Date | Open | High | Low | Close | Volume")
    for row in recent.itertuples():
        idx = row.Index
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        lines.append(
            f"{date_str} | ${row.Open:.2f} | ${row.High:.2f} | "
            f"${row.Low:.2f} | ${row.Close:.2f} | {row.Volume:,.0f}"
        )

    lines.append(f"\nVisible indicators on chart: {indicator_text}")

    return lines


def build_observation_messages(
    symbol: str, period: str, df: pd.DataFrame,
    fundamentals: dict | None = None,
    earnings_info: str | None = None,
    market_context: dict | None = None,
    support_levels: list[float] | None = None,
    resistance_levels: list[float] | None = None,
    strategic_period: str | None = None,
    news_headlines: list[dict] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the observation pass (pass 1).

    Instructs the model to list factual observations only — no predictions
    or recommendations.
    """
    system_prompt = (
        "You are a stock market technical analyst. Your task is to OBSERVE and LIST "
        "every technical signal visible on the chart and in the data. Do NOT make "
        "predictions, recommendations, or draw conclusions.\n\n"
        "List your observations in these categories:\n"
        "- **Price Action**: Trend direction, higher highs/lows, consolidation, gaps\n"
        "- **Indicator Readings**: What each active indicator currently shows\n"
        "- **Candlestick Patterns**: Any notable patterns visible on the chart\n"
        "- **Volume Behavior**: Volume trends, spikes, divergences\n"
        "- **Support & Resistance**: Key levels visible on the chart\n"
        "- **News Context**: Implications of any recent headlines (if provided)\n\n"
        "Be thorough and factual. Stick to what you can see and measure."
    )

    lines = _build_prompt_data_lines(
        symbol, period, df,
        fundamentals=fundamentals,
        earnings_info=earnings_info,
        market_context=market_context,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        strategic_period=strategic_period,
        news_headlines=news_headlines,
    )
    lines.append(
        "\nList all technical observations from the chart image(s) and numeric data above. "
        "Observations only — no predictions or recommendations."
    )

    return system_prompt, "\n".join(lines)


def build_analysis_messages(
    symbol: str, period: str, df: pd.DataFrame,
    fundamentals: dict | None = None,
    earnings_info: str | None = None,
    market_context: dict | None = None,
    support_levels: list[float] | None = None,
    resistance_levels: list[float] | None = None,
    strategic_period: str | None = None,
    news_headlines: list[dict] | None = None,
    observations: str | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the vision analysis."""
    system_prompt = (
        "You are a stock market technical analyst specializing in chart analysis. "
        "You combine visual chart reading with quantitative indicator data to produce "
        "actionable assessments.\n\n"
        "Respond with exactly these sections:\n"
        "**Trend**: Overall trend direction and strength.\n"
        "**Support & Resistance**: Key price levels identified from the chart.\n"
        "**Indicator Signals**: What each active indicator suggests.\n"
        "**Candlestick Patterns**: Notable patterns visible on the chart.\n"
        "**Outlook**: Short-term price outlook with confidence level (High / Medium / Low).\n"
        "**Risk Factors**: Key risks to the outlook.\n\n"
        "If signals are mixed or data is insufficient for a clear call, say so explicitly "
        "rather than forcing a directional prediction. Be concise.\n\n"
        "Example of the expected format and tone:\n"
        "**Trend**: Moderate uptrend. Price is making higher highs and higher lows over the "
        "past 3 weeks, though momentum is slowing as price approaches resistance near $185.\n"
        "**Support & Resistance**: Support at $172 (SMA 50 confluence), resistance at $185 "
        "(prior swing high). A break above $185 targets $192.\n"
        "**Indicator Signals**: SMA 20 > SMA 50 (bullish). RSI at 62 — elevated but not "
        "overbought. MACD histogram shrinking, suggesting waning momentum.\n"
        "**Candlestick Patterns**: Small-bodied candles near resistance suggest indecision.\n"
        "**Outlook**: Cautiously bullish. Likely consolidation near $185 before a decisive "
        "move. Confidence: Medium.\n"
        "**Risk Factors**: Earnings in 5 days could override technicals. Failure to hold "
        "$172 would negate the bullish setup."
    )

    if observations is not None:
        # Slim prompt: just context + observations, no redundant data lines
        latest = df.iloc[-1]
        indicator_text = ", ".join(_ALL_INDICATOR_LABELS)
        lines = [
            f"Synthesize a technical analysis for {symbol.upper()}.",
            f"Timeframe: {period}, Latest close: ${latest['Close']:.2f}",
            f"Indicators: {indicator_text}",
            "\n--- Your prior observations (from pass 1) ---\n"
            f"{observations}\n"
            "--- End of observations ---\n"
            "\nUsing your observations above (and the chart image if provided), synthesize "
            "your analysis into the standard format. Focus on interpreting and drawing "
            "conclusions from your observations rather than re-describing what you see."
        ]
    else:
        lines = _build_prompt_data_lines(
            symbol, period, df,
            fundamentals=fundamentals,
            earnings_info=earnings_info,
            market_context=market_context,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            strategic_period=strategic_period,
            news_headlines=news_headlines,
        )
        lines.append("\nAnalyze both the chart image(s) and the numeric data above.")

    return system_prompt, "\n".join(lines)


@st.cache_data(show_spinner=False, ttl=30)
def fetch_ollama_models(base_url: str) -> list[dict]:
    """Fetch all models with parameter size and capabilities from Ollama."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except Exception:
        logger.warning("Failed to fetch Ollama model list", exc_info=True)
        return []

    def _fetch_model_info(m: dict) -> dict | None:
        try:
            resp = requests.post(
                f"{base_url}/api/show",
                json={"model": m["name"]},
                timeout=5,
            )
            resp.raise_for_status()
            show = resp.json()
            capabilities = show.get("capabilities", [])
            param_size = show.get("details", {}).get("parameter_size", "")
            disk_bytes = m.get("size", 0)
            if disk_bytes >= 1 << 30:
                disk_size = f"{disk_bytes / (1 << 30):.1f} GB"
            else:
                disk_size = f"{disk_bytes / (1 << 20):.0f} MB"
            # Extract context length from model_info: <arch>.context_length
            model_info = show.get("model_info", {})
            arch = model_info.get("general.architecture", "")
            context_length = model_info.get(f"{arch}.context_length") if arch else None

            return {
                "name": m["name"],
                "parameter_size": param_size,
                "disk_size": disk_size,
                "capabilities": capabilities,
                "context_length": context_length,
            }
        except Exception:
            logger.warning("Failed to fetch info for model %s", m["name"], exc_info=True)
            return None

    result = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_model_info, m): m["name"] for m in models}
        for future in as_completed(futures):
            info = future.result()
            if info is not None:
                result.append(info)
    # Preserve original ordering from /api/tags
    name_order = {m["name"]: i for i, m in enumerate(models)}
    result.sort(key=lambda r: name_order.get(r["name"], 0))
    return result


def _escape_markdown(text: str) -> str:
    """Escape characters that Streamlit's markdown renderer misinterprets.

    Only escapes $ (LaTeX trigger) and ~~ (strikethrough). Leaves _ alone
    since models produce intentional markdown formatting with underscores.

    Note: this is NOT a true inverse of _unescape_markdown for text that
    already contains literal ``\\$`` or ``\\~\\~`` — those will round-trip
    incorrectly.  This is acceptable because Ollama model output virtually
    never contains those escape sequences.
    """
    return text.replace("$", "\\$").replace("~~", "\\~\\~")


def _unescape_markdown(text: str) -> str:
    """Reverse _escape_markdown so raw text can be re-used in prompts.

    See caveat on _escape_markdown about round-trip fidelity.
    """
    return text.replace("\\~\\~", "~~").replace("\\$", "$")


def stream_ollama_response(
    model: str,
    system_prompt: str,
    user_prompt: str,
    images_b64: list[str] | None = None,
    temperature: float = 0.4,
    num_ctx: int | None = None,
    base_url: str = "http://localhost:11434",
) -> Iterator[tuple[str, str]]:
    """Generator that yields ``(type, token)`` tuples from Ollama ``/api/chat``.

    *type* is ``"thinking"`` for thinking-mode tokens (e.g. Qwen3) or
    ``"content"`` for regular content tokens.

    If *num_ctx* is provided it is passed as the context window size in the
    Ollama options, overriding the default (2048).

    If the model finishes without producing any content tokens but did emit
    thinking tokens, the accumulated thinking text is yielded as
    ``("content", ...)`` so callers always receive usable output.
    """
    user_message: dict = {"role": "user", "content": user_prompt}
    if images_b64:
        user_message["images"] = images_b64
    options: dict = {"temperature": temperature}
    if num_ctx is not None and isinstance(num_ctx, int):
        options["num_ctx"] = num_ctx
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            user_message,
        ],
        "stream": True,
        "options": options,
    }
    with requests.post(
        f"{base_url}/api/chat", json=payload, stream=True, timeout=(10, 600)
    ) as resp:
        resp.raise_for_status()
        thinking_buf: list[str] = []
        has_content = False
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON from Ollama: %s", line[:200])
                    continue
                if data.get("error"):
                    raise RuntimeError(f"Ollama error: {data['error']}")
                msg = data.get("message", {})
                token = msg.get("content", "")
                if token:
                    has_content = True
                    yield ("content", _escape_markdown(token))
                else:
                    thinking_token = msg.get("thinking", "")
                    if thinking_token:
                        thinking_buf.append(thinking_token)
                        yield ("thinking", _escape_markdown(thinking_token))
                if data.get("done", False):
                    if not has_content and thinking_buf:
                        logger.info("Model %s produced only thinking output; using as fallback", model)
                        yield ("content", _escape_markdown("".join(thinking_buf)))
                    elif not has_content and not thinking_buf:
                        logger.warning("Model %s returned done with no content or thinking tokens", model)
                    return


_STREAM_RENDER_INTERVAL = 0.05  # seconds between UI re-renders during streaming


def _stream_to_ui(
    token_stream,
) -> tuple[str, str | None]:
    """Render a ``stream_ollama_response`` stream into Streamlit, returning text.

    Displays content tokens as markdown.  When thinking tokens are detected,
    the layout switches to two columns (thinking on the left, content on the
    right) so the user can follow the model's reasoning in real time.

    Re-renders are throttled to every ~50 ms to avoid O(n²) rendering overhead
    when the model streams many small tokens quickly.

    Returns ``(content_text, thinking_text_or_none)``.
    """
    placeholder = st.empty()
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    has_thinking = False
    last_render = 0.0
    dirty = False

    def _render() -> None:
        with placeholder.container():
            if has_thinking:
                col_think, col_content = st.columns([1, 2])
                with col_think:
                    st.caption("Thinking\u2026")
                    st.markdown("".join(thinking_parts))
                with col_content:
                    st.markdown("".join(content_parts))
            else:
                st.markdown("".join(content_parts))

    for kind, token in token_stream:
        if kind == "thinking":
            if not has_thinking:
                has_thinking = True
            thinking_parts.append(token)
        else:
            content_parts.append(token)

        now = time.monotonic()
        if now - last_render >= _STREAM_RENDER_INTERVAL:
            _render()
            last_render = now
            dirty = False
        else:
            dirty = True

    # Always render the final state so nothing is lost.
    if dirty or last_render == 0.0:
        _render()

    content_text = "".join(content_parts)
    thinking_text = "".join(thinking_parts) if thinking_parts else None
    return content_text, thinking_text


def _run_ollama_pass(
    model: str,
    system_prompt: str,
    user_prompt: str,
    images_b64: list[str] | None = None,
    label: str = "",
    num_ctx: int | None = None,
    base_url: str = "http://localhost:11434",
) -> tuple[str | None, str | None]:
    """Run a streaming Ollama pass, returning (result, error).

    Handles ConnectionError, HTTPError, and unexpected exceptions with
    user-friendly messages. Returns the streamed text on success, or an
    error string on failure.
    """
    prefix = f"{label}: " if label else ""
    try:
        result, _thinking = _stream_to_ui(
            stream_ollama_response(model, system_prompt, user_prompt, images_b64, num_ctx=num_ctx, base_url=base_url)
        )
        if not result or not result.strip():
            # Some models (e.g. Qwen3-VL) fail silently with multiple images.
            # Retry with only the first image before giving up.
            if images_b64 and len(images_b64) > 1:
                logger.info("Model %s returned empty with %d images; retrying with 1 image", model, len(images_b64))
                # Fix prompt text to match the single image being sent
                retry_prompt = re.sub(
                    r"You are provided two chart images\. "
                    r"Image 1: (\S+) chart \(primary analysis\)\. "
                    r"Image 2: \S+ chart \(strategic context\)\.\n",
                    r"You are provided one chart image for the \1 timeframe.\n",
                    user_prompt,
                )
                result, _thinking = _stream_to_ui(
                    stream_ollama_response(model, system_prompt, retry_prompt, images_b64[:1], num_ctx=num_ctx, base_url=base_url)
                )
            if not result or not result.strip():
                logger.warning("Model %s returned an empty response", model)
                return None, f"{prefix}Model returned an empty response."
        return result, None
    except requests.ConnectionError:
        logger.warning("Cannot connect to Ollama at %s", base_url, exc_info=True)
        return None, (
            f"{prefix}Cannot connect to Ollama at `{base_url}`. "
            "Make sure Ollama is running."
        )
    except requests.Timeout:
        logger.warning("Ollama request timed out", exc_info=True)
        return None, (
            f"{prefix}Ollama is not responding (timed out). "
            "Is a model loaded? Try again or check Ollama status."
        )
    except requests.HTTPError as e:
        logger.warning("Ollama HTTP error", exc_info=True)
        return None, f"{prefix}Ollama returned an error: {e}"
    except Exception as e:
        logger.warning("Unexpected error during Ollama pass", exc_info=True)
        return None, f"{prefix}Unexpected error: {e}"


def _format_model_label(m: dict) -> str:
    """Format a model dict into a display label with size info and vision tag."""
    parts = [m["parameter_size"], m["disk_size"]]
    info = ", ".join(p for p in parts if p)
    label = f"{m['name']} ({info})" if info else m["name"]
    if "vision" in m.get("capabilities", []):
        label += " [vision]"
    return label


def build_consensus_messages(
    symbol: str,
    model_analyses: dict[str, str],
    model_sizes: dict[str, str],
    df: pd.DataFrame,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the consensus summarizer."""
    system_prompt = (
        "You are a financial analysis synthesizer. You combine independent technical "
        "analyses into a balanced consensus view. Larger models (more parameters) may "
        "provide more nuanced analysis — consider this when assessments conflict, but "
        "do not dismiss smaller models outright.\n\n"
        "If the analyses fundamentally disagree, say so clearly rather than "
        "manufacturing false consensus. Rate your overall confidence honestly."
    )

    latest = df.iloc[-1]
    indicator_text = ", ".join(_ALL_INDICATOR_LABELS)

    analyses_text = ""
    for model_name, analysis in model_analyses.items():
        size = model_sizes.get(model_name, "unknown size")
        analyses_text += f"\n--- {model_name} ({size}) ---\n{_unescape_markdown(analysis)}\n"

    model_names = ", ".join(model_analyses.keys())
    user_prompt = (
        f"Below are independent technical analyses of a candlestick chart for "
        f"{symbol.upper()} (latest close: ${latest['Close']:.2f}) "
        f"with these indicators: {indicator_text}.\n\n"
        f"Individual analyses:{analyses_text}\n"
        f"Synthesize these into a consensus report with exactly these sections. "
        f"Refer to each model by its name (e.g. {model_names}) rather than "
        f"generic labels like 'Model 1'.\n\n"
        f"**Consensus Level**: Rate as Strong / Moderate / Mixed / Contradictory based on "
        f"how well the analyses agree.\n\n"
        f"**Points of Agreement**: Key findings where the analysts concur.\n\n"
        f"**Points of Disagreement**: Areas where the analysts diverge or contradict.\n\n"
        f"**Synthesized Outlook**: Your overall assessment combining all perspectives, "
        f"noting confidence level (High / Medium / Low). Be concise."
    )

    return system_prompt, user_prompt


def build_watchlist_prompt(
    symbol: str, df: pd.DataFrame,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a brief watchlist scan."""
    system_prompt = (
        "You are a stock screener assistant. Provide a brief, structured assessment "
        "of the chart. Be direct and concise — this is a quick scan, not a deep analysis."
    )

    latest = df.iloc[-1]
    _prev_close, pct_change, avg_vol, latest_vol, vol_ratio = _price_change_stats(df)

    lines = [
        f"Quick scan for {symbol.upper()}.\n",
        f"- Close: ${latest['Close']:.2f} ({pct_change:+.2f}%)",
        f"- Period high/low: ${df['High'].max():.2f} / ${df['Low'].min():.2f}",
        f"- Volume: {latest_vol:,.0f} ({vol_ratio:.1f}x avg)",
    ]

    sma20 = latest.get("SMA_20")
    sma50 = latest.get("SMA_50")
    if pd.notna(sma20) and pd.notna(sma50):
        lines.append(f"- SMA 20: ${sma20:.2f}, SMA 50: ${sma50:.2f}")
    rsi = latest.get("RSI")
    if pd.notna(rsi):
        lines.append(f"- RSI: {rsi:.1f}")
    macd = latest.get("MACD")
    macd_sig = latest.get("MACD_Signal")
    if pd.notna(macd) and pd.notna(macd_sig):
        lines.append(f"- MACD: {macd:.4f}, Signal: {macd_sig:.4f}")

    lines.append(
        "\nRespond with exactly:\n"
        "**Trend**: (Up / Down / Sideways)\n"
        "**Confidence**: (High / Medium / Low)\n"
        "**Key Signals**: 2-3 bullet points\n"
        "**Outlook**: One sentence"
    )

    return system_prompt, "\n".join(lines)


def save_analysis(
    symbol: str, period: str, models: list[str],
    analyses: dict[str, str], consensus: str,
) -> None:
    """Append an analysis entry to the history file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol.upper(),
        "period": period,
        "models": models,
        "analyses": {k: _unescape_markdown(v) for k, v in analyses.items()},
        "consensus": _unescape_markdown(consensus) if consensus else "",
    }
    history: list[dict] = []
    if ANALYSIS_HISTORY_FILE.exists():
        try:
            with open(ANALYSIS_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load analysis history for save; starting fresh", exc_info=True)
            history = []
    history.append(entry)
    # Keep only the most recent 100 entries to prevent unbounded growth
    history = history[-100:]
    fd, tmp_path = tempfile.mkstemp(
        dir=ANALYSIS_HISTORY_FILE.parent, suffix=".tmp"
    )
    try:
        with open(fd, "w", closefd=True) as f:
            json.dump(history, f, indent=2)
        Path(tmp_path).replace(ANALYSIS_HISTORY_FILE)
    except BaseException:
        # Ensure fd is closed even if open() itself fails (fd not yet owned by file object)
        try:
            os.close(fd)
        except OSError:
            pass
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_analysis_history(symbol: str, limit: int = 5) -> list[dict]:
    """Load recent analysis history for a symbol."""
    if not ANALYSIS_HISTORY_FILE.exists():
        return []
    try:
        with open(ANALYSIS_HISTORY_FILE, "r") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load analysis history", exc_info=True)
        return []
    filtered = [h for h in history if h.get("symbol") == symbol.upper()]
    return filtered[-limit:]


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Stock Chart AI Analyzer", layout="wide")
st.title("Stock Chart AI Analyzer")

# State initialization
for key, default in {
    "analyzing": False,
    "done": False,
    "history_saved": False,
    "strategic_chart_b64": None,
    "strategic_chart_attempted": False,
    "ai_outputs": {},
    "ai_errors": {},
    "consensus_output": "",
    "consensus_error": "",
    "analysis_step": 0,
    "analysis_models": [],
    "consensus_model_name": None,
    "watchlist_analyzing": False,
    "watchlist_done": False,
    "watchlist_symbols": [],
    "watchlist_step": 0,
    "watchlist_results": {},
    "watchlist_model": None,
    "chart_b64": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _on_input_change():
    """Clear 'done' state and stale output/error when any input changes."""
    st.session_state.done = False
    st.session_state.ai_outputs = {}
    st.session_state.ai_errors = {}
    st.session_state.consensus_output = ""
    st.session_state.consensus_error = ""
    st.session_state.analysis_step = 0
    st.session_state.analysis_models = []
    st.session_state.consensus_model_name = None
    st.session_state.chart_b64 = None
    st.session_state.strategic_chart_b64 = None
    st.session_state.strategic_chart_attempted = False
    st.session_state.history_saved = False


def _on_watchlist_input_change():
    """Clear watchlist results when inputs change."""
    st.session_state.watchlist_done = False
    st.session_state.watchlist_results = {}
    st.session_state.watchlist_step = 0
    st.session_state.watchlist_symbols = []


def _on_shared_input_change():
    _on_input_change()
    _on_watchlist_input_change()


def _uppercase_symbol():
    st.session_state.symbol = _SYMBOL_RE.sub("", st.session_state.symbol.strip()).upper()
    _on_input_change()


locked = st.session_state.analyzing or st.session_state.watchlist_analyzing

def _stop_analysis():
    st.session_state.analyzing = False
    st.session_state.watchlist_analyzing = False
    st.session_state.done = bool(st.session_state.ai_outputs)
    st.session_state.watchlist_done = bool(st.session_state.watchlist_results)
    st.toast("Analysis stopped.")

# Sidebar: mode selection
mode = st.sidebar.radio(
    "Mode",
    ["Single Symbol", "Watchlist"],
    horizontal=True,
    disabled=locked,
)
is_single_mode = mode == "Single Symbol"

# Sidebar: symbol input
if is_single_mode:
    symbol = st.sidebar.text_input(
        "Stock symbol", key="symbol", max_chars=10,
        placeholder="e.g. AAPL", on_change=_uppercase_symbol,
        disabled=locked,
    )
else:
    st.sidebar.text_area(
        "Symbols (comma or newline separated)",
        key="watchlist_text",
        placeholder="AAPL, MSFT, GOOGL\nAMZN, TSLA",
        disabled=locked,
        on_change=_on_watchlist_input_change,
    )
    symbol = ""

period = st.sidebar.selectbox(
    "Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2, on_change=_on_shared_input_change, disabled=locked,
)

# Sidebar: model selection + action button
st.sidebar.divider()
_ollama_host_raw = st.sidebar.text_input(
    "Ollama host", value=_OLLAMA_DEFAULT_HOST, disabled=locked,
    placeholder="e.g. localhost or 192.168.1.100:11434",
)
_ollama_host = _ollama_host_raw.strip() or _OLLAMA_DEFAULT_HOST
if not _HOST_RE.match(_ollama_host):
    st.sidebar.error("Invalid host — only letters, digits, dots, hyphens, and colons allowed.")
    _ollama_host = _OLLAMA_DEFAULT_HOST
if ":" in _ollama_host:
    OLLAMA_BASE_URL = f"http://{_ollama_host}"
else:
    OLLAMA_BASE_URL = f"http://{_ollama_host}:11434"
available_models = fetch_ollama_models(OLLAMA_BASE_URL)
vision_models = [m for m in available_models if "vision" in m.get("capabilities", [])]
_model_ctx: dict[str, int | None] = {m["name"]: m.get("context_length") for m in available_models}

selected_vision_names: list[str] = []
selected_consensus_model: str | None = None

if is_single_mode:
    # ── Single mode button label ──
    if st.session_state.analyzing:
        step = st.session_state.get("analysis_step", 0)
        models = st.session_state.get("analysis_models", [])
        total = len(models)
        consensus_model = st.session_state.get("consensus_model_name")
        if step <= total and total > 1:
            button_label = f"Analyzing ({step}/{total} models)..."
        elif step <= total:
            button_label = "Analyzing..."
        elif consensus_model:
            button_label = "Generating consensus..."
        else:
            button_label = "Analyzing..."
    elif st.session_state.done:
        button_label = "Re-analyze"
    else:
        button_label = "Analyze with AI"

    if vision_models:
        vision_labels = {m["name"]: _format_model_label(m) for m in vision_models}
        vision_names = list(vision_labels.keys())

        if "selected_vision" not in st.session_state:
            st.session_state.selected_vision = []

        all_selected = set(st.session_state.selected_vision) == set(vision_names)
        st.session_state.select_all_vision = all_selected

        def _toggle_select_all():
            if st.session_state.select_all_vision:
                st.session_state.selected_vision = list(vision_names)
            else:
                st.session_state.selected_vision = []
            _on_input_change()

        st.sidebar.checkbox(
            "Select all vision models",
            key="select_all_vision",
            disabled=locked,
            on_change=_toggle_select_all,
        )
        selected_vision_names = st.sidebar.multiselect(
            "Vision models",
            options=vision_names,
            key="selected_vision",
            format_func=lambda n: vision_labels[n],
            on_change=_on_input_change,
            disabled=locked,
        )
    else:
        st.sidebar.warning("No vision-capable models found in Ollama.")

    if available_models:
        all_labels = {m["name"]: _format_model_label(m) for m in available_models}
        all_names = list(all_labels.keys())
        consensus_enabled = len(selected_vision_names) >= 2
        consensus_selection = st.sidebar.selectbox(
            "Consensus model",
            options=[None] + all_names,
            index=0,
            format_func=lambda n: "Select a model..." if n is None else all_labels[n],
            on_change=_on_input_change,
            disabled=locked or not consensus_enabled,
            help="Summarizes analyses from multiple vision models. Requires 2+ vision models selected.",
        )
        if consensus_enabled and consensus_selection is not None:
            selected_consensus_model = consensus_selection

    needs_consensus = len(selected_vision_names) >= 2 and selected_consensus_model is None
    button_disabled = locked or not symbol or not selected_vision_names or needs_consensus

    if st.sidebar.button(
        button_label,
        type="primary",
        disabled=button_disabled,
        width="stretch",
    ):
        st.session_state.analyzing = True
        st.session_state.done = False
        st.session_state.analysis_step = 1
        st.session_state.analysis_models = list(selected_vision_names)
        st.session_state.consensus_model_name = selected_consensus_model
        st.session_state.ai_outputs = {}
        st.session_state.ai_errors = {}
        st.session_state.consensus_output = ""
        st.session_state.consensus_error = ""
        st.session_state.history_saved = False
        st.session_state.chart_b64 = None
        st.session_state.strategic_chart_b64 = None
        st.session_state.strategic_chart_attempted = False
        st.rerun()

    if locked:
        st.sidebar.button(
            "Stop after current step",
            on_click=_stop_analysis,
            type="secondary",
            width="stretch",
        )

else:
    # ── Watchlist mode button label ──
    if st.session_state.watchlist_analyzing:
        wl_step = st.session_state.get("watchlist_step", 0)
        wl_total = len(st.session_state.get("watchlist_symbols", []))
        button_label = f"Scanning ({wl_step}/{wl_total} symbols)..."
    elif st.session_state.watchlist_done:
        button_label = "Re-scan"
    else:
        button_label = "Scan Watchlist"

    selected_watchlist_model: str | None = None
    if vision_models:
        vision_labels = {m["name"]: _format_model_label(m) for m in vision_models}
        vision_names = list(vision_labels.keys())
        selected_watchlist_model = st.sidebar.selectbox(
            "Vision model",
            options=[None] + vision_names,
            index=0,
            format_func=lambda n: "Select a model..." if n is None else vision_labels[n],
            disabled=locked,
        )
    else:
        st.sidebar.warning("No vision-capable models found in Ollama.")

    raw_symbols = st.session_state.get("watchlist_text", "")
    parsed_symbols = list(dict.fromkeys(
        cleaned for s in raw_symbols.replace("\n", ",").split(",")
        if (cleaned := _SYMBOL_RE.sub("", s.strip()).upper())
    ))
    wl_button_disabled = (
        locked
        or not parsed_symbols
        or not selected_watchlist_model
    )

    if st.sidebar.button(
        button_label,
        type="primary",
        disabled=wl_button_disabled,
        width="stretch",
    ):
        st.session_state.watchlist_analyzing = True
        st.session_state.watchlist_done = False
        st.session_state.watchlist_step = 1
        st.session_state.watchlist_symbols = parsed_symbols
        st.session_state.watchlist_results = {}
        st.session_state.watchlist_model = selected_watchlist_model
        st.rerun()

    if locked:
        st.sidebar.button(
            "Stop after current step",
            on_click=_stop_analysis,
            type="secondary",
            width="stretch",
            key="stop_watchlist",
        )


# ── Main Content ─────────────────────────────────────────────────────────────

if is_single_mode and symbol:
    with st.spinner("Fetching market data..."):
        with ThreadPoolExecutor(max_workers=5) as pool:
            fut_df = pool.submit(fetch_stock_data, symbol, period)
            fut_info = pool.submit(_fetch_ticker_info, symbol)
            fut_earn = pool.submit(fetch_next_earnings, symbol)
            fut_mkt = pool.submit(fetch_market_context, period)
            fut_news = pool.submit(fetch_news_headlines, symbol)
        try:
            df = fut_df.result(timeout=_DATA_TIMEOUT)
        except TimeoutError:
            df = None
        # Supplementary data degrades gracefully on timeout
        try:
            fut_info.result(timeout=_DATA_TIMEOUT)  # warm the cache
        except TimeoutError:
            pass
        company_name = fetch_company_name(symbol)
        fundamentals = fetch_fundamentals(symbol)
        try:
            earnings_info = fut_earn.result(timeout=_DATA_TIMEOUT)
        except TimeoutError:
            earnings_info = None
        try:
            market_ctx = fut_mkt.result(timeout=_DATA_TIMEOUT)
        except TimeoutError:
            market_ctx = None
        try:
            news_headlines = fut_news.result(timeout=_DATA_TIMEOUT)
        except TimeoutError:
            news_headlines = []

    if df is None or df.empty or len(df) < 5:
        st.error(f"No data found for **{symbol}** (or insufficient history). Check the symbol and try again.")
    else:
        support_levels, resistance_levels = compute_support_resistance(df)
        chart_title = f"{company_name} ({symbol.upper()})"
        if st.session_state.analyzing and st.session_state.chart_b64:
            chart_png = base64.b64decode(st.session_state.chart_b64)
        else:
            chart_png = build_candlestick_chart(df, symbol, title=chart_title)
        st.image(chart_png, width="stretch")

        # Determine strategic period
        strategic_period = STRATEGIC_PERIOD_MAP.get(period)

        if st.session_state.analyzing:
            step = st.session_state.get("analysis_step", 0)
            models = st.session_state.get("analysis_models", [])
            consensus_model = st.session_state.get("consensus_model_name")
            total = len(models)

            # Capture chart image(s) once on first step
            if st.session_state.chart_b64 is None:
                st.session_state.chart_b64 = base64.b64encode(chart_png).decode("utf-8")

            # Capture strategic chart once (skipped if already attempted)
            if (
                strategic_period
                and not st.session_state.strategic_chart_attempted
                and st.session_state.chart_b64 is not None
            ):
                st.session_state.strategic_chart_attempted = True
                try:
                    strategic_df = fetch_stock_data(symbol, strategic_period)
                    if strategic_df is not None and not strategic_df.empty:
                        strategic_title = f"{company_name} ({symbol.upper()}) - {strategic_period}"
                        strategic_png = build_candlestick_chart(
                            strategic_df, symbol, title=strategic_title
                        )
                        st.session_state.strategic_chart_b64 = base64.b64encode(
                            strategic_png
                        ).decode("utf-8")
                except Exception:
                    logger.warning("Failed to capture strategic chart", exc_info=True)

            if st.session_state.chart_b64 is None:
                st.session_state.analyzing = False
                st.session_state.done = True
                st.rerun()

            # Build image list
            images_b64 = [st.session_state.chart_b64]
            actual_strategic = strategic_period
            if st.session_state.strategic_chart_b64 is not None:
                images_b64.append(st.session_state.strategic_chart_b64)
            else:
                actual_strategic = None

            # Common prompt args for both passes
            prompt_args = dict(
                symbol=symbol, period=period, df=df,
                fundamentals=fundamentals,
                earnings_info=earnings_info,
                market_context=market_ctx,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                strategic_period=actual_strategic,
                news_headlines=news_headlines,
            )

            if 1 <= step <= total:
                # Vision model analysis step (two-pass)
                current_model = models[step - 1]
                st.subheader(f"AI Analysis - {current_model} ({step}/{total})")

                try:
                    # --- Pass 1: Observation ---
                    obs_system, obs_user = build_observation_messages(**prompt_args)
                    with st.status("Pass 1/2: Observing...", expanded=True) as status:
                        obs_text, obs_error = _run_ollama_pass(
                            current_model, obs_system, obs_user, images_b64,
                            label="Observation pass",
                            num_ctx=_model_ctx.get(current_model),
                            base_url=OLLAMA_BASE_URL,
                        )
                        if obs_text is not None:
                            status.update(label="Pass 1/2: Observations complete", state="complete")

                    if obs_error:
                        st.session_state.ai_errors[current_model] = obs_error
                    elif obs_text is not None:
                        # --- Pass 2: Synthesis ---
                        syn_system, syn_user = build_analysis_messages(
                            **prompt_args, observations=_unescape_markdown(obs_text),
                        )
                        syn_images = images_b64
                        result, syn_error = _run_ollama_pass(
                            current_model, syn_system, syn_user, syn_images,
                            label="Synthesis pass",
                            num_ctx=_model_ctx.get(current_model),
                            base_url=OLLAMA_BASE_URL,
                        )
                        if syn_error:
                            st.session_state.ai_errors[current_model] = syn_error
                        elif result is not None:
                            st.session_state.ai_outputs[current_model] = result
                except Exception as e:
                    st.session_state.ai_errors[current_model] = f"Unexpected error: {e}"

                st.session_state.analysis_step = step + 1

                # Check if we're done with vision models
                if step == total:
                    successful = st.session_state.ai_outputs
                    if len(successful) >= 2 and consensus_model:
                        # Move to consensus step
                        st.rerun()
                    else:
                        # Skip consensus
                        st.session_state.analyzing = False
                        st.session_state.done = True
                        st.rerun()
                else:
                    st.rerun()

            elif step == total + 1 and consensus_model:
                # Consensus summarizer step
                successful = st.session_state.ai_outputs
                st.subheader("Generating Consensus...")
                try:
                    model_sizes = {
                        m["name"]: m.get("parameter_size", "unknown")
                        for m in available_models
                    }
                    consensus_sys, consensus_user = build_consensus_messages(
                        symbol, successful, model_sizes, df
                    )
                    result, con_error = _run_ollama_pass(
                        consensus_model, consensus_sys, consensus_user,
                        label="Consensus",
                        num_ctx=_model_ctx.get(consensus_model),
                        base_url=OLLAMA_BASE_URL,
                    )
                    if con_error:
                        st.session_state.consensus_error = con_error
                    elif result is not None:
                        st.session_state.consensus_output = result
                except Exception as e:
                    st.session_state.consensus_error = f"Unexpected error: {e}"
                st.session_state.analyzing = False
                st.session_state.done = True
                st.rerun()

        elif st.session_state.done:
            ai_outputs = st.session_state.get("ai_outputs", {})
            ai_errors = st.session_state.get("ai_errors", {})
            consensus_output = st.session_state.get("consensus_output", "")
            consensus_error = st.session_state.get("consensus_error", "")
            model_errors = {k: v for k, v in ai_errors.items() if k != "_chart"}

            if len(ai_outputs) + len(model_errors) == 1 and not consensus_output:
                # Single model: show directly without expanders
                st.subheader("AI Analysis")
                for model_name, output in ai_outputs.items():
                    st.markdown(output)
                for model_name, error in model_errors.items():
                    st.error(f"**{model_name}**: {error}")
            else:
                # Multi-model: show in expanders
                st.subheader("AI Analysis")
                for model_name in st.session_state.get("analysis_models", []):
                    if model_name in ai_outputs:
                        with st.expander(model_name, expanded=False):
                            st.markdown(ai_outputs[model_name])
                    elif model_name in model_errors:
                        with st.expander(model_name, expanded=False):
                            st.error(model_errors[model_name])

                # Consensus section
                if consensus_output:
                    st.subheader("Consensus Summary")
                    st.markdown(consensus_output)
                elif consensus_error:
                    st.subheader("Consensus Summary")
                    st.error(consensus_error)

            # Show chart export error if any
            if ai_errors.get("_chart"):
                st.error(ai_errors["_chart"])

            # Export analysis as Markdown
            if ai_outputs:
                now = datetime.now()
                md_parts = [f"# {symbol.upper()} Analysis ({period})\n"]
                md_parts.append(f"Date: {now.strftime('%Y-%m-%d %H:%M')}\n")
                for model_name in st.session_state.get("analysis_models", []):
                    if model_name in ai_outputs:
                        md_parts.append(f"\n## {model_name}\n")
                        md_parts.append(_unescape_markdown(ai_outputs[model_name]))
                if consensus_output:
                    md_parts.append("\n## Consensus Summary\n")
                    md_parts.append(_unescape_markdown(consensus_output))
                md_text = "\n".join(md_parts)
                st.download_button(
                    "Download Analysis",
                    data=md_text,
                    file_name=f"{symbol.upper()}_analysis_{now.strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                )

            # Save to history (once)
            if ai_outputs and not st.session_state.history_saved:
                try:
                    save_analysis(
                        symbol, period,
                        st.session_state.get("analysis_models", []),
                        ai_outputs,
                        consensus_output,
                    )
                    st.session_state.history_saved = True
                except Exception:
                    logger.warning("Failed to save analysis history", exc_info=True)

            # Show analysis history
            history = load_analysis_history(symbol)
            if history:
                with st.expander("Analysis History", expanded=False):
                    for entry in reversed(history):
                        ts = entry.get("timestamp", "")
                        try:
                            ts_display = datetime.fromisoformat(ts).strftime("%b %d, %Y %I:%M %p")
                        except (ValueError, TypeError):
                            ts_display = ts[:16]
                        models_str = ", ".join(entry.get("models", []))
                        st.markdown(f"**{ts_display}** — {entry.get('period', '')} — {models_str}")
                        for m_name, m_analysis in entry.get("analyses", {}).items():
                            st.markdown(f"*{m_name}:*")
                            st.markdown(_escape_markdown(m_analysis))
                        if entry.get("consensus"):
                            st.markdown("*Consensus:*")
                            st.markdown(_escape_markdown(entry["consensus"]))
                        st.divider()

elif not is_single_mode:
    # ── Watchlist Mode ───────────────────────────────────────────────────────
    if st.session_state.watchlist_analyzing:
        wl_symbols = st.session_state.watchlist_symbols
        wl_step = st.session_state.watchlist_step
        wl_total = len(wl_symbols)
        wl_model = st.session_state.get("watchlist_model")

        if 1 <= wl_step <= wl_total:
            current_sym = wl_symbols[wl_step - 1]

            try:
                wl_df = fetch_stock_data(current_sym, period)
                wl_company = fetch_company_name(current_sym)

                if wl_df is None or wl_df.empty:
                    st.session_state.watchlist_results[current_sym] = {
                        "error": f"No data found for {current_sym}",
                    }
                else:
                    wl_title = f"{wl_company} ({current_sym})"
                    wl_png = build_candlestick_chart(wl_df, current_sym, title=wl_title)
                    wl_img = base64.b64encode(wl_png).decode("utf-8")
                    sys_prompt, usr_prompt = build_watchlist_prompt(current_sym, wl_df)
                    with st.status(f"Scanning {current_sym} ({wl_step}/{wl_total})...", expanded=False) as wl_status:
                        analysis_text, wl_error = _run_ollama_pass(
                            wl_model, sys_prompt, usr_prompt, [wl_img],
                            label=current_sym,
                            num_ctx=_model_ctx.get(wl_model),
                            base_url=OLLAMA_BASE_URL,
                        )
                        if analysis_text is not None:
                            wl_status.update(label=f"{current_sym} complete", state="complete")

                    if wl_error:
                        st.session_state.watchlist_results[current_sym] = {
                            "error": wl_error,
                        }
                    else:
                        latest_price = wl_df["Close"].iloc[-1]
                        st.session_state.watchlist_results[current_sym] = {
                            "analysis": analysis_text,
                            "price": float(latest_price),
                            "company": wl_company,
                        }

            except Exception as e:
                st.session_state.watchlist_results[current_sym] = {
                    "error": f"Error analyzing {current_sym}: {e}",
                }

            st.session_state.watchlist_step = wl_step + 1
            if wl_step == wl_total:
                st.session_state.watchlist_analyzing = False
                st.session_state.watchlist_done = True
            st.rerun()

    if st.session_state.watchlist_done:
        st.subheader("Watchlist Scan Results")
        results = st.session_state.watchlist_results
        for sym in st.session_state.watchlist_symbols:
            res = results.get(sym, {})
            if "error" in res:
                with st.expander(f"{sym} — Error", expanded=False):
                    st.error(res["error"])
            elif "analysis" in res:
                price = res.get("price", 0)
                with st.expander(f"{sym} — ${price:.2f}", expanded=False):
                    st.markdown(res["analysis"])

        # Export watchlist scan as Markdown
        scan_parts = [f"# Watchlist Scan ({period})\n"]
        now = datetime.now()
        scan_parts.append(f"Date: {now.strftime('%Y-%m-%d %H:%M')}\n")
        has_analyses = False
        for sym in st.session_state.watchlist_symbols:
            res = results.get(sym, {})
            if "analysis" in res:
                has_analyses = True
                company = res.get("company", sym)
                price = res.get("price", 0)
                scan_parts.append(f"\n## {company} ({sym}) — ${price:.2f}\n")
                scan_parts.append(_unescape_markdown(res["analysis"]))
            elif "error" in res:
                scan_parts.append(f"\n## {sym} — Error\n")
                scan_parts.append(res["error"])
        if has_analyses:
            st.download_button(
                "Download Scan Results",
                data="\n".join(scan_parts),
                file_name=f"watchlist_scan_{now.strftime('%Y%m%d')}.md",
                mime="text/markdown",
            )
