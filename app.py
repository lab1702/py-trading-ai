import base64
import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

STRATEGIC_PERIOD_MAP = {
    "1mo": "1y",
    "3mo": "2y",
    "6mo": "2y",
    "1y": "5y",
    "2y": "5y",
}

ANALYSIS_HISTORY_FILE = Path(__file__).parent / "analysis_history.json"

CHART_BASE_HEIGHT = 500
CHART_SUBCHART_HEIGHT = 200


@dataclass
class Indicators:
    sma: bool = True
    ema: bool = True
    bb: bool = True
    rsi: bool = True
    macd: bool = True
    atr: bool = True
    adx: bool = True

    def active_labels(self) -> list[str]:
        labels = []
        if self.sma:
            labels.append("SMA (20, 50)")
        if self.ema:
            labels.append("EMA (12, 26)")
        if self.bb:
            labels.append("Bollinger Bands (20, 2σ)")
        if self.rsi:
            labels.append("RSI (14)")
        if self.macd:
            labels.append("MACD (12, 26, 9)")
        if self.atr:
            labels.append("ATR (14)")
        if self.adx:
            labels.append("ADX (14)")
        return labels


@st.cache_data(show_spinner=False, ttl=300)
def fetch_stock_data(symbol: str, period: str) -> pd.DataFrame | None:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        return None
    _compute_indicators(df)
    return df


@st.cache_data(show_spinner=False, ttl=300)
def fetch_company_name(symbol: str) -> str:
    """Look up the long company name for a ticker symbol."""
    try:
        info = yf.Ticker(symbol).info
        return info.get("longName") or info.get("shortName") or symbol.upper()
    except Exception:
        logger.warning("Failed to fetch company name for %s", symbol, exc_info=True)
        return symbol.upper()


@st.cache_data(show_spinner=False, ttl=300)
def fetch_fundamentals(symbol: str) -> dict | None:
    """Fetch fundamental data for a ticker symbol."""
    try:
        info = yf.Ticker(symbol).info
        return {
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "marketCap": info.get("marketCap"),
            "dividendYield": info.get("dividendYield"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        logger.warning("Failed to fetch fundamentals for %s", symbol, exc_info=True)
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
        if highs.iloc[i] == segment_high.max():
            resistance_pts.append(float(highs.iloc[i]))
        segment_low = lows.iloc[i - window:i + window + 1]
        if lows.iloc[i] == segment_low.min():
            support_pts.append(float(lows.iloc[i]))

    def cluster_levels(points: list[float]) -> list[float]:
        if not points:
            return []
        sorted_pts = sorted(points)
        clusters: list[list[float]] = [[sorted_pts[0]]]
        for p in sorted_pts[1:]:
            mean_last = sum(clusters[-1]) / len(clusters[-1])
            if mean_last != 0 and abs(p - mean_last) / mean_last <= 0.01:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        clusters.sort(key=len, reverse=True)
        return [sum(c) / len(c) for c in clusters[:n_levels]]

    return cluster_levels(support_pts), cluster_levels(resistance_pts)


def _add_overlays(fig: go.Figure, df: pd.DataFrame, ind: Indicators) -> None:
    """Add overlay indicator traces to the price row."""
    if ind.sma:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20",
                       line=dict(width=1, color="#ff9800")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50",
                       line=dict(width=1, color="#e040fb")),
            row=1, col=1,
        )

    if ind.ema:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_12"], name="EMA 12",
                       line=dict(width=1, color="#00e5ff", dash="dot")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_26"], name="EMA 26",
                       line=dict(width=1, color="#ff4081", dash="dot")),
            row=1, col=1,
        )

    if ind.bb:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                       line=dict(width=1, color="#78909c", dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                       line=dict(width=1, color="#78909c", dash="dash"),
                       fill="tonexty", fillcolor="rgba(120,144,156,0.1)"),
            row=1, col=1,
        )


def _add_volume(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add volume bar chart to row 2."""
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=vol_colors, opacity=0.5, showlegend=False),
        row=2, col=1,
    )


def _add_rsi(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Add RSI sub-chart at the given row."""
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                   line=dict(width=1, color="#e040fb")),
        row=row, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350",
                  line_width=1, row=row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a",
                  line_width=1, row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=row, col=1)


def _add_macd(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Add MACD sub-chart at the given row."""
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                   line=dict(width=1, color="#42a5f5")),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                   line=dict(width=1, color="#ff9800")),
        row=row, col=1,
    )
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
               marker_color=colors, showlegend=False),
        row=row, col=1,
    )


def _add_atr(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Add ATR sub-chart at the given row."""
    fig.add_trace(
        go.Scatter(x=df.index, y=df["ATR"], name="ATR",
                   line=dict(width=1, color="#ffa726")),
        row=row, col=1,
    )


def _add_adx(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Add ADX sub-chart at the given row."""
    fig.add_trace(
        go.Scatter(x=df.index, y=df["ADX"], name="ADX",
                   line=dict(width=2, color="#ab47bc")),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Plus_DI"], name="+DI",
                   line=dict(width=1, color="#26a69a", dash="dot")),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Minus_DI"], name="-DI",
                   line=dict(width=1, color="#ef5350", dash="dot")),
        row=row, col=1,
    )
    fig.add_hline(y=25, line_dash="dash", line_color="#78909c",
                  line_width=1, row=row, col=1)


def build_candlestick_chart(df: pd.DataFrame, symbol: str, ind: Indicators,
                            title: str | None = None) -> go.Figure:
    # Determine subplot layout
    rows = 2
    row_heights = [0.5, 0.15]
    price_title = title or f"{symbol.upper()} Price"
    subplot_titles = [price_title, "Volume"]

    # Only add sub-chart rows for indicators with valid data
    show_rsi = ind.rsi and df["RSI"].notna().any()
    show_macd = ind.macd and df["MACD"].notna().any()
    show_atr = ind.atr and df["ATR"].notna().any()
    show_adx = ind.adx and df["ADX"].notna().any()

    if show_rsi:
        rows += 1
        row_heights.append(0.2)
        subplot_titles.append("RSI (14)")
    if show_macd:
        rows += 1
        row_heights.append(0.2)
        subplot_titles.append("MACD")
    if show_atr:
        rows += 1
        row_heights.append(0.2)
        subplot_titles.append("ATR (14)")
    if show_adx:
        rows += 1
        row_heights.append(0.2)
        subplot_titles.append("ADX (14)")

    # Normalize row heights so they sum to 1
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Row 1: Candlestick + overlays
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            showlegend=False,
        ),
        row=1, col=1,
    )
    _add_overlays(fig, df, ind)

    # Row 2: Volume
    _add_volume(fig, df)

    # Remaining rows: RSI / MACD / ATR / ADX (only if data exists)
    current_row = 3
    if show_rsi:
        _add_rsi(fig, df, current_row)
        current_row += 1
    if show_macd:
        _add_macd(fig, df, current_row)
        current_row += 1
    if show_atr:
        _add_atr(fig, df, current_row)
        current_row += 1
    if show_adx:
        _add_adx(fig, df, current_row)

    extra = sum([show_rsi, show_macd, show_atr, show_adx])
    chart_height = int(CHART_BASE_HEIGHT + extra * CHART_SUBCHART_HEIGHT)

    # Dark theme
    grid_color = "#2a2a4a"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(size=14),
        xaxis_rangeslider_visible=False,
        height=chart_height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Apply grid color to all axes
    fig.update_xaxes(gridcolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color)

    return fig


def chart_to_base64_png(fig: go.Figure, ind: Indicators, df: pd.DataFrame) -> str:
    extra = sum([
        ind.rsi and df["RSI"].notna().any(),
        ind.macd and df["MACD"].notna().any(),
        ind.atr and df["ATR"].notna().any(),
        ind.adx and df["ADX"].notna().any(),
    ])
    img_height = int(CHART_BASE_HEIGHT + extra * CHART_SUBCHART_HEIGHT)
    img_bytes = fig.to_image(format="png", width=1200, height=img_height, scale=2)
    return base64.b64encode(img_bytes).decode("utf-8")


def _price_change_stats(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return (pct_change, avg_vol, latest_vol, vol_ratio) for the last bar."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    prev_close = prev["Close"]
    pct_change = ((latest["Close"] - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0
    avg_vol = df["Volume"].mean()
    latest_vol = latest["Volume"]
    vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 0
    return pct_change, avg_vol, latest_vol, vol_ratio


def _build_prompt_data_lines(
    symbol: str, period: str, df: pd.DataFrame, ind: Indicators,
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
    prev = df.iloc[-2] if len(df) > 1 else latest
    pct_change, avg_vol, latest_vol, vol_ratio = _price_change_stats(df)

    active = ind.active_labels()
    indicator_text = ", ".join(active) if active else "None"

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
        f"- Previous close: ${prev['Close']:.2f}",
        f"- Change: {pct_change:+.2f}%",
        f"- Period high: ${df['High'].max():.2f}",
        f"- Period low: ${df['Low'].min():.2f}",
        f"- Average volume: {avg_vol:,.0f}",
        f"- Latest volume: {latest_vol:,.0f} ({vol_ratio:.1f}x average)",
    ]

    # --- Actual indicator values ---
    lines.append("\nIndicator readings (latest bar):")
    if ind.sma:
        sma20 = latest.get("SMA_20")
        sma50 = latest.get("SMA_50")
        sma20_str = f"${sma20:.2f}" if pd.notna(sma20) else "N/A (insufficient data)"
        sma50_str = f"${sma50:.2f}" if pd.notna(sma50) else "N/A (insufficient data)"
        lines.append(f"- SMA 20: {sma20_str}, SMA 50: {sma50_str}")
        if pd.notna(sma20) and pd.notna(sma50):
            cross = "above" if sma20 > sma50 else "below"
            lines.append(f"  SMA 20 is {cross} SMA 50 ({'bullish' if cross == 'above' else 'bearish'} alignment)")
    if ind.ema:
        ema12 = latest.get("EMA_12")
        ema26 = latest.get("EMA_26")
        ema12_str = f"${ema12:.2f}" if pd.notna(ema12) else "N/A"
        ema26_str = f"${ema26:.2f}" if pd.notna(ema26) else "N/A"
        lines.append(f"- EMA 12: {ema12_str}, EMA 26: {ema26_str}")
        if pd.notna(ema12) and pd.notna(ema26):
            cross = "above" if ema12 > ema26 else "below"
            lines.append(f"  EMA 12 is {cross} EMA 26 ({'bullish' if cross == 'above' else 'bearish'} alignment)")
    if ind.bb:
        bb_upper = latest.get("BB_Upper")
        bb_lower = latest.get("BB_Lower")
        bb_mid = latest.get("BB_Mid")
        if pd.notna(bb_upper) and pd.notna(bb_lower):
            close = latest["Close"]
            lines.append(f"- Bollinger Bands: Upper ${bb_upper:.2f}, Mid ${bb_mid:.2f}, Lower ${bb_lower:.2f}")
            if close > bb_upper:
                lines.append("  Price is ABOVE upper band (overbought / breakout)")
            elif close < bb_lower:
                lines.append("  Price is BELOW lower band (oversold / breakdown)")
            elif bb_upper != bb_lower:
                pct_bb = (close - bb_lower) / (bb_upper - bb_lower) * 100
                lines.append(f"  Price is at {pct_bb:.0f}% of band width")
            else:
                lines.append("  Price is at mid-band (bands are flat)")
        else:
            lines.append("- Bollinger Bands: N/A (insufficient data)")
    if ind.rsi:
        rsi = latest.get("RSI")
        if pd.notna(rsi):
            rsi_label = " (overbought)" if rsi > 70 else " (oversold)" if rsi < 30 else ""
            lines.append(f"- RSI (14): {rsi:.1f}{rsi_label}")
        else:
            lines.append("- RSI (14): N/A (insufficient data)")
    if ind.macd:
        macd = latest.get("MACD")
        macd_sig = latest.get("MACD_Signal")
        macd_hist = latest.get("MACD_Hist")
        if pd.notna(macd) and pd.notna(macd_sig):
            cross = "above" if macd > macd_sig else "below"
            lines.append(f"- MACD: {macd:.4f}, Signal: {macd_sig:.4f}, Histogram: {macd_hist:.4f}")
            lines.append(f"  MACD is {cross} signal line ({'bullish' if cross == 'above' else 'bearish'})")
        else:
            lines.append("- MACD: N/A (insufficient data)")
    if ind.atr:
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
    if ind.adx:
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
    has_fundamentals = fundamentals and any(v is not None for v in fundamentals.values())
    if has_fundamentals:
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
    for idx, row in recent.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        lines.append(
            f"{date_str} | ${row['Open']:.2f} | ${row['High']:.2f} | "
            f"${row['Low']:.2f} | ${row['Close']:.2f} | {row['Volume']:,.0f}"
        )

    lines.append(f"\nVisible indicators on chart: {indicator_text}")

    return lines


def build_observation_messages(
    symbol: str, period: str, df: pd.DataFrame, ind: Indicators,
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
        symbol, period, df, ind,
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
    symbol: str, period: str, df: pd.DataFrame, ind: Indicators,
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
        active = ind.active_labels()
        indicator_text = ", ".join(active) if active else "None"
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
            symbol, period, df, ind,
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
def fetch_ollama_models() -> list[dict]:
    """Fetch all models with parameter size and capabilities from Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except Exception:
        logger.warning("Failed to fetch Ollama model list", exc_info=True)
        return []

    def _fetch_model_info(m: dict) -> dict | None:
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/show",
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
            return {
                "name": m["name"],
                "parameter_size": param_size,
                "disk_size": disk_size,
                "capabilities": capabilities,
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
    """
    return text.replace("$", "\\$").replace("~~", "\\~\\~")


def _unescape_markdown(text: str) -> str:
    """Reverse _escape_markdown so raw text can be re-used in prompts."""
    return text.replace("\\~\\~", "~~").replace("\\$", "$")


def stream_ollama_response(
    model: str,
    system_prompt: str,
    user_prompt: str,
    images_b64: list[str] | None = None,
    temperature: float = 0.4,
):
    """Generator that yields text chunks from the Ollama /api/chat endpoint.

    Some models (e.g. Qwen3) use a "thinking" mode where initial tokens
    arrive in a ``thinking`` field instead of ``content``.  We buffer those
    thinking tokens and only emit them as a fallback if the model finishes
    without producing any regular content.
    """
    user_message: dict = {"role": "user", "content": user_prompt}
    if images_b64:
        user_message["images"] = images_b64
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            user_message,
        ],
        "stream": True,
        "options": {"temperature": temperature},
    }
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=(10, 600)
    ) as resp:
        resp.raise_for_status()
        thinking_buf: list[str] = []
        has_content = False
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                msg = data.get("message", {})
                token = msg.get("content", "")
                if token:
                    has_content = True
                    yield _escape_markdown(token)
                else:
                    thinking_token = msg.get("thinking", "")
                    if thinking_token:
                        thinking_buf.append(thinking_token)
                if data.get("done", False):
                    if not has_content and thinking_buf:
                        logger.info("Model %s produced only thinking output; using as fallback", model)
                        yield _escape_markdown("".join(thinking_buf))
                    elif not has_content and not thinking_buf:
                        logger.warning("Model %s returned done with no content or thinking tokens", model)
                    return


def _run_ollama_pass(
    model: str,
    system_prompt: str,
    user_prompt: str,
    images_b64: list[str] | None = None,
    label: str = "",
) -> tuple[str | None, str | None]:
    """Run a streaming Ollama pass, returning (result, error).

    Handles ConnectionError, HTTPError, and unexpected exceptions with
    user-friendly messages. Returns the streamed text on success, or an
    error string on failure.
    """
    prefix = f"{label}: " if label else ""
    try:
        result = st.write_stream(
            stream_ollama_response(model, system_prompt, user_prompt, images_b64)
        )
        if not result or not result.strip():
            # Some models (e.g. Qwen3-VL) fail silently with multiple images.
            # Retry with only the first image before giving up.
            if images_b64 and len(images_b64) > 1:
                logger.info("Model %s returned empty with %d images; retrying with 1 image", model, len(images_b64))
                result = st.write_stream(
                    stream_ollama_response(model, system_prompt, user_prompt, images_b64[:1])
                )
            if not result or not result.strip():
                logger.warning("Model %s returned an empty response", model)
                return None, f"{prefix}Model returned an empty response."
        return result, None
    except requests.ConnectionError:
        logger.warning("Cannot connect to Ollama at %s", OLLAMA_BASE_URL, exc_info=True)
        return None, (
            f"{prefix}Cannot connect to Ollama at `{OLLAMA_BASE_URL}`. "
            "Make sure Ollama is running."
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
    ind: Indicators,
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
    active = ind.active_labels()
    indicator_text = ", ".join(active) if active else "None"

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
    symbol: str, df: pd.DataFrame, ind: Indicators
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a brief watchlist scan."""
    system_prompt = (
        "You are a stock screener assistant. Provide a brief, structured assessment "
        "of the chart. Be direct and concise — this is a quick scan, not a deep analysis."
    )

    latest = df.iloc[-1]
    pct_change, avg_vol, latest_vol, vol_ratio = _price_change_stats(df)

    lines = [
        f"Quick scan for {symbol.upper()}.\n",
        f"- Close: ${latest['Close']:.2f} ({pct_change:+.2f}%)",
        f"- Period high/low: ${df['High'].max():.2f} / ${df['Low'].min():.2f}",
        f"- Volume: {latest_vol:,.0f} ({vol_ratio:.1f}x avg)",
    ]

    if ind.sma:
        sma20 = latest.get("SMA_20")
        sma50 = latest.get("SMA_50")
        if pd.notna(sma20) and pd.notna(sma50):
            lines.append(f"- SMA 20: ${sma20:.2f}, SMA 50: ${sma50:.2f}")
    if ind.rsi:
        rsi = latest.get("RSI")
        if pd.notna(rsi):
            lines.append(f"- RSI: {rsi:.1f}")
    if ind.macd:
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
        with open(fd, "w") as f:
            json.dump(history, f, indent=2)
        Path(tmp_path).replace(ANALYSIS_HISTORY_FILE)
    except BaseException:
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
    st.session_state.watchlist_text = st.session_state.watchlist_text.upper()
    st.session_state.watchlist_done = False
    st.session_state.watchlist_results = {}
    st.session_state.watchlist_step = 0
    st.session_state.watchlist_symbols = []


def _on_shared_input_change():
    _on_input_change()
    _on_watchlist_input_change()


def _uppercase_symbol():
    st.session_state.symbol = st.session_state.symbol.strip().upper()
    _on_input_change()


locked = st.session_state.analyzing or st.session_state.watchlist_analyzing

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
    watchlist_text = st.sidebar.text_area(
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

# Sidebar: indicator toggles
st.sidebar.header("Technical Indicators")
show_sma = st.sidebar.checkbox("SMA (20, 50)", value=True, on_change=_on_shared_input_change,
                               disabled=locked)
show_ema = st.sidebar.checkbox("EMA (12, 26)", value=True, on_change=_on_shared_input_change,
                               disabled=locked)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=True, on_change=_on_shared_input_change,
                              disabled=locked)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True, on_change=_on_shared_input_change,
                               disabled=locked)
show_macd = st.sidebar.checkbox("MACD", value=True, on_change=_on_shared_input_change,
                                disabled=locked)
show_atr = st.sidebar.checkbox("ATR (14)", value=True, on_change=_on_shared_input_change,
                               disabled=locked)
show_adx = st.sidebar.checkbox("ADX (14)", value=True, on_change=_on_shared_input_change,
                               disabled=locked)

ind = Indicators(
    sma=show_sma,
    ema=show_ema,
    bb=show_bb,
    rsi=show_rsi,
    macd=show_macd,
    atr=show_atr,
    adx=show_adx,
)

st.sidebar.header("Analysis Settings")
send_images_both_passes = st.sidebar.checkbox(
    "Send chart to both passes",
    value=True,
    disabled=locked,
    help="When enabled, chart images are sent to both the observation and synthesis passes. "
         "Disabling halves vision inference cost but may reduce quality.",
)

# Sidebar: model selection + action button
st.sidebar.divider()
available_models = fetch_ollama_models()
vision_models = [m for m in available_models if "vision" in m.get("capabilities", [])]

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
        s.strip().upper()
        for s in raw_symbols.replace("\n", ",").split(",")
        if s.strip()
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


# ── Main Content ─────────────────────────────────────────────────────────────

if is_single_mode and symbol:
    with st.spinner("Fetching market data..."):
        df = fetch_stock_data(symbol, period)
        company_name = fetch_company_name(symbol)
        fundamentals = fetch_fundamentals(symbol)
        earnings_info = fetch_next_earnings(symbol)
        market_ctx = fetch_market_context(period)
        news_headlines = fetch_news_headlines(symbol)

    if df is None or df.empty:
        st.error(f"No data found for **{symbol}**. Check the symbol and try again.")
    else:
        support_levels, resistance_levels = compute_support_resistance(df)
        chart_title = f"{company_name} ({symbol.upper()})"
        fig = build_candlestick_chart(df, symbol, ind, title=chart_title)
        st.plotly_chart(fig, width="stretch", height=int(fig.layout.height))

        # Determine strategic period
        strategic_period = STRATEGIC_PERIOD_MAP.get(period)

        if st.session_state.analyzing:
            step = st.session_state.get("analysis_step", 0)
            models = st.session_state.get("analysis_models", [])
            consensus_model = st.session_state.get("consensus_model_name")
            total = len(models)

            # Capture chart image(s) once on first step
            if st.session_state.chart_b64 is None:
                with st.spinner("Capturing chart..."):
                    try:
                        st.session_state.chart_b64 = chart_to_base64_png(fig, ind, df)
                    except Exception as e:
                        st.session_state.analyzing = False
                        st.session_state.done = True
                        st.session_state.ai_errors = {"_chart": f"Chart export failed: {e}"}
                        st.rerun()

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
                        strategic_fig = build_candlestick_chart(
                            strategic_df, symbol, ind, title=strategic_title
                        )
                        st.session_state.strategic_chart_b64 = chart_to_base64_png(
                            strategic_fig, ind, strategic_df
                        )
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
                symbol=symbol, period=period, df=df, ind=ind,
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
                    with st.status("Pass 1/2: Observing...", expanded=False) as status:
                        obs_text, obs_error = _run_ollama_pass(
                            current_model, obs_system, obs_user, images_b64,
                            label="Observation pass",
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
                        syn_images = images_b64 if send_images_both_passes else None
                        result, syn_error = _run_ollama_pass(
                            current_model, syn_system, syn_user, syn_images,
                            label="Synthesis pass",
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
                    all_model_info = fetch_ollama_models()
                    model_sizes = {
                        m["name"]: m.get("parameter_size", "unknown")
                        for m in all_model_info
                    }
                    consensus_sys, consensus_user = build_consensus_messages(
                        symbol, successful, model_sizes, df, ind
                    )
                    result, con_error = _run_ollama_pass(
                        consensus_model, consensus_sys, consensus_user,
                        label="Consensus",
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
                md_parts = [f"# {symbol.upper()} Analysis ({period})\n"]
                md_parts.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                for model_name in st.session_state.get("analysis_models", []):
                    if model_name in ai_outputs:
                        md_parts.append(f"\n## {model_name}\n")
                        md_parts.append(_unescape_markdown(ai_outputs[model_name]))
                if consensus_output:
                    md_parts.append("\n## Consensus Summary\n")
                    md_parts.append(_unescape_markdown(consensus_output))
                md_text = "\n".join(md_parts)
                date_str = datetime.now().strftime("%Y%m%d")
                st.download_button(
                    "Download Analysis",
                    data=md_text,
                    file_name=f"{symbol.upper()}_analysis_{date_str}.md",
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
                except Exception:
                    logger.warning("Failed to save analysis history", exc_info=True)
                st.session_state.history_saved = True

            # Show analysis history
            history = load_analysis_history(symbol)
            if history:
                with st.expander("Analysis History", expanded=False):
                    for entry in reversed(history):
                        ts = entry.get("timestamp", "")
                        models_str = ", ".join(entry.get("models", []))
                        st.markdown(f"**{ts[:16]}** — {entry.get('period', '')} — {models_str}")
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
                    wl_fig = build_candlestick_chart(wl_df, current_sym, ind, title=wl_title)
                    wl_img = chart_to_base64_png(wl_fig, ind, wl_df)
                    sys_prompt, usr_prompt = build_watchlist_prompt(current_sym, wl_df, ind)
                    with st.status(f"Scanning {current_sym} ({wl_step}/{wl_total})...", expanded=False) as wl_status:
                        analysis_text, wl_error = _run_ollama_pass(
                            wl_model, sys_prompt, usr_prompt, [wl_img],
                            label=current_sym,
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
