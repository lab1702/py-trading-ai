import base64
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class Indicators:
    sma: bool = True
    ema: bool = True
    bb: bool = True
    rsi: bool = True
    macd: bool = True

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
        return symbol.upper()


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


def _add_overlays(fig: go.Figure, df: pd.DataFrame, ind: Indicators) -> None:
    """Add overlay indicator traces to the price row."""
    if ind.sma:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20",
                       line=dict(width=1, color="orange")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50",
                       line=dict(width=1, color="purple")),
            row=1, col=1,
        )

    if ind.ema:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_12"], name="EMA 12",
                       line=dict(width=1, color="cyan", dash="dot")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_26"], name="EMA 26",
                       line=dict(width=1, color="magenta", dash="dot")),
            row=1, col=1,
        )

    if ind.bb:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                       line=dict(width=1, color="gray", dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                       line=dict(width=1, color="gray", dash="dash"),
                       fill="tonexty", fillcolor="rgba(128,128,128,0.1)"),
            row=1, col=1,
        )



def _add_volume(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add volume bar chart to row 2."""
    vol_colors = ["green" if c >= o else "red"
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
                   line=dict(width=1, color="purple")),
        row=row, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red",
                  line_width=1, row=row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green",
                  line_width=1, row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=row, col=1)


def _add_macd(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Add MACD sub-chart at the given row."""
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                   line=dict(width=1, color="blue")),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                   line=dict(width=1, color="orange")),
        row=row, col=1,
    )
    colors = ["green" if v >= 0 else "red" for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
               marker_color=colors, showlegend=False),
        row=row, col=1,
    )


def build_candlestick_chart(df: pd.DataFrame, symbol: str, ind: Indicators,
                            title: str | None = None) -> go.Figure:
    # Determine subplot layout
    rows = 2
    row_heights = [0.5, 0.15]
    price_title = title or f"{symbol.upper()} Price"
    subplot_titles = [price_title, "Volume"]

    if ind.rsi:
        rows += 1
        row_heights.append(0.2)
        subplot_titles.append("RSI (14)")
    if ind.macd:
        rows += 1
        row_heights.append(0.2)
        subplot_titles.append("MACD")

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

    # Remaining rows: RSI / MACD
    current_row = 3
    if ind.rsi:
        _add_rsi(fig, df, current_row)
        current_row += 1
    if ind.macd:
        _add_macd(fig, df, current_row)

    base_height = 500
    sub_chart_height = 200
    extra = (1 if ind.rsi else 0) + (1 if ind.macd else 0)
    chart_height = base_height + extra * sub_chart_height

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=chart_height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def chart_to_base64_png(fig: go.Figure, ind: Indicators) -> str:
    extra = (1 if ind.rsi else 0) + (1 if ind.macd else 0)
    img_height = 600 + extra * 200
    img_bytes = fig.to_image(format="png", width=1200, height=img_height, scale=2)
    return base64.b64encode(img_bytes).decode("utf-8")


def build_analysis_messages(
    symbol: str, period: str, df: pd.DataFrame, ind: Indicators
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
        "rather than forcing a directional prediction. Be concise."
    )

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    pct_change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
    avg_vol = df["Volume"].mean()
    latest_vol = latest["Volume"]
    vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 0

    active = ind.active_labels()
    indicator_text = ", ".join(active) if active else "None"

    # --- Core data ---
    lines = [
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
            else:
                pct_bb = (close - bb_lower) / (bb_upper - bb_lower) * 100
                lines.append(f"  Price is at {pct_bb:.0f}% of band width")
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
    lines.append("\nAnalyze both the chart image and the numeric data above.")

    return system_prompt, "\n".join(lines)


@st.cache_data(show_spinner=False, ttl=30)
def fetch_ollama_models() -> list[dict]:
    """Fetch all models with parameter size and capabilities from Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except Exception:
        return []

    result = []
    for m in models:
        try:
            show = requests.post(
                f"{OLLAMA_BASE_URL}/api/show",
                json={"model": m["name"]},
                timeout=5,
            ).json()
            capabilities = show.get("capabilities", [])
            param_size = show.get("details", {}).get("parameter_size", "")
            disk_bytes = m.get("size", 0)
            if disk_bytes >= 1 << 30:
                disk_size = f"{disk_bytes / (1 << 30):.1f} GB"
            else:
                disk_size = f"{disk_bytes / (1 << 20):.0f} MB"
            result.append({
                "name": m["name"],
                "parameter_size": param_size,
                "disk_size": disk_size,
                "capabilities": capabilities,
            })
        except Exception:
            continue
    return result


def _escape_markdown(text: str) -> str:
    """Escape characters that Streamlit's markdown renderer misinterprets."""
    return text.replace("$", "\\$").replace("_", "\\_")


def _unescape_markdown(text: str) -> str:
    """Reverse _escape_markdown so raw text can be re-used in prompts."""
    return text.replace("\\$", "$").replace("\\_", "_")


def stream_ollama_response(
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None = None,
    temperature: float = 0.4,
):
    """Generator that yields text chunks from the Ollama /api/chat endpoint."""
    user_message: dict = {"role": "user", "content": user_prompt}
    if image_b64 is not None:
        user_message["images"] = [image_b64]
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
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield _escape_markdown(token)
                if data.get("done", False):
                    return


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


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Stock Chart AI Analyzer", layout="wide")
st.title("Stock Chart AI Analyzer")

# State initialization
if "analyzing" not in st.session_state:
    st.session_state.analyzing = False
if "done" not in st.session_state:
    st.session_state.done = False

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
    st.session_state.pop("chart_b64", None)

def _uppercase_symbol():
    st.session_state.symbol = st.session_state.symbol.strip().upper()
    _on_input_change()

locked = st.session_state.analyzing

# Sidebar: symbol & period at the top
symbol = st.sidebar.text_input("Stock symbol", key="symbol", max_chars=10,
                               placeholder="e.g. AAPL", on_change=_uppercase_symbol,
                               disabled=locked)
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                              index=2, on_change=_on_input_change, disabled=locked)

# Sidebar: indicator toggles
st.sidebar.header("Technical Indicators")
show_sma = st.sidebar.checkbox("SMA (20, 50)", value=True, on_change=_on_input_change,
                               disabled=locked)
show_ema = st.sidebar.checkbox("EMA (12, 26)", value=True, on_change=_on_input_change,
                               disabled=locked)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=True, on_change=_on_input_change,
                              disabled=locked)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True, on_change=_on_input_change,
                               disabled=locked)
show_macd = st.sidebar.checkbox("MACD", value=True, on_change=_on_input_change,
                                disabled=locked)

ind = Indicators(
    sma=show_sma,
    ema=show_ema,
    bb=show_bb,
    rsi=show_rsi,
    macd=show_macd,
)

# Sidebar: analyze button
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
    button_label = "Done"
else:
    button_label = "Analyze with AI"

st.sidebar.divider()
available_models = fetch_ollama_models()
vision_models = [m for m in available_models if "vision" in m.get("capabilities", [])]

selected_vision_names: list[str] = []
selected_consensus_model: str | None = None

if vision_models:
    vision_labels = {m["name"]: _format_model_label(m) for m in vision_models}
    vision_names = list(vision_labels.keys())

    if "selected_vision" not in st.session_state:
        st.session_state.selected_vision = []

    all_selected = len(vision_names) > 0 and set(st.session_state.selected_vision) == set(vision_names)
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
button_disabled = locked or not symbol or not selected_vision_names or needs_consensus or st.session_state.done

if st.sidebar.button(
    button_label,
    type="primary",
    disabled=button_disabled,
    width="stretch",
):
    st.session_state.analyzing = True
    st.session_state.analysis_step = 1
    st.session_state.analysis_models = list(selected_vision_names)
    st.session_state.consensus_model_name = selected_consensus_model
    st.session_state.ai_outputs = {}
    st.session_state.ai_errors = {}
    st.session_state.consensus_output = ""
    st.session_state.consensus_error = ""
    st.rerun()

if symbol:
    with st.spinner("Fetching market data..."):
        df = fetch_stock_data(symbol, period)
        company_name = fetch_company_name(symbol)

    if df is None or df.empty:
        st.error(f"No data found for **{symbol}**. Check the symbol and try again.")
    else:
        chart_title = f"{company_name} ({symbol.upper()})"
        fig = build_candlestick_chart(df, symbol, ind, title=chart_title)
        st.plotly_chart(fig, width="stretch")

        if st.session_state.analyzing:
            step = st.session_state.get("analysis_step", 0)
            models = st.session_state.get("analysis_models", [])
            consensus_model = st.session_state.get("consensus_model_name")
            total = len(models)

            # Capture chart image once on first step
            if "chart_b64" not in st.session_state:
                with st.spinner("Capturing chart..."):
                    try:
                        st.session_state.chart_b64 = chart_to_base64_png(fig, ind)
                    except Exception as e:
                        st.session_state.analyzing = False
                        st.session_state.done = True
                        st.session_state.ai_errors["_chart"] = f"Chart export failed: {e}"
                        st.rerun()

            if "chart_b64" not in st.session_state:
                st.session_state.analyzing = False
                st.session_state.done = True
                st.rerun()

            image_b64 = st.session_state.chart_b64

            if step >= 1 and step <= total:
                # Vision model analysis step
                current_model = models[step - 1]
                system_prompt, user_prompt = build_analysis_messages(symbol, period, df, ind)

                st.subheader(f"AI Analysis - {current_model} ({step}/{total})")
                try:
                    result = st.write_stream(
                        stream_ollama_response(current_model, system_prompt, user_prompt, image_b64)
                    )
                    st.session_state.ai_outputs[current_model] = result
                except requests.ConnectionError:
                    st.session_state.ai_errors[current_model] = (
                        f"Cannot connect to Ollama at `{OLLAMA_BASE_URL}`. "
                        "Make sure Ollama is running."
                    )
                except requests.HTTPError as e:
                    st.session_state.ai_errors[current_model] = (
                        f"Ollama returned an error: {e}"
                    )
                except Exception as e:
                    st.session_state.ai_errors[current_model] = (
                        f"Unexpected error during analysis: {e}"
                    )

                st.session_state.analysis_step = step + 1

                # Check if we're done with vision models
                if step == total:
                    successful = st.session_state.ai_outputs
                    if len(successful) >= 2 and consensus_model:
                        # Move to consensus step
                        st.rerun()
                    else:
                        # Skip consensus: single model, no consensus model, or not enough successes
                        st.session_state.analyzing = False
                        st.session_state.done = True
                        st.rerun()
                else:
                    st.rerun()

            elif step == total + 1 and consensus_model:
                # Consensus summarizer step
                successful = st.session_state.ai_outputs
                st.subheader("Generating Consensus...")
                # Build model size lookup for weighting context
                all_model_info = fetch_ollama_models()
                model_sizes = {
                    m["name"]: m.get("parameter_size", "unknown")
                    for m in all_model_info
                }
                consensus_sys, consensus_user = build_consensus_messages(
                    symbol, successful, model_sizes, df, ind
                )
                try:
                    result = st.write_stream(
                        stream_ollama_response(consensus_model, consensus_sys, consensus_user)
                    )
                    st.session_state.consensus_output = result
                except requests.ConnectionError:
                    st.session_state.consensus_error = (
                        f"Cannot connect to Ollama at `{OLLAMA_BASE_URL}`. "
                        "Make sure Ollama is running."
                    )
                except requests.HTTPError as e:
                    st.session_state.consensus_error = (
                        f"Ollama returned an error: {e}"
                    )
                except Exception as e:
                    st.session_state.consensus_error = (
                        f"Unexpected error during consensus: {e}"
                    )
                finally:
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
