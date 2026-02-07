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
    vwap: bool = True
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
        if self.vwap:
            labels.append("VWAP")
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
    bb_std = close.rolling(window=20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

    # VWAP
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (typical_price * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum()
    df["VWAP"] = cum_tp_vol / cum_vol.replace(0, np.nan)

    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
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

    if ind.vwap:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                       line=dict(width=1.5, color="blue", dash="dashdot")),
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


def build_candlestick_chart(df: pd.DataFrame, symbol: str, ind: Indicators) -> go.Figure:
    # Determine subplot layout
    rows = 2
    row_heights = [0.5, 0.15]
    subplot_titles = [f"{symbol.upper()} Price", "Volume"]

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


def build_analysis_prompt(symbol: str, df: pd.DataFrame, ind: Indicators) -> str:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    pct_change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

    active = ind.active_labels()
    indicator_text = ", ".join(active) if active else "None"

    return (
        f"You are a stock market technical analyst. Analyze this candlestick chart for {symbol.upper()}.\n\n"
        f"Key data points:\n"
        f"- Latest close: ${latest['Close']:.2f}\n"
        f"- Previous close: ${prev['Close']:.2f}\n"
        f"- Change: {pct_change:+.2f}%\n"
        f"- Period high: ${df['High'].max():.2f}\n"
        f"- Period low: ${df['Low'].min():.2f}\n"
        f"- Average volume: {df['Volume'].mean():,.0f}\n\n"
        f"Visible indicators on the chart: {indicator_text}\n\n"
        f"Analyze the chart image and the visible technical indicators. "
        f"Identify trends, support/resistance levels, candlestick patterns, "
        f"and indicator signals. Provide a short-term outlook. Be concise."
    )


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


def stream_ollama_response(model: str, prompt: str, image_b64: str | None = None):
    """Generator that yields text chunks from the Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    if image_b64 is not None:
        payload["images"] = [image_b64]
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate", json=payload, stream=True, timeout=(10, 600)
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
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


def build_consensus_prompt(
    symbol: str,
    model_analyses: dict[str, str],
    df: pd.DataFrame,
    ind: Indicators,
) -> str:
    """Build a prompt for the consensus summarizer model."""
    latest = df.iloc[-1]
    active = ind.active_labels()
    indicator_text = ", ".join(active) if active else "None"

    analyses_text = ""
    for model_name, analysis in model_analyses.items():
        analyses_text += f"\n--- {model_name} ---\n{_unescape_markdown(analysis)}\n"

    return (
        f"You are a financial analysis synthesizer. Below are independent technical analyses "
        f"of a candlestick chart for {symbol.upper()} (latest close: ${latest['Close']:.2f}) "
        f"with these indicators: {indicator_text}.\n\n"
        f"Individual analyses:{analyses_text}\n"
        f"Synthesize these analyses into a consensus report with exactly these four sections. "
        f"Refer to each model by its name (e.g. {', '.join(model_analyses.keys())}) rather than "
        f"generic labels like 'Model 1'.\n\n"
        f"**Consensus Level**: Rate as Strong / Moderate / Mixed / Contradictory based on "
        f"how well the analyses agree.\n\n"
        f"**Points of Agreement**: Key findings where the analysts concur.\n\n"
        f"**Points of Disagreement**: Areas where the analysts diverge or contradict.\n\n"
        f"**Synthesized Outlook**: Your overall assessment combining all perspectives, "
        f"noting confidence level. Be concise."
    )


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
show_vwap = st.sidebar.checkbox("VWAP", value=True, on_change=_on_input_change,
                                disabled=locked)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True, on_change=_on_input_change,
                               disabled=locked)
show_macd = st.sidebar.checkbox("MACD", value=True, on_change=_on_input_change,
                                disabled=locked)

ind = Indicators(
    sma=show_sma,
    ema=show_ema,
    bb=show_bb,
    vwap=show_vwap,
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
    selected_vision_names = st.sidebar.multiselect(
        "Vision models",
        options=vision_names,
        default=[],
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

    if df is None or df.empty:
        st.error(f"No data found for **{symbol}**. Check the symbol and try again.")
    else:
        fig = build_candlestick_chart(df, symbol, ind)
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
                prompt = build_analysis_prompt(symbol, df, ind)

                st.subheader(f"AI Analysis - {current_model} ({step}/{total})")
                try:
                    result = st.write_stream(
                        stream_ollama_response(current_model, prompt, image_b64)
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
                consensus_prompt = build_consensus_prompt(symbol, successful, df, ind)
                try:
                    result = st.write_stream(
                        stream_ollama_response(consensus_model, consensus_prompt)
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
