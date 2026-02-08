# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stock Chart AI Analyzer — a Streamlit web app that displays interactive candlestick charts (via Plotly/yfinance) and sends chart screenshots to one or more local Ollama vision models for AI-powered technical analysis, with optional multi-model consensus summaries.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the app (requires Ollama running locally with a vision-capable model)
streamlit run app.py
```

There is no test suite or linter configured.

## Architecture

This is a single-file application (`app.py`). All logic lives there:

- **Data layer**: `fetch_stock_data()` pulls OHLCV data from Yahoo Finance via `yfinance`, cached for 5 minutes (`@st.cache_data(ttl=300)`). `_compute_indicators()` adds technical indicator columns (SMA, EMA, Bollinger Bands, RSI, MACD) to the DataFrame in-place.
- **Chart layer**: `build_candlestick_chart()` constructs a Plotly `Figure` with dynamic subplots — the number of rows varies based on which indicators (RSI, MACD) are enabled. Helper functions `_add_overlays`, `_add_volume`, `_add_rsi`, `_add_macd` each add traces to specific subplot rows. `chart_to_base64_png()` exports the figure to a base64-encoded PNG using Kaleido.
- **AI layer**: `fetch_ollama_models()` queries the local Ollama API (`/api/tags` + `/api/show`) to discover all models and their capabilities, cached for 30 seconds. `stream_ollama_response()` sends a prompt (and optionally a chart image) to `/api/generate` and yields streamed text tokens. `build_analysis_prompt()` builds the vision analysis prompt with latest price data and active indicator names. `build_consensus_prompt()` constructs a synthesis prompt containing all individual analyses for the consensus model. `_format_model_label()` formats model display labels with size info and a `[vision]` tag.
- **UI layer** (bottom of file): Streamlit sidebar controls for symbol, period, indicator toggles, vision model multiselect, and consensus model selector. Uses `st.session_state` with `analyzing`/`done` flags and a step-based pipeline (`analysis_step`) to process one model per rerun cycle. Inputs are locked during analysis via a `locked` flag.

## Key Design Decisions

- Ollama base URL is hardcoded as `OLLAMA_BASE_URL = "http://localhost:11434"` at the top of the file.
- The `Indicators` dataclass centralizes which indicators are active and provides `active_labels()` for prompt construction.
- Chart height is dynamic: base 500px + 200px per sub-chart (RSI/MACD). Image export uses 2x scale at 1200px width.
- The app uses `st.rerun()` after each pipeline step to manage Streamlit's execution model. Analysis runs one model per rerun via `analysis_step`, with the consensus summarizer as the final step.
- Users must select vision models explicitly (no defaults). A "Select all vision models" checkbox provides a shortcut. The consensus model selector is disabled until 2+ vision models are chosen, and the analyze button is disabled until a consensus model is also selected.
- With a single vision model, the app behaves like before (direct output, no expanders, no consensus). With multiple models, individual analyses appear in collapsed expanders followed by a prominent consensus summary.

## Dependencies

Python 3.13+. Key packages: `streamlit`, `plotly`, `yfinance`, `requests`, `kaleido` (for Plotly image export). NumPy and Pandas are transitive deps used directly.
