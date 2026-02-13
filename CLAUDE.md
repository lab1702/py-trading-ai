# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stock Chart AI Analyzer — a Streamlit web app that displays interactive candlestick charts (via Plotly/yfinance) and sends chart screenshots to one or more local Ollama vision models for AI-powered technical analysis, with optional multi-model consensus summaries. Supports single-symbol deep analysis and a watchlist mode for quick multi-symbol scans.

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

This is a single-file application (`app.py`, ~1750 lines). All logic lives there, organized into layers:

### Data Layer
- `fetch_stock_data()` pulls OHLCV data from Yahoo Finance via `yfinance`, cached 5 min.
- `_compute_indicators()` adds technical indicator columns (SMA, EMA, Bollinger Bands, RSI, MACD, ATR, ADX) to the DataFrame in-place. ADX has a 28-bar warmup period blanked to NaN.
- `fetch_company_name()`, `fetch_fundamentals()`, `fetch_next_earnings()`, `fetch_news_headlines()`, `fetch_market_context()` — supplementary data fetchers, all cached 5 min. These run in parallel via `ThreadPoolExecutor` on cold cache to reduce initial load time.
- `compute_support_resistance()` finds support/resistance levels from local extrema using a sliding window (`>=`/`<=` comparisons to avoid float-equality issues), then clusters nearby levels. Returned levels are sorted ascending by price for clarity in prompts.
- Symbol input is sanitized via `_SYMBOL_RE` (strips characters outside `A-Za-z0-9.\-^=`) before being passed to yfinance — applies to both single-symbol and watchlist modes. Watchlist parsing filters out symbols that become empty after sanitization.
- `_price_change_stats()` returns `(prev_close, pct_change, avg_vol, latest_vol, vol_ratio)` — callers use the returned `prev_close` rather than recomputing it.

### Chart Layer
- `build_candlestick_chart()` constructs a Plotly `Figure` with dynamic subplots — the number of rows varies based on which indicators (RSI, MACD, ATR, ADX) are enabled and have valid data. Helper functions `_add_overlays`, `_add_volume`, `_add_rsi`, `_add_macd`, `_add_atr`, `_add_adx` each add traces to specific subplot rows.
- `chart_to_base64_png(fig)` exports the figure to a base64-encoded PNG using Kaleido (1200px width, 2x scale). It reads height directly from `fig.layout.height` to stay in sync with `build_candlestick_chart()`.

### AI Layer
- `fetch_ollama_models()` queries the local Ollama API (`/api/tags` + `/api/show` in parallel via ThreadPoolExecutor) to discover all models and their capabilities, cached 30 sec.
- **Two-pass analysis**: Single-symbol analysis runs two Ollama calls per vision model:
  1. **Observation pass** (`build_observation_messages`) — model lists factual observations only (no predictions).
  2. **Synthesis pass** (`build_analysis_messages`) — model receives its own observations and produces a structured analysis with Trend, Support/Resistance, Indicator Signals, Candlestick Patterns, Outlook, and Risk Factors sections.
- `_build_prompt_data_lines()` assembles shared data context (price, indicator values, fundamentals, earnings, market context, news, support/resistance, recent price action) used by both prompts.
- `stream_ollama_response()` sends messages to `/api/chat` and yields streamed text tokens. It also handles "thinking" models (e.g. Qwen3) that emit tokens in a `thinking` field instead of `content`, falling back to thinking output if no content is produced. Malformed JSON lines from the stream are logged and skipped. Mid-stream errors from Ollama (JSON objects with an `error` field) are raised as `RuntimeError` so callers see the actual error message. `_run_ollama_pass()` wraps this with Streamlit's `st.write_stream` and error handling; if a model returns empty with multiple images, it automatically retries with only the primary chart image (some models like qwen3-vl fail silently with multi-image input). Connection errors, timeouts, and HTTP errors each have specific user-facing messages.
- `build_consensus_messages()` constructs a synthesis prompt with all individual analyses for the consensus model.
- `build_watchlist_prompt()` produces a brief scan prompt for watchlist mode (single pass, no observations step).
- **Strategic period charts**: `STRATEGIC_PERIOD_MAP` maps short periods to longer ones (e.g., "1mo" → "1y"). When available, a second chart image for the strategic period is sent alongside the primary chart to give the model longer-term context.

### Markdown Escaping
- `_escape_markdown()` escapes `$` and `~~` in Ollama output to prevent Streamlit from rendering LaTeX or strikethrough.
- `_unescape_markdown()` reverses this when raw text is needed (e.g., for prompts or file export).
- Caveat: the pair is not a true round-trip for text that already contains literal `\$` or `\~\~`, but this is acceptable since model output virtually never contains those sequences.

### Persistence
- `save_analysis()` / `load_analysis_history()` persist analyses to `analysis_history.json` (capped at 100 entries, atomic writes via temp file + rename). The temp-file cleanup catches `Exception` (not `BaseException`) so `KeyboardInterrupt`/`SystemExit` propagate cleanly. The `history_saved` flag is only set on successful save, allowing retry on transient failures. This file is gitignored.

### UI Layer (bottom of file)
- Two modes selectable via sidebar radio: **Single Symbol** and **Watchlist**.
- Uses `st.session_state` with `analyzing`/`done` flags and a step-based pipeline (`analysis_step`) to process one model per `st.rerun()` cycle. Inputs are locked during analysis via a `locked` flag. A "Stop after current step" button appears in the sidebar during analysis, allowing users to cancel between rerun cycles (shows a toast notification on stop).
- Single mode: vision model multiselect, consensus model selector (enabled at 2+ vision models), "Send chart to both passes" toggle.
- Watchlist mode: single vision model selector, processes symbols sequentially one per rerun.
- With a single vision model, output renders directly (no expanders, no consensus). With multiple models, individual analyses appear in collapsed expanders followed by a consensus summary.
- Both modes offer a "Download" button to export results as Markdown (single-symbol analysis or watchlist scan).

## Key Design Decisions

- Ollama base URL is hardcoded as `OLLAMA_BASE_URL = "http://localhost:11434"` at the top of the file.
- The `Indicators` dataclass centralizes which indicators are active and provides `active_labels()` for prompt construction.
- Chart height is dynamic: base 500px + 200px per sub-chart (RSI/MACD/ATR/ADX). Only sub-charts with valid data are shown.
- The app uses `st.rerun()` after each pipeline step to manage Streamlit's execution model. Analysis runs one model per rerun via `analysis_step`, with the consensus summarizer as the final step.
- Users must select vision models explicitly (no defaults). A "Select all vision models" checkbox provides a shortcut.
- The `send_images_both_passes` toggle controls whether chart images are sent to both passes or only the observation pass, trading quality for halved vision inference cost.

## Dependencies

Python 3.13+. Key packages: `streamlit`, `plotly`, `yfinance`, `requests`, `kaleido` (for Plotly image export), `lxml`. NumPy and Pandas are transitive deps used directly.

- `logging.basicConfig()` is called at module level so that `logger.warning()` / `logger.info()` calls throughout the app produce visible output.
