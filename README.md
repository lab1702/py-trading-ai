# Stock Chart AI Analyzer

A web-based stock technical analysis tool that combines interactive candlestick charting with AI-powered analysis using local Ollama models, featuring two-pass analysis, multi-model consensus, and a watchlist scanner.

## Features

- **Interactive candlestick charts** for any publicly-traded stock via Yahoo Finance
- **Technical indicators** — toggle on/off:
  - SMA (20, 50)
  - EMA (12, 26)
  - Bollinger Bands (20, 2σ)
  - RSI (14-period)
  - MACD (12, 26, 9)
  - ATR (14-period)
  - ADX (14-period) with +DI/-DI
- **Two-pass AI analysis** — each vision model first lists factual observations (pass 1), then synthesizes a structured analysis from those observations (pass 2), producing more thorough and grounded results
- **Enriched context** — prompts include fundamentals (P/E, market cap, dividend yield), next earnings date, recent news headlines, S&P 500 market context, and algorithmically computed support/resistance levels
- **Strategic period charts** — automatically sends a longer-timeframe chart alongside the primary chart (e.g. 1y context for a 1mo analysis) to give the model broader perspective
- **Multi-model consensus** — select multiple vision models for independent analyses, then a consensus model synthesizes them into a report with agreement/disagreement points and an overall outlook
- **Watchlist mode** — scan multiple symbols quickly with a single vision model for at-a-glance trend, confidence, key signals, and outlook
- **Dynamic model selector** — automatically detects all Ollama models and their capabilities; vision models appear in a multiselect, all models are available as the consensus summarizer
- **Analysis history** — analyses are saved locally and viewable in the UI, with Markdown export/download
- **Chart export** to PNG (1200px width, dynamic height, 2x scale)
- **Streaming AI responses** for a responsive experience
- **5-minute data caching** to reduce redundant API calls

## Prerequisites

- Python 3.13+
- [Ollama](https://ollama.com/) running locally with at least one vision-capable model pulled. We recommend trying `gemma3` first:
  ```bash
  ollama pull gemma3
  ```
  Other vision-capable models (e.g. `llava`, `llama3.2-vision`) will also work — the app automatically detects and lists all available vision models. Any Ollama model (including text-only) can serve as the consensus summarizer.

## Installation

```bash
git clone https://github.com/lab1702/py-trading-ai
cd py-trading-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Make sure Ollama is running, then start the app:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Use the sidebar to switch between two modes:

### Single Symbol

1. Enter a stock symbol (e.g. `AAPL`)
2. Select a time period
3. Toggle technical indicators
4. Select one or more vision-capable models for chart analysis
5. Optionally select a consensus model (enabled when 2+ vision models are chosen) to synthesize all analyses
6. Click **Analyze with AI** to get streamed two-pass AI analysis of the chart

### Watchlist

1. Enter multiple symbols (comma or newline separated)
2. Select a time period and indicators
3. Select a single vision model
4. Click **Scan Watchlist** for a quick trend/outlook scan of each symbol

## Tech Stack

| Component | Library |
|---|---|
| Web UI | Streamlit |
| Charting | Plotly |
| Market data | yfinance |
| Data processing | Pandas, NumPy |
| Image export | Kaleido (0.2.x — self-contained, no system browser needed) |
| AI analysis | Ollama (vision models for chart analysis, any model for consensus) |
