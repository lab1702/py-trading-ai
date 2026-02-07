# Stock Chart AI Analyzer

A web-based stock technical analysis tool that combines interactive candlestick charting with AI-powered analysis using local Ollama models, featuring multi-model consensus analysis.

## Features

- **Interactive candlestick charts** for any publicly-traded stock via Yahoo Finance
- **Technical indicators** — toggle on/off:
  - SMA (20, 50)
  - EMA (12, 26)
  - Bollinger Bands
  - VWAP
  - RSI (14-period)
  - MACD (12, 26, 9)
- **AI-powered chart analysis** using local Ollama vision models, including trend analysis, support/resistance levels, candlestick pattern recognition, and indicator signals
- **Multi-model consensus** — select multiple vision models for independent analyses, then a consensus model synthesizes them into a report with agreement/disagreement points and an overall outlook
- **Dynamic model selector** — automatically detects all Ollama models and their capabilities; vision models appear in a multiselect, all models are available as the consensus summarizer
- **Chart export** to PNG (1200x600+, 2x scale)
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
git clone <repo-url>
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

The app opens at `http://localhost:8501`. Use the sidebar to:

1. Enter a stock symbol (e.g. `AAPL`)
2. Select a time period
3. Toggle technical indicators
4. Select one or more vision-capable models for chart analysis
5. Optionally select a consensus model (enabled when 2+ vision models are chosen) to synthesize all analyses
6. Click **Analyze with AI** to get streamed AI analysis of the chart

## Tech Stack

| Component | Library |
|---|---|
| Web UI | Streamlit |
| Charting | Plotly |
| Market data | yfinance |
| Data processing | Pandas, NumPy |
| Image export | Kaleido |
| AI analysis | Ollama (vision models for chart analysis, any model for consensus) |
