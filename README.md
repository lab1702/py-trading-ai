# Stock Chart AI Analyzer

A web-based stock technical analysis tool that combines interactive candlestick charting with AI-powered analysis using a local Ollama model.

## Features

- **Interactive candlestick charts** for any publicly-traded stock via Yahoo Finance
- **Technical indicators** — toggle on/off:
  - SMA (20, 50)
  - EMA (12, 26)
  - Bollinger Bands
  - VWAP
  - RSI (14-period)
  - MACD (12, 26, 9)
- **AI-powered chart analysis** using a local Ollama vision model, including trend analysis, support/resistance levels, candlestick pattern recognition, and indicator signals
- **Dynamic model selector** — automatically detects vision-capable Ollama models and lets you choose from a dropdown
- **Chart export** to PNG (1200x600+, 2x scale)
- **Streaming AI responses** for a responsive experience
- **5-minute data caching** to reduce redundant API calls

## Prerequisites

- Python 3.13+
- [Ollama](https://ollama.com/) running locally with at least one vision-capable model pulled. We recommend trying `gemma3` first:
  ```bash
  ollama pull gemma3
  ```
  Other vision-capable models (e.g. `llava`, `llama3.2-vision`) will also work — the app automatically detects and lists all available vision models.

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
4. Choose a vision-capable Ollama model from the dropdown
5. Click **Analyze with AI** to get a streamed AI analysis of the chart

## Tech Stack

| Component | Library |
|---|---|
| Web UI | Streamlit |
| Charting | Plotly |
| Market data | yfinance |
| Data processing | Pandas, NumPy |
| Image export | Kaleido |
| AI analysis | Ollama (any vision-capable model) |
