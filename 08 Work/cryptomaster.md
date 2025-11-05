Here’s a **comprehensive MVP prompt** in Markdown that you can pass directly to your AI agent. It clearly defines the end-goal, tech stack, requirements, and output format so your agent knows exactly how to build your crypto trading helper.

---

# 🚀 Crypto Overnight Trading Helper – MVP Build Prompt

## 🎯 Goal

Build a **crypto overnight trading assistant** that:

- Runs **locally on Linux Mint** (Python-based).
    
- Executes **after 9 PM IST** and closes trades by **7 AM IST**.
    
- Selects the **Top 5 crypto pairs** to invest in overnight.
    
- Provides **entry, exit, stop-loss, and leverage-adjusted targets**.
    
- Uses **5x leverage** in risk/return calculations.
    
- Outputs results **both in terminal and via Telegram bot**.
    

---

## 🛠️ Tech Stack

- **Python 3.10+**
    
- **TA-Lib** → Technical indicators (EMA, RSI, MACD, ATR).
    
- **vectorbt** → Backtesting, strategy evaluation.
    
- **vaderSentiment** → Sentiment analysis for crypto news & tweets.
    
- **NewsAPI** → News headlines feed for crypto sentiment.
    
- **Binance API (CCXT)** → Market data (OHLCV, volume).
    
- **python-telegram-bot** → Send daily trade signals to Telegram.
    
- **dotenv** → Load API keys/secrets from `.env`.
    

---

## 🔑 Environment Variables (`.env`)

```ini
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET=your_binance_secret
NEWSAPI_KEY=your_newsapi_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 📊 Data Flow

1. **Market Data Agent**
    
    - Pull top 50 liquid crypto pairs (USDT) from Binance using CCXT.
        
    - Fetch OHLCV data (15m + 1h + daily).
        
2. **Indicator Agent**
    
    - Calculate:
        
        - EMA (20, 50 crossover for trend)
            
        - RSI (30/70 levels for oversold/overbought)
            
        - MACD (momentum)
            
        - ATR (volatility for stop-loss).
            
    - Generate **entry/exit candidates**.
        
3. **News & Sentiment Agent**
    
    - Pull top 20 crypto headlines from NewsAPI.
        
    - Analyze sentiment using **vaderSentiment**.
        
    - Score coins based on positive/negative news.
        
4. **Signal Scoring Engine**
    
    - Combine **technical scores** + **sentiment score**.
        
    - Rank pairs and select **Top 5 overnight candidates**.
        
5. **Risk/Leverage Adjustment**
    
    - Position sizing with **5x leverage**.
        
    - Stop-loss = entry − (ATR × 1.5).
        
    - Target = entry + (ATR × 2–3).
        
    - Ensure no trade risks > 2% of account equity.
        
6. **Output Agent**
    
    - Format final results:
        
        ```
        📊 Overnight Signals (9PM–7AM IST)
        
        1. ETH/USDT
           Entry: 2520
           Exit: 2650
           Stop-Loss: 2470
           Leverage: 5x
           Sentiment: Bullish 👍
        
        2. BTC/USDT
           Entry: ...
        ```
        
    - Print to **terminal**.
        
    - Send formatted message to **Telegram channel**.
        

---

## ✅ End-to-End Flow

- Cronjob or systemd timer runs script daily at **9:05 PM IST**.
    
- Script fetches data → computes signals → ranks top 5 → outputs results.
    
- Trader checks Telegram in the morning to exit trades or adjust.
    

---

## 🔮 Stretch Goals (Optional Later)

- Add **Twitter sentiment** (via snscrape).
    
- Plug into **TradingView webhook** for alerts.
    
- Add **vectorbt backtest module** to refine thresholds.
    

---

## 📌 Deliverables

- A single Python project with:
    
    - `/src` folder containing modular agents (data, indicators, sentiment, scoring, output).
        
    - `.env` file support.
        
    - `main.py` to orchestrate flow.
        
    - Telegram + terminal output working.
        
- Installation docs (`requirements.txt`).
    
- Usage docs (how to run daily).
    

---

