from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
from fredapi import Fred
import pandas as pd
import requests
import datetime
from google_news_feed import GoogleNewsFeed 
# 或者简单使用 feedparser 解析 RSS，为了稳定性，下文新闻部分我写了纯 RSS 解析版

app = FastAPI()

# --- 配置区 ---
# 去 https://fred.stlouisfed.org/api_key 申请免费 Key
FRED_API_KEY = '1014ec3cfff8ba1a3b5fe862130d5887' 
fred = Fred(api_key=FRED_API_KEY)

# --- 1. 市场行情 & 技术指标 (Yahoo Finance) ---
def get_market_data():
    tickers = {
        "SPX": "^GSPC",
        "NDX": "^NDX",
        "10Y_Yield": "^TNX",
        "Gold": "GC=F",
        "Crude_Oil": "CL=F",
        "DXY": "DX-Y.NYB",
        "VIX": "^VIX"
    }
    
    data_summary = []
    
    for name, symbol in tickers.items():
        try:
            # 获取最近2个月数据以计算MA20
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2mo")
            
            if hist.empty:
                continue

            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change_pct = ((current_price - prev_price) / prev_price) * 100
            
            # 计算技术指标
            # MA20
            ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            status_ma = "ABOVE" if current_price > ma20 else "BELOW"
            
            # RSI (简单版14天)
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            data_summary.append({
                "Asset": name,
                "Price": round(current_price, 2),
                "Change_%": round(change_pct, 2),
                "MA20_Status": status_ma,
                "RSI": round(rsi, 2)
            })
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            
    return data_summary

# --- 2. 宏观流动性 (FRED) ---
def get_macro_data():
    try:
        # 联邦基金利率
        fed_funds = fred.get_series('FEDFUNDS', limit=1).iloc[-1]
        # 隔夜逆回购 (流动性指标) - 需用正确的Series ID，这里以 RRPONTSYD 为例
        # 注意：FRED部分数据有延迟，取最新可用即可
        return {
            "Fed_Funds_Rate": f"{fed_funds}%",
            "Liquidity_Note": "Check 10Y Yield in Market Data for real-time liquidity stress."
        }
    except Exception as e:
        return {"Error": str(e)}

# --- 3. 情绪指标 (Fear & Greed) ---
def get_fear_greed():
    # CNN 官方没有公开API，但这个Endpoint目前可用
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        # 获取最新评分
        score = int(data['fear_and_greed']['score'])
        rating = data['fear_and_greed']['rating']
        return {"Score": score, "Rating": rating}
    except Exception as e:
        return {"Score": "N/A", "Error": "Failed to fetch CNN data, using VIX instead."}

# --- 4. 宏观新闻 (RSS / Google News) ---
import feedparser

def get_macro_news():
    # 使用 Yahoo Finance 的宏观新闻 RSS 或者 Google News 搜索特定关键词
    # 关键词：US Economy, Federal Reserve, Inflation
    rss_url = "https://news.google.com/rss/search?q=US+Economy+Federal+Reserve+Inflation+when:3d&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    news_items = []
    
    # 取前 5 条最相关的新闻
    for entry in feed.entries[:5]:
        news_items.append({
            "Title": entry.title,
            "Source": entry.source.title if hasattr(entry, 'source') else "Google News",
            "Published": entry.published,
            "Link": entry.link
        })
    return news_items

# --- 汇总接口 (供 Dify 调用) ---
@app.get("/daily_analysis_context")
async def get_full_context():
    print("Fetching Market Data...")
    market = get_market_data()
    
    print("Fetching Macro Data...")
    macro = get_macro_data()
    
    print("Fetching Sentiment...")
    sentiment = get_fear_greed()
    
    print("Fetching News...")
    news = get_macro_news()
    
    # 组装成 LLM 易读的 Markdown 格式
    return {
        "status": "success",
        "data_json": {
            "market": market,
            "macro": macro,
            "sentiment": sentiment,
            "news": news
        },
        # 预处理好的 Prompt Context，直接喂给 Dify
        "llm_context_text": f"""
=== MARKET DATA (Technical & Price) ===
{pd.DataFrame(market).to_markdown(index=False)}

=== LIQUIDITY & MACRO ===
- Fed Funds Rate: {macro.get('Fed_Funds_Rate')}
- 10Y Yield Trend: See '10Y_Yield' in Market Data.

=== SENTIMENT ===
- CNN Fear & Greed Index: {sentiment.get('Score')} ({sentiment.get('Rating')})
- VIX Level: See 'VIX' in Market Data.

=== KEY MACRO NEWS (Last 3 Days) ===
{chr(10).join([f"- {n['Title']} (Source: {n['Source']})" for n in news])}
        """
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)