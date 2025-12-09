
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import requests
from xml.etree import ElementTree as ET
from urllib.parse import quote_plus
import pandas as pd
import numpy as np
import torch

# --- NEW: Import Hugging Face Transformers ---
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("Transformers library not found. Please install: pip install transformers torch")
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
# ---------------------------------------------

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, newsapi_key: Optional[str] = None, twitter_bearer_token: Optional[str] = None):
        """
        Initialize Sentiment Analyzer.
        Loads the FinBERT model and tokenizer on startup.
        """
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY', '')
        self.twitter_bearer_token = twitter_bearer_token or os.getenv('TWITTER_BEARER_TOKEN', '')
        
        # --- MODIFIED: Load FinBERT Model ---
        self.finbert_model = None
        self.finbert_tokenizer = None
        
        if AutoTokenizer and AutoModelForSequenceClassification:
            try:
                model_name = "ProsusAI/finbert"
                self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                logger.info("✓ FinBERT model loaded successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to load FinBERT model: {e}")
                logger.warning("FinBERT unavailable. Sentiment analysis will be disabled.")
        else:
            logger.error("❌ Transformers/Torch not installed. Sentiment analysis is disabled.")
        # --- END MODIFIED ---
        
        if self.newsapi_key:
            logger.info("✓ NewsAPI configured")
        else:
            logger.warning("⚠️ NewsAPI key not found. News sentiment will be limited.")
        
        if self.twitter_bearer_token:
            logger.info("✓ Twitter API configured")
        else:
            logger.info("ℹ️ Twitter API not configured (optional)")
        
        logger.info("✓ Sentiment Analyzer initialized")
    
    
    def analyze_sentiment(self, symbol: str, days_back: int = 7, max_articles: int = 20) -> Dict:
        company_name = self._get_company_name(symbol)
        articles = self._fetch_news(company_name, days_back, max_articles)
        
        if not articles or not self.finbert_model:
            logger.debug(f"No articles found or FinBERT model not loaded for {symbol}")
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'classification': 'NEUTRAL',
                'confidence': 0.0,
                'article_count': 0,
                'sources': []
            }
        
        sentiments = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            score = self._analyze_text(text) # Calls the new FinBERT function
            sentiments.append(score)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        confidence = min(1.0, len(sentiments) / 10) # Simple confidence score
        
        # Classification based on the -1 to +1 score
        if avg_sentiment > 0.1:
            classification = 'BULLISH'
        elif avg_sentiment < -0.1:
            classification = 'BEARISH'
        else:
            classification = 'NEUTRAL'
        
        return {
            'symbol': symbol,
            'sentiment_score': float(avg_sentiment),
            'classification': classification,
            'confidence': float(confidence),
            'article_count': len(articles),
            'sources': [a.get('source', {}).get('name', 'Unknown') for a in articles[:5]]
        }

    # --- NEW: FinBERT Analysis Function ---
    def _analyze_text(self, text: str) -> float:
        """
        Analyze text sentiment using FinBERT.
        Returns a single score between -1 (bearish) and +1 (bullish).
        """
        if not self.finbert_model or not text:
            return 0.0
        
        try:
            # 1. Tokenize the text
            inputs = self.finbert_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding=True
            )
            
            # 2. Get model prediction (logits)
            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = self.finbert_model(**inputs)
            
            # 3. Convert logits to probabilities (softmax)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT output labels are: [positive, negative, neutral]
            positive_prob = probabilities[0][0].item()
            negative_prob = probabilities[0][1].item()
            # neutral_prob = probabilities[0][2].item()
            
            # 4. Calculate a single composite score
            # This score ranges from -1 (max negative) to +1 (max positive)
            sentiment_score = positive_prob - negative_prob
            
            return float(sentiment_score)
            
        except Exception as e:
            logger.debug(f"FinBERT analysis failed: {e}")
            return 0.0
    # --- END NEW ---

    def _fetch_news(self, query: str, days_back: int, max_articles: int) -> List[Dict]:
        """Fetch news from available sources"""
        articles = []
        
        # Try NewsAPI
        if self.newsapi_key:
            articles.extend(self._fetch_newsapi(query, days_back, max_articles))
        
        # Try Yahoo Finance RSS (free, no API key)
        if len(articles) < max_articles:
            articles.extend(self._fetch_yahoo_finance_news(query, max_articles - len(articles)))
        
        return articles[:max_articles]
    
    def _fetch_newsapi(self, query: str, days_back: int, max_articles: int) -> List[Dict]:
        """Fetch from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'pageSize': max_articles,
                'apiKey': self.newsapi_key,
                'language': 'en'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                logger.debug(f"NewsAPI returned {response.status_code}")
                return []
        
        except Exception as e:
            logger.debug(f"NewsAPI fetch failed: {e}")
            return []
    
    def _fetch_yahoo_finance_news(self, query: str, max_articles: int) -> List[Dict]:
        """Fetch from Yahoo Finance RSS (free, no API key needed)"""
        try:
            # Yahoo Finance RSS feed
            encoded_query = quote_plus(query)
            url = f"https://finance.yahoo.com/rss/headline?s={encoded_query}"
            
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                articles = []
                
                for item in root.findall('.//item')[:max_articles]:
                    title = item.find('title')
                    description = item.find('description')
                    link = item.find('link')
                    
                    articles.append({
                        'title': title.text if title is not None else '',
                        'description': description.text if description is not None else '',
                        'url': link.text if link is not None else '',
                        'source': {'name': 'Yahoo Finance'}
                    })
                
                return articles
            else:
                return []
        
        except Exception as e:
            logger.debug(f"Yahoo Finance RSS fetch failed: {e}")
            return []
    
    def _get_company_name(self, symbol: str) -> str:
        """Convert symbol to company name for better search results"""
        # Remove .NS suffix for Indian stocks
        clean_symbol = symbol.replace('.NS', '')
        
        # Basic mapping (expand as needed)
        name_map = {
            'RELIANCE': 'Reliance Industries',
            'TCS': 'Tata Consultancy Services',
            'HDFCBANK': 'HDFC Bank',
            'ICICIBANK': 'ICICI Bank',
            'INFY': 'Infosys',
            'ITC': 'ITC Limited',
            'SBIN': 'State Bank of India',
            'BHARTIARTL': 'Bharti Airtel',
            'WIPRO': 'Wipro',
            'TATAMOTORS': 'Tata Motors',
        }
        
        return name_map.get(clean_symbol, clean_symbol)


logger.info("✓ SentimentAnalyzer module loaded successfully")