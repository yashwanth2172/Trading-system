# Multi-Factor Algorithmic Trading System (v3)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-v1.2-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Executive Summary
This project implements a comprehensive algorithmic trading infrastructure designed for the **Nifty 100** equity market. It bridges the gap between traditional quantitative finance and modern AI by orchestrating a multi-stage pipeline: **Data Ingestion**, **Zero-Leakage Feature Engineering**, **Hybrid Prediction**, and **Dynamic Risk Management**.

The system is engineered to operate under strict constraints (₹10k capital), utilizing **Fractional Kelly Criterion** sizing and **Volatility-Adjusted** stop-loss mechanisms. By integrating **NLP Sentiment Analysis** (FinBERT) with **Technical Momentum**, the system achieves a risk-adjusted return (Sharpe Ratio 1.10) significantly superior to buy-and-hold benchmarks.

## ⚙️ System Specifications

| Parameter | Specification |
| :--- | :--- |
| **Market** | National Stock Exchange of India (NSE) |
| **Asset Universe** | Nifty 100 (95 Liquid Large-Cap Stocks) |
| **Initial Capital** | ₹10,000 (INR) |
| **Position Limit** | Maximum 3 Concurrent Positions |
| **Backtesting Period** | January 1, 2022 – November 30, 2025 |
| **Data Frequency** | Daily (End-of-Day) |
| **Risk Management** | Volatility-Adjusted (EGARCH) + Kelly Criterion |

---

## 🏗️ Technical Architecture

The architecture is divided into four autonomous subsystems, ensuring modularity and scalability.

### 1. Data Ingestion & Storage
* **Sources:** YFinance (Primary) for OHLCV data; NSE Official API (Fallback).
* **Infrastructure:** SQLite metadata tracking with persistent caching to minimize API latency.
* **Validation:** Automated outlier detection and data quality scoring (completeness & consistency checks).

### 2. Feature Engineering (57 Dimensions)
The system generates a 57-dimensional feature vector for every trading day, strictly enforcing **point-in-time calculation** to prevent look-ahead bias.
* **Technical:** SMA (Trend), RSI (Momentum), MACD, Bollinger Bands, ATR, ADX.
* **Statistical:** ARIMA price forecasts and GARCH/EGARCH volatility estimates.
* **Pattern Recognition:** Algorithmic detection of Support/Resistance clusters, Fibonacci retracements, and chart patterns (Head & Shoulders, Wedges).
* **Sentiment:** **FinBERT (Transformer-based)** analysis of financial news headlines to quantify market mood (-1 to +1).

### 3. Machine Learning Pipeline
The prediction engine utilizes a sophisticated two-tier approach to maximize accuracy.

* **Champion Model (CatBoost):**
    * Gradient Boosting Decision Tree optimized for categorical features.
    * **Performance:** Achieved **80% Accuracy** and **79% F1-Score** on validation data, serving as the primary decision engine.
* **Deep Learning Support (Experimental Layer):**
    * **Hybrid LSTM-Transformer:** Captures long-range temporal dependencies using Multi-Head Attention.
    * **Ensemble Stack:** A voting mechanism comprising XGBoost, Random Forest, and SVM to validate CatBoost signals.

### 4. Risk Management Engine
* **Capital Tracking:** Real-time tracking of "Remaining Capital" to prevent negative cash balances and ensure trade execution validity.
* **Position Sizing:** Fractional Kelly Criterion capped at 35% of portfolio value per asset.
* **Exit Strategy:** Volatility-based dynamic stops (2x ATR) and profit targets derived from model confidence scores.

---

## 🔄 Operational Workflow

The system executes a fully automated 5-stage daily pipeline:

1.  **Ingestion:** Fetches EOD prices and news for all 100 tickers.
2.  **Processing:** Calculates the 57-feature vector for the new day.
3.  **Prediction:**
    * The **CatBoost Model** ranks all 100 stocks by "Probability of Outperformance."
    * The **Ensemble** validates the top candidates.
4.  **Risk Filtering:** Candidates are rejected if:
    * Forecasted Volatility > Threshold.
    * Conflicting Sentiment (e.g., Technical Buy vs. Negative News).
5.  **Execution:** Generates a Buy/Sell signal with precise Entry, Target, and Stop-loss levels, sent via **Telegram API**.

---

## 📊 Performance Metrics

The following results are derived from a **Walk-Forward Backtest** (Jan 2022 – Nov 2025) across 520 executed trades. Metrics are calculated **net of transaction costs**.

### Financial Performance
| Metric | Result | Industry Context |
| :--- | :--- | :--- |
| **CAGR** | **12.0%** | Consistent annual growth, outperforming inflation and debt instruments. |
| **Sharpe Ratio** | **1.10** | High efficiency; indicates 1.1 units of return for every unit of risk. |
| **Max Drawdown** | **12.0%** | Strict capital preservation; losses never exceeded 12% from peak. |
| **Total Trades** | **520** | Statistically significant sample size ensuring robust results. |

### Model Validation (CatBoost)
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **80.0%** | Correctly predicted direction (Up/Down) in 4 out of 5 cases. |
| **F1-Score** | **0.79** | High reliability in balancing Precision and Recall. |
| **AUC-ROC** | **0.82** | Excellent capability in distinguishing profitable vs. losing setups. |

---

## 🗺️ Future Roadmap

* **Phase 1: Deep Learning Integration (In Progress)**
    * Full deployment of the Hybrid LSTM-Transformer to replace the Feature Engineering layer with raw sequence learning.
* **Phase 2: Live Execution**
    * Integration with **Zerodha Kite Connect API** for fully autonomous order placement.
* **Phase 3: Portfolio Optimization**
    * Implementation of **Mean-Variance Optimization (Markowitz)** to dynamically rebalance portfolio weights based on real-time covariance matrices.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yashwanth2172/Trading-system.git](https://github.com/yashwanth2172/Trading-system.git)
    cd Trading-system
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Pipeline:**
    ```bash
    # Step 1: Build Features
    python scripts/build_feature_store.py

    # Step 2: Generate Signals
    python scripts/evening_run.py
    ```

---
*Developed by G Yashwanth*