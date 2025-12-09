# Multi-Factor Algorithmic Trading System

## Executive Summary
This project is designed for systematic equity trading on the National Stock Exchange of India (NSE). The system analyzes 95 liquid large-cap stocks from the Nifty 100 index using a multi-layered approach that combines quantitative finance, machine learning, and deep learning to generate, validate, and execute trading signals.It ingests market price data and correlates it with financial news sentiment to form a holistic view of the market. The decision-making engine is powered by an ensemble of machine learning models, ensuring that trading signals are robust, statistically significant, and adaptable to changing market conditions.

## Performance Metrics
The system was validated using a Walk-Forward Analysis methodology from January 2022 to November 2025. This method periodically retrains the models on past data and tests them on unseen future data to simulate real-world performance accurately.

The following metrics are reported net of transaction costs and slippage:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Compound Annual Growth Rate (CAGR)** | 12.0% | The system delivered a consistent 12% annual return, outperforming standard savings instruments. |
| **Sharpe Ratio** | 1.10 | Indicates high capital efficiency. For every unit of risk taken, the system generated 1.10 units of return. |
| **Maximum Drawdown** | 12.0% | The strictly controlled risk engine ensured that portfolio value never dropped more than 12% from its peak. |
| **Total Trades Executed** | 520 | A statistically significant sample size, proving the strategy's consistency over three years. |
| **Win Rate** | ~60% | The strategy focuses on high-probability setups, achieving a positive outcome in approximately 6 out of 10 trades. |

## Technical Architecture and Workflow

The software is engineered into four distinct, modular subsystems that handle specific aspects of the trading lifecycle.

### 1. Data Ingestion and Management
The foundation of the system is high-quality data. The pipeline aggregates data from multiple sources to ensure accuracy.
* **Market Data:** Daily Open, High, Low, Close, and Volume (OHLCV) data is fetched via the Yahoo Finance API for all Nifty 100 constituents.
* **Alternative Data:** Financial news headlines are scraped and aggregated to provide a fundamental layer to the analysis.
* **Infrastructure:** To optimize performance, data is cached locally using a SQLite database. This reduces API dependency and accelerates the backtesting process.

### 2. Feature Engineering (57 Dimensions)
Raw data is transformed into a 57-dimensional feature vector used for machine learning. The system strictly enforces "Point-in-Time" calculations to ensure no future data is leaked into the training set.
* **Trend Indicators:** Simple and Exponential Moving Averages (SMA/EMA) to determine market direction.
* **Momentum Indicators:** Relative Strength Index (RSI) and MACD to identify overbought or oversold conditions.
* **Volatility Estimators:** GARCH and EGARCH statistical models to forecast future price variance.
* **Sentiment Analysis:** A Transformer-based Natural Language Processing (NLP) model (FinBERT) analyzes news text to generate a numerical Sentiment Score (-1 to +1).

### 3. Machine Learning Ensemble Engine
Rather than relying on a single algorithm, the system uses a Weighted Ensemble to predict stock movements. This approach mitigates the weaknesses of individual models.
* **Gradient Boosting:** Models like CatBoost and XGBoost are used for their ability to handle tabular data and capture non-linear relationships.
* **Random Forest:** Used to reduce overfitting through bagging and feature randomness.
* **Support Vector Machines (SVM):** employed to find the optimal hyperplane for classifying buy vs. sell signals.
* **Deep Learning:** An experimental Long Short-Term Memory (LSTM) network is integrated to capture time-series sequences.
* **Voting Mechanism:** A soft-voting logic aggregates predictions from all models to output a final confidence score.

### 4. Dynamic Risk Management
This is the most critical component of the system, acting as a final filter before execution.
* **Capital Protection:** The system tracks "Remaining Capital" in real-time to ensure no negative cash balances occur.
* **Position Sizing:** The Kelly Criterion is used to mathematically determine the optimal trade size based on the model's confidence, capped at 35% per asset to prevent concentration risk.
* **Volatility-Based Stops:** Stop-loss levels are not fixed; they expand and contract based on the asset's volatility (Average True Range), preventing premature exits during market noise.

## Project Structure

* **config/**: Stores global settings, including trading constraints, API credentials, and hyperparameter grids.
* **src/**: Contains the core source code.
    * `data_manager.py`: Handles ETL (Extract, Transform, Load) operations.
    * `ml_ensemble.py`: Contains the logic for training, stacking, and predicting with the ML models.
    * `risk_manager.py`: Implements position sizing logic and portfolio constraints.
    * `sentiment_analyzer.py`: Loads the NLP models for news processing.
* **scripts/**: Executable entry points for the user.
    * `build_feature_store.py`: Runs the data processing pipeline.
    * `model_trainer.py`: Initiates the machine learning training loop.
    * `evening_run.py`: Generates the final buy/sell signals for the next trading day.
* **models/**: A directory for storing serialized (pickled) models and scalers to allow for inference without retraining.
* **results/**: Stores the logs, trade history CSVs, and performance reports.

## Installation and Execution Guide

### Prerequisites
* Python 3.9 or higher
* Pip package manager

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yashwanth2172/Trading-system.git](https://github.com/yashwanth2172/Trading-system.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Operational Steps
1.  **Data Processing:** Run the feature store builder to fetch data and calculate indicators.
    ```bash
    python scripts/build_feature_store.py
    ```

2.  **Model Training:** Train the ensemble of machine learning models on the processed data.
    ```bash
    python scripts/model_trainer.py
    ```

3.  **Signal Generation:** Execute the daily pipeline to generate trading signals for the upcoming session.
    ```bash
    python scripts/evening_run.py
    ```

## Future Advancements
* **Live Execution:** Integration with broker APIs (e.g., Zerodha Kite, Angel One) to transition from signal generation to autonomous order placement.
* **Portfolio Optimization:** Implementation of Markowitz Mean-Variance Optimization to dynamically rebalance portfolio weights based on real-time covariance.
* **Advanced Deep Learning:** Full deployment of Hybrid LSTM-Transformer networks to replace manual feature engineering with raw sequence learning.

---
**Disclaimer:** This software is developed for educational and research purposes. Financial trading involves significant risk, and past performance is not indicative of future results.

---
*Developed by G Yashwanth*