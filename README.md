# 🚀 Institutional-Grade Algorithmic Trading System (v3)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![Scikit-Learn](https://img.shields.io/badge/sklearn-1.2-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview
A sophisticated, automated trading architecture designed for the Indian Equity Markets (NSE). This system bridges the gap between traditional technical analysis and modern AI by orchestrating a **Hybrid LSTM-Transformer** neural network alongside a **13-model Machine Learning Ensemble**.

Engineered for strict risk management, the system operates on a ₹10k capital constraint, utilizing **Kelly Criterion** sizing and **FinBERT** sentiment analysis to generate high-probability signals.

## 🏗️ System Architecture
The project is structured into four distinct modules:

* **`src/` (Core Logic)**: Contains the pattern detectors, ML ensembles, and risk management engines.
* **`scripts/` (Execution)**: Entry points for training models, running backtests, and daily signal generation.
* **`models/` (Deep Learning)**: Stores the trained Hybrid LSTM-Transformer and CNN weights.
* **`config/` (Settings)**: Centralized configuration for hyperparameters and trading constants.

## ⚡ Key Capabilities

### 1. Hybrid AI Prediction
Combines temporal sequence modeling with attention mechanisms:
* **CNN-LSTM**: Extracts local trend features.
* **Transformer**: Captures long-range market dependencies.
* **Ensemble**: A dynamic voting stack of XGBoost, CatBoost, and Random Forest.

### 2. Advanced Feature Engineering
* **Zero-Leakage Pipeline**: Custom feature store builder ensures no look-ahead bias during training.
* **Pattern Detection**: Algorithmic recognition of Head & Shoulders, Wedges, and Support/Resistance clusters.

### 3. Risk Management Engine
* **Capital Tracking**: Prevents negative cash balances by tracking "Remaining Capital" in real-time.
* **Dynamic Sizing**: Adjusts position sizes based on volatility (EGARCH) and model confidence.

## 🛠️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yashwanth2172/Trading-system.git](https://github.com/yashwanth2172/Trading-system.git)