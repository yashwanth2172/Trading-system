
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class TimeSeriesEngine:
    def __init__(self):
        self.arima_order = (5, 1, 2)
        self.garch_p = 1
        self.garch_q = 1
        self.lstm_units = 64
        self.sequence_length = 60
        
        # Models
        self.arima_model = None
        self.garch_model = None
        self.egarch_model = None
        self.lstm_model = None
        
        # Scalers
        self.price_scaler = MinMaxScaler()
        
        logger.info("✓ TimeSeriesEngine initialized (ARIMA, GARCH & EGARCH)")
    

    # ARIMA FORECASTING
    
    def forecast_arima(self, prices: pd.Series, steps: int = 1) -> Tuple[float, float]:
        try:
            if len(prices) < 50:
                return float(prices.iloc[-1]), 0.3
        
            model = ARIMA(prices, order=self.arima_order)
            self.arima_model = model.fit()
            
            forecast = self.arima_model.forecast(steps=steps)
            forecast_price = float(forecast.iloc[0]) if steps == 1 else float(forecast.mean())
            fitted = self.arima_model.fittedvalues
            if len(fitted) > 0:
                errors = np.abs((prices[-len(fitted):].values - fitted.values) / prices[-len(fitted):].values)
                mape = np.mean(errors) * 100
                confidence = max(0.1, 1.0 - (mape / 100))
            else:
                confidence = 0.5
            
            return forecast_price, confidence
            
        except Exception as e:
            logger.warning(f"ARIMA forecast failed: {e}")
            return float(prices.iloc[-1]), 0.3
    

    # GARCH VOLATILITY FORECASTING

    
    def forecast_garch(self, prices: pd.Series) -> Dict:
        try:
            returns = prices.pct_change().dropna() * 100
            
            if len(returns) < 30:
                return {'volatility': returns.std(), 'confidence': 0.3}
            
            model = arch_model(returns, vol='Garch', p=self.garch_p, q=self.garch_q)
            self.garch_model = model.fit(disp='off', show_warning=False)
            
            forecast = self.garch_model.forecast(horizon=1)
            volatility = np.sqrt(forecast.variance.values[-1, 0])
            
            confidence = 0.7
            
            return {
                'volatility': float(volatility),
                'confidence': confidence,
                'mean_return': self.garch_model.params.get('mu', 0.0)
            }
            
        except Exception as e:
            logger.warning(f"GARCH forecast failed: {e}")
            returns = prices.pct_change().dropna() * 100
            return {'volatility': returns.std(), 'confidence': 0.3, 'mean_return': 0.0}
    
    # EGARCH VOLATILITY FORECASTING 
    
    def forecast_egarch(self, prices: pd.Series) -> Dict:
        try:
            returns = prices.pct_change().dropna() * 100
            
            if len(returns) < 30:
                return {
                    'volatility': returns.std(),
                    'confidence': 0.3,
                    'leverage_effect': 0.0
                }
            
            model = arch_model(returns, vol='EGARCH', p=self.garch_p, q=self.garch_q)
            self.egarch_model = model.fit(disp='off', show_warning=False)
            
            forecast = self.egarch_model.forecast(horizon=1)
            volatility = np.sqrt(forecast.variance.values[-1, 0])
            
            params = self.egarch_model.params
            gamma = params.get('gamma[1]', 0.0) if 'gamma[1]' in params else 0.0
            
            aic = self.egarch_model.aic
            confidence = max(0.1, 1.0 / (1.0 + abs(aic) / 1000))
            
            return {
                'volatility': float(volatility),
                'confidence': confidence,
                'leverage_effect': float(gamma),
                'aic': aic
            }
            
        except Exception as e:
            logger.warning(f"EGARCH forecast failed: {e}")
            returns = prices.pct_change().dropna() * 100
            return {
                'volatility': returns.std(),
                'confidence': 0.3,
                'leverage_effect': 0.0
            }
    
    # LSTM PRICE FORECASTING
    
    def _build_lstm(self):
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def forecast_lstm(self, prices: pd.Series) -> Tuple[float, float]:
        try:
            if len(prices) < self.sequence_length + 20:
                return float(prices.iloc[-1]), 0.3
            
            scaled = self.price_scaler.fit_transform(prices.values.reshape(-1, 1))
            
            X, y = [], []
            for i in range(self.sequence_length, len(scaled)):
                X.append(scaled[i-self.sequence_length:i, 0])
                y.append(scaled[i, 0])
            
            X = np.array(X).reshape((len(X), self.sequence_length, 1))
            y = np.array(y)
            
            self.lstm_model = self._build_lstm()
            self.lstm_model.fit(X, y, epochs=30, batch_size=16, verbose=0, validation_split=0.2)
            
            last_seq = X[-1].reshape(1, self.sequence_length, 1)
            pred_scaled = self.lstm_model.predict(last_seq, verbose=0)[0, 0]
            forecast_price = self.price_scaler.inverse_transform([[pred_scaled]])[0, 0]
            
            val_loss = self.lstm_model.history.history.get('val_loss', [1.0])[-1]
            confidence = max(0.1, 1.0 - min(val_loss, 1.0))
            
            return float(forecast_price), confidence
            
        except Exception as e:
            logger.warning(f"LSTM forecast failed: {e}")
            return float(prices.iloc[-1]), 0.3
    
    # HYBRID ARIMA-LSTM-EGARCH ENSEMBLE
    
    def forecast_hybrid(self, prices: pd.Series) -> Dict:
        logger.debug(f"Running hybrid forecast on {len(prices)} data points")
        arima_price, arima_conf = self.forecast_arima(prices)
        lstm_price, lstm_conf = self.forecast_lstm(prices)
        egarch_result = self.forecast_egarch(prices)
        total_conf = arima_conf + lstm_conf + egarch_result['confidence']
        
        if total_conf > 0:
            w_arima = arima_conf / total_conf
            w_lstm = lstm_conf / total_conf
            w_egarch = egarch_result['confidence'] / total_conf
        else:
            w_arima = w_lstm = w_egarch = 0.33
        
        price_forecast = (w_arima * arima_price + w_lstm * lstm_price) / (w_arima + w_lstm)
        current_price = float(prices.iloc[-1])
        expected_return = (price_forecast - current_price) / current_price
        volatility_adj = max(0.5, 1.0 - (egarch_result['volatility'] / 10))
        direction_prob = 0.5 + (expected_return * volatility_adj)
        direction_prob = np.clip(direction_prob, 0.3, 0.85)
        overall_confidence = (arima_conf + lstm_conf + egarch_result['confidence']) / 3
        
        return {
            'price_forecast': float(price_forecast),
            'volatility_forecast': egarch_result['volatility'],
            'confidence': float(overall_confidence),
            'direction_probability': float(direction_prob),
            'expected_return_pct': float(expected_return * 100),
            'leverage_effect': egarch_result['leverage_effect'],
            'components': {
                'arima': {
                    'prediction': float(arima_price),
                    'confidence': float(arima_conf),
                    'weight': float(w_arima)
                },
                'lstm': {
                    'prediction': float(lstm_price),
                    'confidence': float(lstm_conf),
                    'weight': float(w_lstm)
                },
                'egarch': {
                    'volatility': egarch_result['volatility'],
                    'confidence': egarch_result['confidence'],
                    'weight': float(w_egarch)
                }
            }
        }


logger.info("✓ TimeSeriesEngine module loaded")
