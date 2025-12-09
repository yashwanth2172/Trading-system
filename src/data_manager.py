
import logging
import os
import pickle
import sqlite3
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


class DataManager:
   
    
    def __init__(self, 
                 cache_dir: str = 'data_cache',
                 max_workers: int = 1,
                 sleep_between_calls: float = 5.0,
                 max_retries: int = 5,
                 alpha_vantage_key: Optional[str] = None,
                 twelve_data_key: Optional[str] = None):
       
        # Determine project root
        project_root = Path(__file__).parent.parent
        self.cache_dir = project_root / cache_dir
        self.pickle_cache_dir = self.cache_dir / 'pickle_data'
        self.metadata_db_path = self.cache_dir / 'metadata.db'
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pickle_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.max_workers = max_workers
        self.sleep_between_calls = sleep_between_calls
        self.max_retries = max_retries
        self.semaphore = threading.Semaphore(max_workers)
        
        # API keys for fallback sources
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_KEY', '')
        self.twelve_data_key = twelve_data_key or os.getenv('TWELVE_DATA_KEY', '')
        
        # Initialize metadata database
        self._init_metadata_db()
        
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'yfinance_success': 0,
            'yfinance_failures': 0,
            'nse_api_success': 0,  
            'nse_api_failures': 0,  
            'alpha_vantage_success': 0,
            'alpha_vantage_failures': 0,
            'twelve_data_success': 0,
            'twelve_data_failures': 0,
            'total_requests': 0,
            'total_api_calls': 0
        }

        
        logger.info(
            f"✓ DataManager initialized. Workers: {max_workers}, Semaphore: {max_workers}, "
            f"Sleep: {sleep_between_calls}s, Max Retries: {max_retries}. "
            f"Cache: {self.pickle_cache_dir}, MetaDB: {self.metadata_db_path}"
        )
    
    def _init_metadata_db(self):
        """Initialize SQLite metadata database for cache tracking"""
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    symbol TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    last_updated TEXT,
                    row_count INTEGER,
                    file_path TEXT,
                    source TEXT,
                    data_quality_score REAL,
                    PRIMARY KEY (symbol, start_date, end_date)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fetch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    source TEXT,
                    timestamp TEXT,
                    success INTEGER,
                    error_message TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"✓ Metadata database initialized/connected at {self.metadata_db_path}")
        
        except Exception as e:
            logger.error(f"Failed to initialize metadata database: {e}")
            raise
    
    # ========== PUBLIC API ==========
    
    def fetch_data(self, 
                   symbols: List[str],
                   start_date: datetime,
                   end_date: datetime,
                   use_cache: bool = True,
                   interval: str = '1d',
                   force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
      
        logger.info(f"📊 Fetching data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        logger.info(f"   Interval: {interval}, Cache: {'Enabled' if use_cache else 'Disabled'}, "
                   f"Force Refresh: {force_refresh}")
        
        results = {}
        
        if self.max_workers == 1:
            # Sequential processing (safer, recommended for stability)
            for i, symbol in enumerate(symbols, 1):
                if i % 10 == 0 or i == len(symbols):
                    logger.info(f"   Progress: {i}/{len(symbols)} symbols processed...")
                
                df = self._fetch_single_symbol(
                    symbol, start_date, end_date, use_cache, interval, force_refresh
                )
                
                if df is not None and not df.empty:
                    results[symbol] = df
        else:
            # Parallel processing (faster but requires careful rate limiting)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(
                        self._fetch_single_symbol, 
                        symbol, start_date, end_date, use_cache, interval, force_refresh
                    ): symbol
                    for symbol in symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            results[symbol] = df
                    except Exception as e:
                        logger.error(f"Exception processing {symbol}: {e}")
        
        success_rate = (len(results) / len(symbols) * 100) if symbols else 0
        logger.info(f"✓ Successfully fetched data for {len(results)}/{len(symbols)} symbols ({success_rate:.1f}%)")
        
        return results
    def fetch_multiple(self, 
                       symbols: List[str], 
                       period: str = '1y',
                       interval: str = '1d', 
                       use_cache: bool = True,
                       force_refresh: bool = False) -> Dict[str, pd.DataFrame]:

        end_date = datetime.now()
        
        # Parse period string
        if period.endswith('y'):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=365 * years)
        elif period.endswith('mo'):
            months = int(period[:-2])
            start_date = end_date - timedelta(days=30 * months)
        elif period.endswith('d'):
            days = int(period[:-1])
            start_date = end_date - timedelta(days=days)
        else:
            # Default to 1 year
            logger.warning(f"Unknown period format '{period}', defaulting to 1 year")
            start_date = end_date - timedelta(days=365)
        
        logger.info(f"fetch_multiple: period='{period}' -> {start_date.date()} to {end_date.date()}")
        
        return self.fetch_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            interval=interval,
            force_refresh=force_refresh
        )
    
    def get_historical_data(self,
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime,
                           interval: str = '1d') -> Optional[pd.DataFrame]:
        
        logger.info(f"get_historical_data: {symbol} from {start_date.date()} to {end_date.date()}")
        
        result = self.fetch_data(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
            interval=interval,
            force_refresh=False
        )
        
        return result.get(symbol)
    
    def prepare_ml_features(self,
                           df: pd.DataFrame,
                           label_horizon: int = 5,
                           dropna: bool = True,
                           min_return_pct: float = 0.005) -> Tuple[pd.DataFrame, pd.Series]:
     
        if df.empty or len(df) < 50:
            logger.debug("Input dataframe too short for feature preparation (need at least 50 rows)")
            return pd.DataFrame(), pd.Series(dtype=int)
        
        df = df.copy()
        
        # Calculate future returns for labels
        df['future_return'] = df['close'].pct_change(label_horizon).shift(-label_horizon)
        df['label'] = (df['future_return'] > min_return_pct).astype(int)
        
        # Select feature columns (exclude label and non-feature columns)
        exclude_cols = ['label', 'future_return', 'dividends', 'stock splits', 'capital gains']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['label'].copy()
        
        # Drop NaN/Inf values if requested
        if dropna:
            # Check for NaN in X
            nan_mask_x = X.isna().any(axis=1)
            # Check for Inf in X
            inf_mask_x = np.isinf(X.select_dtypes(include=[np.number])).any(axis=1)
            # Check for NaN in y
            nan_mask_y = y.isna()
            
            # Combine masks
            valid_mask = ~(nan_mask_x | inf_mask_x | nan_mask_y)
            
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:
                logger.warning(f"After dropping NaN/Inf, only {len(X)} samples remain (< 50). Feature set may be insufficient.")
        
        return X, y
    
    def get_stats(self) -> Dict:
            
            stats = self.stats.copy()
            
            # Calculate derived metrics
            if stats['total_requests'] > 0:
                stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_requests']
                stats['overall_success_rate'] = (
                    (stats['yfinance_success'] + stats['nse_api_success'] +
                    stats['alpha_vantage_success'] + stats['twelve_data_success']) / 
                    stats['total_requests']
                )
            else:
                stats['cache_hit_rate'] = 0.0
                stats['overall_success_rate'] = 0.0
            
            return stats
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            if older_than_days:
                cutoff_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                cursor.execute('''
                    SELECT file_path FROM cache_metadata 
                    WHERE last_updated < ?
                ''', (cutoff_date,))
                
                files_to_delete = cursor.fetchall()
                
                for (file_path,) in files_to_delete:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not delete cache file {file_path}: {e}")
                
                cursor.execute('DELETE FROM cache_metadata WHERE last_updated < ?', (cutoff_date,))
                logger.info(f"✓ Cleared cache older than {older_than_days} days ({len(files_to_delete)} files)")
            else:
                cursor.execute('SELECT file_path FROM cache_metadata')
                all_files = cursor.fetchall()
                
                for (file_path,) in all_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not delete cache file {file_path}: {e}")
                
                cursor.execute('DELETE FROM cache_metadata')
                cursor.execute('DELETE FROM fetch_history')
                logger.info(f"✓ Cleared all cache ({len(all_files)} files)")
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    # ========== PRIVATE METHODS - DATA FETCHING ==========
    
    def _fetch_single_symbol(self, symbol: str, start_date: datetime, end_date: datetime,
                            use_cache: bool, interval: str, force_refresh: bool) -> Optional[pd.DataFrame]:
        """
        Fetch data for single symbol with cache, fallback logic, and quality checks
        """
        self.stats['total_requests'] += 1
        
        # Try cache first (unless force_refresh is True)
        if use_cache and not force_refresh:
            cached_df = self._load_from_cache(symbol, start_date, end_date)
            if cached_df is not None:
                self.stats['cache_hits'] += 1
                logger.debug(f"✓ Cache hit: {symbol}")
                return cached_df
        
        self.stats['cache_misses'] += 1
        
        # Acquire semaphore for rate limiting
        with self.semaphore:
            # Try data sources in order of preference
            sources = [
                ('yfinance', self._fetch_with_yfinance),
                ('nse_api', self._fetch_with_nse_api),
            ]
            
            # Add optional sources if API keys are available
            if self.alpha_vantage_key:
                sources.append(('alpha_vantage', self._fetch_with_alpha_vantage))
            
            if self.twelve_data_key:
                sources.append(('twelve_data', self._fetch_with_twelve_data))
            
            for source_name, fetch_function in sources:
                try:
                    if source_name == 'yfinance':
                        df = fetch_function(symbol, start_date, end_date, interval, self.max_retries)
                    elif source_name == 'nse_api':
                        df = fetch_function(symbol, start_date, end_date, self.max_retries)
                    else:
                        df = fetch_function(symbol, start_date, end_date)
                    
                    if df is not None and not df.empty:
                        quality_score = self._calculate_data_quality(df)
                        
                        if quality_score >= 0.7:  
                            self.stats['nse_api_success'] += 1
                            self.stats[f'{source_name}_success'] += 1
                            self.stats['total_api_calls'] += 1
                            

                            df = self._calculate_technical_indicators(df)
                            
                            # Save to cache
                            if use_cache:
                                self._save_to_cache(symbol, df, start_date, end_date, source_name, quality_score)
                            
                            # Log success
                            self._log_fetch_history(symbol, source_name, True, None)
                            
                            return df
                        else:
                            logger.warning(f"Data quality too low for {symbol} from {source_name}: {quality_score:.2f}")
                    
                except Exception as e:
                    logger.debug(f"Error fetching {symbol} from {source_name}: {e}")
                    self.stats[f'{source_name}_failures'] += 1
                    self._log_fetch_history(symbol, source_name, False, str(e))
                
                # Fallback message
                if source_name != sources[-1][0]:  # Not the last source
                    logger.info(f"🔁 {source_name} failed for {symbol}. Trying next source...")
            
            # All sources failed
            logger.error(f"❌ All sources failed for {symbol}")
            return None
    
    def _fetch_with_yfinance(self, symbol: str, start_date: datetime, end_date: datetime,
                            interval: str = '1d', max_retries: int = 5) -> Optional[pd.DataFrame]:
        for attempt in range(1, max_retries + 1):
            try:
                # ✅ FIXED: Use yf.download() with minimal, universally-supported parameters
                df = yf.download(
                    tickers=symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=True  # Disable progress bar
                )
                
                if df is not None and not df.empty:
                    # Clean and standardize
                    df = self._clean_dataframe(df)
                    
                    if not df.empty:
                        logger.debug(f"✓ yfinance: {symbol} -> {len(df)} rows")
                        return df
                
                # Empty result, retry with delay
                if attempt < max_retries:
                    time.sleep(self.sleep_between_calls * attempt)
            
            except Exception as e:
                logger.warning(
                    f"yfinance error for {symbol} (Attempt {attempt}/{max_retries}): {e}"
                )
                
                if attempt < max_retries:
                    time.sleep(self.sleep_between_calls * attempt)
        
        logger.error(f"❌ yfinance failed for {symbol} after {max_retries} attempts")
        return None
    
    def _fetch_with_nse_api(self, symbol: str, start_date: datetime, end_date: datetime,
                           max_retries: int = 5) -> Optional[pd.DataFrame]:
       
        # Remove .NS suffix for NSE API
        nse_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        for attempt in range(1, max_retries + 1):
            try:
                # NSE API endpoint (historical data)
                url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={nse_symbol}"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.nseindia.com/'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        logger.warning(f"NSE API returned invalid JSON for {symbol}")
                        if attempt < max_retries:
                            time.sleep(self.sleep_between_calls * attempt * 2)
                        continue
                    
                    if 'data' in data and data['data']:
                        df = pd.DataFrame(data['data'])
                        
                        # Convert date column
                        if 'CH_TIMESTAMP' in df.columns:
                            df['Date'] = pd.to_datetime(df['CH_TIMESTAMP'])
                            df = df.set_index('Date')
                            
                            # Rename columns to standard OHLCV
                            column_mapping = {
                                'CH_OPENING_PRICE': 'open',
                                'CH_TRADE_HIGH_PRICE': 'high',
                                'CH_TRADE_LOW_PRICE': 'low',
                                'CH_CLOSING_PRICE': 'close',
                                'CH_TOT_TRADED_QTY': 'volume'
                            }
                            
                            df = df.rename(columns=column_mapping)
                            
                            # Keep only OHLCV columns
                            required_cols = ['open', 'high', 'low', 'close', 'volume']
                            if all(col in df.columns for col in required_cols):
                                df = df[required_cols]
                                
                                # Filter date range
                                df = df[(df.index >= start_date) & (df.index <= end_date)]
                                
                                # Clean data
                                df = self._clean_dataframe(df)
                                
                                if not df.empty:
                                    logger.debug(f"✓ NSE API: {symbol} -> {len(df)} rows")
                                    return df
                
                # Rate limit handling
                if attempt < max_retries:
                    time.sleep(self.sleep_between_calls * attempt)
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"NSE API request error for {symbol} (Attempt {attempt}): {e}")
                if attempt < max_retries:
                    time.sleep(self.sleep_between_calls * attempt * 2)
            
            except Exception as e:
                logger.warning(f"NSE API error for {symbol} (Attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(self.sleep_between_calls * attempt)
        
        logger.error(f"❌ NSE API failed for {symbol} after {max_retries} attempts")
        return None

# FILE: src/data_manager.py - PART 2 (Continuation)

    def _fetch_with_alpha_vantage(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage API (requires API key)
        """
        if not self.alpha_vantage_key:
            return None
        
        # Remove exchange suffix for Alpha Vantage
        av_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': av_symbol,
                'outputsize': 'full',
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    ts_data = data['Time Series (Daily)']
                    
                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(ts_data, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    
                    # Rename columns
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    
                    # Convert to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Filter date range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    # Clean
                    df = self._clean_dataframe(df)
                    
                    if not df.empty:
                        logger.debug(f"✓ Alpha Vantage: {symbol} -> {len(df)} rows")
                        return df
        
        except Exception as e:
            logger.debug(f"Alpha Vantage error for {symbol}: {e}")
        
        return None
    
    def _fetch_with_twelve_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch data from Twelve Data API (requires API key)
        """
        if not self.twelve_data_key:
            return None
        
        # Remove exchange suffix
        td_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        try:
            url = 'https://api.twelvedata.com/time_series'
            params = {
                'symbol': td_symbol,
                'interval': '1day',
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'apikey': self.twelve_data_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' in data:
                    df = pd.DataFrame(data['values'])
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.set_index('datetime')
                    df = df.sort_index()
                    
                    # Convert to numeric
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Clean
                    df = self._clean_dataframe(df)
                    
                    if not df.empty:
                        logger.debug(f"✓ Twelve Data: {symbol} -> {len(df)} rows")
                        return df
        
        except Exception as e:
            logger.debug(f"Twelve Data error for {symbol}: {e}")
        
        return None
    
    # ========== PRIVATE METHODS - DATA PROCESSING ==========
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize dataframe
        """
        if df.empty:
            return df
        
        # Handle multi-level columns (from yf.download)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.debug(f"Missing required columns. Have: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Keep only OHLCV columns
        df = df[required_cols].copy()
        
        # Convert to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Remove rows where all values are NaN
        df.dropna(how='all', inplace=True)
        
        # Remove rows where critical columns (open, close) are NaN
        df.dropna(subset=['open', 'close'], inplace=True)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Remove duplicate dates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score (0-1)
        
        Checks:
        - Completeness (no missing values)
        - Consistency (OHLC relationships)
        - Reasonableness (no extreme outliers)
        """
        if df.empty:
            return 0.0
        
        score = 1.0
        
        # Check 1: Completeness (penalize missing values)
        missing_ratio = df.isna().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 0.3
        
        # Check 2: OHLC consistency
        invalid_ohlc = 0
        if 'high' in df.columns and 'low' in df.columns:
            invalid_ohlc += (df['high'] < df['low']).sum()
        if 'open' in df.columns and 'high' in df.columns:
            invalid_ohlc += (df['open'] > df['high']).sum()
        if 'close' in df.columns and 'low' in df.columns:
            invalid_ohlc += (df['close'] < df['low']).sum()
        
        ohlc_error_ratio = invalid_ohlc / len(df) if len(df) > 0 else 0
        score -= ohlc_error_ratio * 0.3
        
        # Check 3: Volume reasonableness
        if 'volume' in df.columns:
            zero_volume_ratio = (df['volume'] == 0).sum() / len(df)
            score -= zero_volume_ratio * 0.2
        
        # Check 4: Price continuity (no huge gaps)
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            extreme_moves = (returns.abs() > 0.5).sum()  # >50% daily move
            extreme_ratio = extreme_moves / len(df) if len(df) > 0 else 0
            score -= extreme_ratio * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        
        Includes:
        - Moving averages (SMA, EMA)
        - Momentum indicators (RSI, MACD, ROC)
        - Volatility indicators (ATR, Bollinger Bands)
        - Volume indicators
        - Trend indicators
        """
        if df.empty or len(df) < 20:
            return df
        
        df = df.copy()
        
        try:
            # ========== BASIC RETURNS ==========
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # ========== MOVING AVERAGES ==========
            for period in [5, 10, 20, 50, 200]:
                df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # ========== RSI (Relative Strength Index) ==========
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # ========== MACD ==========
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # ========== BOLLINGER BANDS ==========
            sma_20 = df['close'].rolling(20, min_periods=1).mean()
            std_20 = df['close'].rolling(20, min_periods=1).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20.replace(0, 1)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1)
            
            # ========== ATR (Average True Range) ==========
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14, min_periods=1).mean()
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            
            # ========== VOLUME INDICATORS ==========
            df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
            df['volume_sma_50'] = df['volume'].rolling(50, min_periods=1).mean()
            
            # OBV (On-Balance Volume)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # ========== VOLATILITY ==========
            df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
            df['volatility_annual'] = df['volatility'] * np.sqrt(252)
            
            # ========== MOMENTUM INDICATORS ==========
            df['momentum'] = df['close'] - df['close'].shift(10)
            df['roc'] = (df['close'] / df['close'].shift(10).replace(0, 1) - 1) * 100  # Rate of Change
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(14, min_periods=1).min()
            high_14 = df['high'].rolling(14, min_periods=1).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14).replace(0, 1)
            df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()
            
            # Williams %R
            df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14).replace(0, 1)
            
            # ========== TREND INDICATORS ==========
            # ADX (Average Directional Index) - simplified
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs()
            plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
            minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
            
            atr_14 = df['atr']
            plus_di = 100 * (plus_dm.rolling(14, min_periods=1).mean() / atr_14.replace(0, 1))
            minus_di = 100 * (minus_dm.rolling(14, min_periods=1).mean() / atr_14.replace(0, 1))
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
            df['adx'] = dx.rolling(14, min_periods=1).mean()
            
            # ========== PRICE PATTERNS ==========
            # Price position in range
            rolling_min_14 = df['low'].rolling(14, min_periods=1).min()
            rolling_max_14 = df['high'].rolling(14, min_periods=1).max()
            df['price_position'] = (df['close'] - rolling_min_14) / (rolling_max_14 - rolling_min_14).replace(0, 1)
            
            # Distance from moving averages
            df['dist_from_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20'].replace(0, 1)
            df['dist_from_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50'].replace(0, 1)
            
            # ========== CANDLESTICK PATTERNS (SIMPLIFIED) ==========
            # Body size
            df['candle_body'] = (df['close'] - df['open']).abs()
            df['candle_range'] = df['high'] - df['low']
            df['body_to_range'] = df['candle_body'] / df['candle_range'].replace(0, 1)
            
            # Upper/Lower shadows
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # ========== CUSTOM FEATURES ==========
            # Trend strength
            df['trend_strength'] = (df['ema_10'] - df['ema_50']).abs() / df['ema_50'].replace(0, 1)
            
            # Price momentum
            df['price_momentum_5'] = df['close'] / df['close'].shift(5).replace(0, 1) - 1
            df['price_momentum_20'] = df['close'] / df['close'].shift(20).replace(0, 1) - 1
            
            # Volatility ratio
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50, min_periods=1).mean().replace(0, 1)
            
            # Volume trend
            df['volume_trend'] = df['volume'] / df['volume'].rolling(50, min_periods=1).mean().replace(0, 1)
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
        
        return df
    
    # ========== PRIVATE METHODS - CACHING ==========
    
    def _generate_cache_key(self, symbol: str, start_date: datetime, end_date: datetime) -> str:
        """Generate unique cache key using hash"""
        key_str = f"{symbol}_{start_date.date()}_{end_date.date()}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and fresh
        """
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_path, last_updated, row_count, data_quality_score 
                FROM cache_metadata 
                WHERE symbol = ? AND start_date = ? AND end_date = ?
            ''', (symbol, str(start_date.date()), str(end_date.date())))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                file_path, last_updated, row_count, quality_score = result
                
                # Check cache freshness (24 hours for daily data)
                cache_time = datetime.fromisoformat(last_updated)
                cache_age = datetime.now() - cache_time
                
                # Use cache if less than 24 hours old and quality is good
                if cache_age.days < 1 and quality_score >= 0.7:
                    # Load pickle
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            df = pickle.load(f)
                        
                        # Validate loaded data
                        if len(df) == row_count and not df.empty:
                            return df
        
        except Exception as e:
            logger.debug(f"Cache load error for {symbol}: {e}")
        
        return None
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame, start_date: datetime, 
                      end_date: datetime, source: str, quality_score: float):
        """
        Save data to cache with metadata
        """
        try:
            cache_key = self._generate_cache_key(symbol, start_date, end_date)
            file_path = self.pickle_cache_dir / f"{cache_key}.pkl"
            
            # Save pickle
            with open(file_path, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache_metadata 
                (symbol, start_date, end_date, last_updated, row_count, file_path, source, data_quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                str(start_date.date()),
                str(end_date.date()),
                datetime.now().isoformat(),
                len(df),
                str(file_path),
                source,
                quality_score
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"✓ Cached {symbol}: {len(df)} rows from {source} (quality: {quality_score:.2f})")
        
        except Exception as e:
            logger.debug(f"Cache save error for {symbol}: {e}")
    
    def _log_fetch_history(self, symbol: str, source: str, success: bool, error_message: Optional[str]):
        """
        Log fetch attempt to history table
        """
        try:
            conn = sqlite3.connect(str(self.metadata_db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fetch_history 
                (symbol, source, timestamp, success, error_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol,
                source,
                datetime.now().isoformat(),
                1 if success else 0,
                error_message
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.debug(f"Failed to log fetch history: {e}")


logger.info("✓ DataManager module loaded successfully")
