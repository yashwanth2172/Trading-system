"""
Enhanced Sequence Model Trainer
Trains: CNN-LSTM, Transformer, AND Hybrid LSTM-Transformer
COMPLETE VERSION
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    TRADE_UNIVERSE, PROJECT_DIRECTORIES, ML_CONFIG,
    DL_SEQUENCE_LENGTH, ML_FEATURES
)
from src.data_manager import DataManager
from src.technical_engine import TechnicalEngine
from src.sequence_models import (
    build_cnn_lstm, build_transformer, build_hybrid_lstm_transformer,
    build_bigru_cnn_hybrid
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SequenceModelTrainer:
    """
    Complete sequence model trainer with all models
    """
    
    def __init__(self):
        self.data_manager = DataManager()
        self.tech_engine = TechnicalEngine()
        self.sequence_length = DL_SEQUENCE_LENGTH
        
        # Directories
        self.model_dir = Path(PROJECT_DIRECTORIES['MODELS'])
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info("✓ Sequence Model Trainer initialized")
    
    def prepare_sequences(self, df: pd.DataFrame, 
                         target_column: str = 'target') -> tuple:
        """
        Prepare sequences for training
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
        
        Returns:
            (X_sequences, y_labels)
        """
        if target_column not in df.columns:
            # Create target (next day price > today)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df = df[:-1]  # Remove last row (no future price)
        
        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['target', 'date', 'symbol', 'timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(self.sequence_length, len(df)):
            # Sequence of features
            seq = df[feature_cols].iloc[i-self.sequence_length:i].values
            sequences.append(seq)
            
            # Label
            labels.append(df[target_column].iloc[i])
        
        X = np.array(sequences)
        y = np.array(labels)
        
        logger.info(f"   Created {len(X)} sequences")
        logger.info(f"   Sequence shape: {X.shape}")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Target distribution: {np.bincount(y)}")
        
        return X, y, feature_cols
    
    def load_and_prepare_data(self, symbols: list, 
                              start_date: str = '2020-01-01',
                              end_date: str = '2024-12-31') -> tuple:
        """
        Load data for all symbols and prepare training set
        
        Returns:
            (X_train, X_val, y_train, y_val, scaler)
        """
        logger.info(f"Loading data for {len(symbols)} symbols...")
        
        all_sequences = []
        all_labels = []
        
        for i, symbol in enumerate(symbols, 1):
            try:
                # Fetch data
                df = self.data_manager.fetch_data_with_retry(
                    symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is None or len(df) < self.sequence_length + 100:
                    logger.warning(f"   Skipping {symbol}: insufficient data")
                    continue
                
                # Calculate technical indicators
                df = self.tech_engine.calculate_all_indicators(df)
                
                # Create sequences
                X, y, feature_cols = self.prepare_sequences(df)
                
                all_sequences.append(X)
                all_labels.append(y)
                
                if i % 10 == 0:
                    logger.info(f"   Processed {i}/{len(symbols)} symbols")
                
            except Exception as e:
                logger.error(f"   Error processing {symbol}: {e}")
                continue
        
        if not all_sequences:
            raise ValueError("No valid data loaded!")
        
        # Combine all sequences
        X_combined = np.vstack(all_sequences)
        y_combined = np.hstack(all_labels)
        
        logger.info(f"\n✓ Combined dataset:")
        logger.info(f"   Total sequences: {len(X_combined)}")
        logger.info(f"   Shape: {X_combined.shape}")
        logger.info(f"   Class 0: {np.sum(y_combined == 0)}, Class 1: {np.sum(y_combined == 1)}")
        
        # Normalize features
        logger.info("\nNormalizing features...")
        n_samples, n_timesteps, n_features = X_combined.shape
        
        # Reshape to 2D for scaling
        X_reshaped = X_combined.reshape(-1, n_features)
        
        # Fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        
        # Reshape back to 3D
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_combined,
            test_size=0.2,
            random_state=42,
            stratify=y_combined
        )
        
        logger.info(f"\n✓ Split complete:")
        logger.info(f"   Training: {len(X_train)} samples")
        logger.info(f"   Validation: {len(X_val)} samples")
        
        return X_train, X_val, y_train, y_val, scaler, feature_cols
    
    def train_model(self, model, model_name: str,
                   X_train, y_train, X_val, y_val,
                   epochs: int = 100, batch_size: int = 32) -> dict:
        """
        Train a sequence model
        
        Returns:
            Training history dict
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TRAINING {model_name.upper()}")
        logger.info(f"{'=' * 80}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / f'{model_name}_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        logger.info(f"\n✓ Training complete for {model_name}")
        
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        logger.info(f"   Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        logger.info(f"   Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Save final model
        model_path = self.model_dir / f'{model_name}_model.h5'
        model.save(str(model_path))
        logger.info(f"   Model saved: {model_path}")
        
        return {
            'history': history.history,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
    
    def train_all_models(self, symbols: list = None, epochs: int = 100):
        """
        Train all sequence models:
        1. CNN-LSTM (original)
        2. Transformer (original)
        3. Hybrid LSTM-Transformer (NEW)
        4. BiGRU-CNN (NEW - optional)
        """
        if symbols is None:
            symbols = TRADE_UNIVERSE[:95]
        
        logger.info("=" * 80)
        logger.info("SEQUENCE MODEL TRAINING - ALL MODELS")
        logger.info("=" * 80)
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Epochs: {epochs}")
        
        # Load and prepare data
        X_train, X_val, y_train, y_val, scaler, feature_cols = self.load_and_prepare_data(symbols)
        
        n_features = X_train.shape[2]
        
        # Save scaler
        scaler_path = self.model_dir / 'sequence_scaler.pkl'
        joblib.dump(scaler, str(scaler_path))
        logger.info(f"\n✓ Scaler saved: {scaler_path}")
        
        # Save feature columns
        feature_path = self.model_dir / 'sequence_features.pkl'
        joblib.dump(feature_cols, str(feature_path))
        
        results = {}
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL 1: CNN-LSTM (Original)
        # ═══════════════════════════════════════════════════════════════
        
        try:
            logger.info("\n" + "=" * 80)
            logger.info("MODEL 1/4: CNN-LSTM")
            logger.info("=" * 80)
            
            cnn_lstm = build_cnn_lstm(self.sequence_length, n_features)
            results['cnn_lstm'] = self.train_model(
                cnn_lstm, 'cnn_lstm',
                X_train, y_train, X_val, y_val,
                epochs=epochs
            )
        except Exception as e:
            logger.error(f"CNN-LSTM training failed: {e}")
            results['cnn_lstm'] = {'error': str(e)}
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL 2: Transformer (Original)
        # ═══════════════════════════════════════════════════════════════
        
        try:
            logger.info("\n" + "=" * 80)
            logger.info("MODEL 2/4: Transformer")
            logger.info("=" * 80)
            
            transformer = build_transformer(self.sequence_length, n_features)
            results['transformer'] = self.train_model(
                transformer, 'transformer',
                X_train, y_train, X_val, y_val,
                epochs=epochs
            )
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            results['transformer'] = {'error': str(e)}
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL 3: Hybrid LSTM-Transformer (NEW - BEST)
        # ═══════════════════════════════════════════════════════════════
        
        try:
            logger.info("\n" + "=" * 80)
            logger.info("MODEL 3/4: Hybrid LSTM-Transformer (ENHANCED)")
            logger.info("=" * 80)
            
            hybrid = build_hybrid_lstm_transformer(self.sequence_length, n_features)
            results['hybrid_lstm_transformer'] = self.train_model(
                hybrid, 'hybrid_lstm_transformer',
                X_train, y_train, X_val, y_val,
                epochs=epochs
            )
        except Exception as e:
            logger.error(f"Hybrid LSTM-Transformer training failed: {e}")
            results['hybrid_lstm_transformer'] = {'error': str(e)}
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL 4: BiGRU-CNN (NEW - Alternative)
        # ═══════════════════════════════════════════════════════════════
        
        try:
            logger.info("\n" + "=" * 80)
            logger.info("MODEL 4/4: BiGRU-CNN Hybrid (ALTERNATIVE)")
            logger.info("=" * 80)
            
            bigru_cnn = build_bigru_cnn_hybrid(self.sequence_length, n_features)
            results['bigru_cnn'] = self.train_model(
                bigru_cnn, 'bigru_cnn',
                X_train, y_train, X_val, y_val,
                epochs=epochs
            )
        except Exception as e:
            logger.error(f"BiGRU-CNN training failed: {e}")
            results['bigru_cnn'] = {'error': str(e)}
        
        # ═══════════════════════════════════════════════════════════════
        # FINAL SUMMARY
        # ═══════════════════════════════════════════════════════════════
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - ALL MODELS")
        logger.info("=" * 80)
        
        logger.info("\n📊 FINAL RESULTS:")
        logger.info("-" * 80)
        
        for model_name, result in results.items():
            if 'error' in result:
                logger.info(f"{model_name:25} - FAILED: {result['error']}")
            else:
                logger.info(
                    f"{model_name:25} - Val Acc: {result['val_acc']:.4f}, "
                    f"Val Loss: {result['val_loss']:.4f}"
                )
        
        logger.info("-" * 80)
        
        # Save results summary
        results_path = self.model_dir / 'training_results.pkl'
        joblib.dump(results, str(results_path))
        logger.info(f"\n✓ Results saved: {results_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL SEQUENCE MODELS TRAINED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return results


def main():
    """Main training function"""
    trainer = SequenceModelTrainer()
    
    # Get symbols
    symbols = TRADE_UNIVERSE[:95]
    
    # Train all models
    results = trainer.train_all_models(
        symbols=symbols,
        epochs=100  # Adjust as needed
    )
    
    return results


if __name__ == '__main__':
    main()
