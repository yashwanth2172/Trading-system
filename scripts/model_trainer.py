
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import joblib 
from datetime import datetime

# --- Add project root to path ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# ---------------------------------

from config.settings import (
    TRADE_UNIVERSE, ML_CONFIG, ML_FEATURES, PROJECT_DIRECTORIES
)
from src.ml_ensemble import Enhanced13ModelEnsemble

# --- Setup logging ---
LOGS_DIR = project_root / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'model_training_fast.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FastModelTrainer:
    def __init__(self):
        """Initialize trainer components"""
        logger.info("="*80)
        logger.info("13-MODEL ENSEMBLE TRAINER (FAST - From Feature Store)")
        logger.info("="*80)
        
        self.feature_store_dir = project_root / 'data_cache' / 'feature_store'
        self.ml_ensemble = Enhanced13ModelEnsemble()
        
        if not self.feature_store_dir.exists():
            logger.error(f"Feature store not found at: {self.feature_store_dir}")
            logger.error("Please run scripts/build_feature_store.py first!")
            raise FileNotFoundError("Feature store not found")

    def run(self):
        logger.info(f"\n🚀 Starting model training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load all data from feature store
        logger.info("\n" + "="*80)
        logger.info(f"STEP 1: LOADING FEATURES FROM STORE ({len(TRADE_UNIVERSE)} stocks)")
        logger.info("="*80)
        
        all_features_df = []
        for symbol in tqdm(TRADE_UNIVERSE, desc="Loading features"):
            feature_file = self.feature_store_dir / f"{symbol}.pkl"
            if feature_file.exists():
                try:
                    df = joblib.load(feature_file) 
                    all_features_df.append(df)
                except Exception as e:
                    logger.error(f"Could not load feature file for {symbol}: {e}")
            else:
                logger.warning(f"No feature file found for {symbol}. Skipping.")
        
        if not all_features_df:
            logger.error("Feature Store is empty! Run scripts/build_feature_store.py first.")
            return

        full_dataset = pd.concat(all_features_df)
        
        # Drop rows with NaNs (e.g., from label generation)
        full_dataset = full_dataset.dropna(subset=ML_FEATURES + ['label'])
        
        logger.info(f"✓ Total training samples: {len(full_dataset)}")
        
        # Step 2: Prepare features and labels
        logger.info("\n" + "="*80)
        logger.info("STEP 2: PREPARING FEATURES & LABELS")
        logger.info("="*80)
        
        X = full_dataset[ML_FEATURES]
        y = full_dataset['label']
        
        logger.info(f"✓ Features shape: {X.shape}")
        logger.info(f"✓ Labels shape: {y.shape}")
        logger.info(f"✓ Class distribution: {dict(y.value_counts(normalize=True))}")
        
        # Step 3: Train 13-model ensemble
        logger.info("\n" + "="*80)
        logger.info("STEP 3: TRAINING 13-MODEL ENSEMBLE")
        logger.info("="*80)
        
        training_scores = self.ml_ensemble.train(X, y, symbol="ENSEMBLE")
        
        # Step 4: Display results
        logger.info("\n" + "="*80)
        logger.info("STEP 4: TRAINING RESULTS")
        logger.info("="*80)
        
        self.display_training_results(training_scores)
        
        # Step 5: Save models
        logger.info("\n" + "="*80)
        logger.info("STEP 5: SAVING ENSEMBLE MODEL")
        logger.info("="*80)
        
        success = self.ml_ensemble.save_models(symbol="ENSEMBLE")
        
        if success:
            logger.info("✅ 13-Model Ensemble 'General Manager' saved successfully")
        else:
            logger.error("⚠️  Error saving ensemble model")
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE (FAST)")
        logger.info("="*80)

    def display_training_results(self, scores: dict):
        """Display training results"""
        if not scores or 'error' in scores:
            logger.error("Training failed")
            return
        
        logger.info("\n📊 MODEL PERFORMANCE:")
        logger.info("-" * 80)
        
        categories = {
            'Tree Models': ['rf', 'et'],
            'Gradient Boosting': ['xgb', 'lgbm', 'catboost', 'adaboost', 'gb'],
            'Linear Models': ['lr', 'ridge'],
            'Advanced Models': ['svm', 'mlp', 'gnb', 'knn'],
            'Meta Ensembles': ['stacking', 'voting']
        }
        
        for category, models in categories.items():
            logger.info(f"\n{category}:")
            for model in models:
                if model in scores:
                    score = scores[model]
                    logger.info(f"  {model.upper():12s} - Acc: {score['accuracy']:.3f}, "
                              f"F1: {score['f1']:.3f}, Prec: {score['precision']:.3f}, "
                              f"Rec: {score['recall']:.3f}")
        
        logger.info("\n" + "="*80)
        accuracies = [s['accuracy'] for s in scores.values() if 'accuracy' in s]
        f1_scores = [s['f1'] for s in scores.values() if 'f1' in s]
        
        if accuracies and f1_scores:
            logger.info(f"Average Accuracy: {np.mean(accuracies):.3f}")
            logger.info(f"Average F1 Score: {np.mean(f1_scores):.3f}") 
            logger.info(f"Best Model (Accuracy): {max(scores.items(), key=lambda x: x[1].get('accuracy', 0))[0].upper()}")
            logger.info(f"Best Model (F1): {max(scores.items(), key=lambda x: x[1].get('f1', 0))[0].upper()}")

def main():
    """Main entry point"""
    try:
        trainer = FastModelTrainer()
        trainer.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()