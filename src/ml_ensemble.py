"""
Enhanced 13-Model ML Ensemble with Dynamic Weighted Ensemble
COMPLETE VERSION with all original functionality + new features
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                             GradientBoostingClassifier, AdaBoostClassifier,
                             VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available - hyperparameter optimization disabled")

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DYNAMIC WEIGHTED ENSEMBLE MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DynamicWeightedEnsemble:
    """
    Adaptive ensemble that adjusts weights based on recent performance
    
    Features:
    1. Performance-based weighting
    2. Adaptive to market conditions
    3. Handles varying model accuracies
    4. Automatic underperformer filtering
    5. Exponential weighting (favors recent performance)
    """
    
    def __init__(self, models: Dict[str, any], lookback_window: int = 20, 
                 min_accuracy: float = 0.55, weighting_method: str = 'exponential'):
        """
        Args:
            models: Dict of {model_name: model_object}
            lookback_window: Days to consider for performance
            min_accuracy: Minimum accuracy to keep model active
            weighting_method: 'equal', 'accuracy', 'exponential', 'softmax'
        """
        self.models = models
        self.lookback_window = lookback_window
        self.min_accuracy = min_accuracy
        self.weighting_method = weighting_method
        
        # Initialize equal weights
        n_models = len(models)
        self.weights = {name: 1.0/n_models for name in models.keys()}
        
        # Performance tracking
        self.performance_history = {name: [] for name in models.keys()}
        self.prediction_history = []
        
        # Model status
        self.active_models = set(models.keys())
        
        # Statistics
        self.total_predictions = 0
        self.correct_predictions = 0
        
        logger.info(f"✓ Dynamic Ensemble initialized with {n_models} models")
        logger.info(f"   Lookback: {lookback_window} days, Min accuracy: {min_accuracy:.1%}")
        logger.info(f"   Weighting method: {weighting_method}")
    
    def predict(self, X) -> Dict:
        """
        Generate weighted ensemble prediction
        
        Returns:
            Dict with ensemble_prediction, individual_predictions, 
            weights_used, confidence, agreement, uncertainty
        """
        predictions = {}
        prediction_probabilities = {}
        
        # Get predictions from active models only
        for name in self.active_models:
            try:
                model = self.models[name]
                
                # Handle different model types
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    pred_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                    predictions[name] = float(pred_prob[0])
                    prediction_probabilities[name] = proba[0] if len(proba.shape) > 1 else proba
                elif hasattr(model, 'decision_function'):
                    # For SVM without probability
                    decision = model.decision_function(X)
                    # Convert to probability-like score [0, 1]
                    pred_prob = 1 / (1 + np.exp(-decision))
                    predictions[name] = float(pred_prob[0])
                else:
                    pred = model.predict(X)
                    predictions[name] = float(pred[0])
                
            except Exception as e:
                logger.warning(f"   Model {name} prediction failed: {e}")
                predictions[name] = 0.5  # Neutral
        
        if not predictions:
            return {
                'ensemble_prediction': 0.5,
                'individual_predictions': {},
                'weights_used': {},
                'confidence': 0.3,
                'agreement': 0.0,
                'uncertainty': 1.0,
                'active_models': 0
            }
        
        # Calculate weighted prediction
        weighted_pred = sum(
            predictions[name] * self.weights.get(name, 0)
            for name in predictions.keys()
        )
        
        # Calculate agreement metrics
        pred_values = list(predictions.values())
        variance = np.var(pred_values) if len(pred_values) > 1 else 0.0
        agreement = max(0.0, 1.0 - (variance * 2))  # Scale variance to 0-1
        
        # Calculate uncertainty (entropy-based)
        uncertainty = self._calculate_uncertainty(pred_values)
        
        # Confidence based on agreement and average weight of agreeing models
        avg_weight = np.mean(list(self.weights.values()))
        confidence = (agreement * 0.7) + (avg_weight * 0.3)
        confidence = np.clip(confidence, 0.3, 0.95)
        
        # Get top contributing models
        top_contributors = sorted(
            [(name, predictions[name] * self.weights[name]) for name in predictions.keys()],
            key=lambda x: abs(x[1] - 0.5),
            reverse=True
        )[:3]
        
        return {
            'ensemble_prediction': float(weighted_pred),
            'individual_predictions': predictions,
            'weights_used': {k: self.weights[k] for k in predictions.keys()},
            'confidence': float(confidence),
            'agreement': float(agreement),
            'uncertainty': float(uncertainty),
            'prediction_variance': float(variance),
            'active_models': len(self.active_models),
            'top_contributors': top_contributors
        }
    
    def _calculate_uncertainty(self, predictions: List[float]) -> float:
        """Calculate prediction uncertainty using entropy"""
        if not predictions:
            return 1.0
        
        # Treat as probability distribution
        predictions = np.array(predictions)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)  # Avoid log(0)
        
        # Binary entropy
        entropy = -(predictions * np.log2(predictions) + 
                   (1 - predictions) * np.log2(1 - predictions))
        
        return float(np.mean(entropy))
    
    def update_weights(self, predictions: Dict[str, float], actual: float):
        """
        Update model weights based on recent accuracy
        
        Args:
            predictions: {model_name: prediction_probability}
            actual: Actual outcome (0 or 1)
        """
        # Record performance
        for name, pred in predictions.items():
            # Binary accuracy
            pred_class = 1 if pred > 0.5 else 0
            correct = 1.0 if pred_class == actual else 0.0
            
            self.performance_history[name].append(correct)
            
            # Keep only recent history
            if len(self.performance_history[name]) > self.lookback_window:
                self.performance_history[name].pop(0)
        
        # Update global statistics
        self.total_predictions += 1
        ensemble_pred = sum(predictions[name] * self.weights[name] for name in predictions.keys())
        ensemble_correct = 1 if (ensemble_pred > 0.5 and actual == 1) or (ensemble_pred <= 0.5 and actual == 0) else 0
        self.correct_predictions += ensemble_correct
        
        # Calculate recent accuracies
        accuracies = {}
        for name in self.models.keys():
            if len(self.performance_history[name]) >= 5:  # Minimum history
                accuracies[name] = np.mean(self.performance_history[name])
            else:
                accuracies[name] = 0.5  # Neutral for new models
        
        # Filter out underperformers
        self.active_models = {
            name for name, acc in accuracies.items()
            if acc >= self.min_accuracy
        }
        
        if not self.active_models:
            # Fallback: reactivate all models if all filtered out
            self.active_models = set(self.models.keys())
            logger.warning("⚠️ All models filtered out - reactivating all")
        
        # Recalculate weights based on method
        if self.weighting_method == 'equal':
            n_active = len(self.active_models)
            self.weights = {name: 1.0/n_active if name in self.active_models else 0.0 
                          for name in self.models.keys()}
        
        elif self.weighting_method == 'accuracy':
            # Direct accuracy weighting
            total_acc = sum(accuracies[name] for name in self.active_models)
            if total_acc > 0:
                self.weights = {
                    name: accuracies[name] / total_acc if name in self.active_models else 0.0
                    for name in self.models.keys()
                }
        
        elif self.weighting_method == 'exponential':
            # Exponential weighting (favors better models more)
            exp_accuracies = {
                name: np.exp(accuracies[name]) 
                for name in self.active_models
            }
            total_exp = sum(exp_accuracies.values())
            if total_exp > 0:
                self.weights = {
                    name: exp_accuracies.get(name, 0) / total_exp if name in self.active_models else 0.0
                    for name in self.models.keys()
                }
        
        elif self.weighting_method == 'softmax':
            # Softmax weighting (smooth exponential)
            acc_array = np.array([accuracies[name] for name in self.active_models])
            softmax_weights = np.exp(acc_array * 5) / np.sum(np.exp(acc_array * 5))  # Temperature=5
            self.weights = {
                name: float(softmax_weights[i]) if name in self.active_models else 0.0
                for i, name in enumerate(self.models.keys())
            }
        
        logger.debug(f"   Weights updated | Active: {len(self.active_models)}/{len(self.models)}")
        logger.debug(f"   Top 3: {self.get_top_models(3)}")
    
    def get_top_models(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top K performing models with their accuracies"""
        accuracies = {
            name: np.mean(hist) if hist else 0.0
            for name, hist in self.performance_history.items()
        }
        
        sorted_models = sorted(
            accuracies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_models[:top_k]
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        accuracies = {
            name: np.mean(hist) if hist else 0.0
            for name, hist in self.performance_history.items()
        }
        
        ensemble_accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0
        
        return {
            'model_accuracies': accuracies,
            'weights': self.weights,
            'active_models': list(self.active_models),
            'inactive_models': list(set(self.models.keys()) - self.active_models),
            'top_3_models': self.get_top_models(3),
            'average_accuracy': np.mean(list(accuracies.values())),
            'ensemble_accuracy': ensemble_accuracy,
            'total_predictions': self.total_predictions
        }
    
    def reset_performance(self):
        """Reset all performance tracking"""
        self.performance_history = {name: [] for name in self.models.keys()}
        self.total_predictions = 0
        self.correct_predictions = 0
        self.active_models = set(self.models.keys())
        n_models = len(self.models)
        self.weights = {name: 1.0/n_models for name in self.models.keys()}
        logger.info("✓ Performance history reset")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENHANCED 13-MODEL ENSEMBLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Enhanced13ModelEnsemble:
    """
    Enhanced 13-model ensemble with:
    - Dynamic weighting
    - Cross-validation
    - Feature importance
    - Hyperparameter optimization (optional)
    - Multiple ensemble methods (voting, stacking)
    - Performance tracking
    
    Models:
    1. Random Forest
    2. Extra Trees
    3. XGBoost
    4. LightGBM
    5. CatBoost
    6. Gradient Boosting
    7. AdaBoost
    8. Logistic Regression
    9. SVM
    10. KNN
    11. Naive Bayes
    12. Decision Tree
    13. Neural Network
    """
    
    def __init__(self, ensemble_method: str = 'dynamic', use_optimization: bool = False):
        """
        Args:
            ensemble_method: 'dynamic', 'voting', 'stacking'
            use_optimization: Enable hyperparameter optimization with Optuna
        """
        self.ensemble_method = ensemble_method
        self.use_optimization = use_optimization and OPTUNA_AVAILABLE
        
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importances_ = None
        self.training_date = None
        self.training_metrics = {}
        
        # Ensemble components
        self.dynamic_ensemble = None
        self.voting_ensemble = None
        self.stacking_ensemble = None
        
        # Cross-validation
        self.cv_scores = {}
        
        logger.info("✓ Enhanced 13-Model Ensemble initialized")
        logger.info(f"   Ensemble method: {ensemble_method}")
        logger.info(f"   Optimization: {'Enabled' if self.use_optimization else 'Disabled'}")
    
    def _build_models(self):
        """Build all 13 models with optimized parameters"""
        
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            
            'et': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                min_child_weight=1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            ),
            
            'lgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                class_weight='balanced'
            ),
            
            'catboost': cb.CatBoostClassifier(
                iterations=200,
                depth=7,
                learning_rate=0.1,
                l2_leaf_reg=3.0,
                subsample=0.8,
                colsample_bylevel=0.8,
                random_seed=42,
                verbose=False,
                auto_class_weights='Balanced'
            ),
            
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            ),
            
            'adaboost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.5,
                random_state=42
            ),
            
            'lr': LogisticRegression(
                max_iter=1000,
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            
            'knn': KNeighborsClassifier(
                n_neighbors=11,
                weights='distance',
                metric='minkowski',
                p=2,
                n_jobs=-1
            ),
            
            'nb': GaussianNB(),
            
            'dt': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'
            ),
            
            'nn': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        logger.info(f"✓ Built {len(self.models)} models")
    
    def _optimize_hyperparameters(self, X, y, model_name: str):
        """Optimize hyperparameters using Optuna"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available - skipping optimization")
            return
        
        logger.info(f"   Optimizing {model_name} with Optuna...")
        
        def objective(trial):
            if model_name == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
            
            elif model_name == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            
            else:
                return 0.5
            
            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        logger.info(f"      Best score: {study.best_value:.4f}")
        
        # Update model with best parameters
        if model_name == 'xgb':
            self.models[model_name] = xgb.XGBClassifier(
                **study.best_params,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
        elif model_name == 'rf':
            self.models[model_name] = RandomForestClassifier(
                **study.best_params,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
    
    def fit(self, X, y, optimize_models: List[str] = None):
        """
        Train all models
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            optimize_models: List of model names to optimize ['xgb', 'rf', 'lgbm']
        """
        logger.info("=" * 80)
        logger.info(f"Training 13-model ensemble on {len(X)} samples, {X.shape[1]} features")
        logger.info("=" * 80)
        
        # Build models
        self._build_models()
        
        # Optimize if requested
        if self.use_optimization and optimize_models:
            for model_name in optimize_models:
                if model_name in self.models:
                    self._optimize_hyperparameters(X, y, model_name)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model with cross-validation
        for name, model in self.models.items():
            logger.info(f"\nTraining {name}...")
            
            try:
                # Cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                self.cv_scores[name] = cv_scores
                
                logger.info(f"   CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Train on full data
                model.fit(X_scaled, y)
                
                # Calculate training metrics
                y_pred = model.predict(X_scaled)
                y_pred_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, zero_division=0)
                recall = recall_score(y, y_pred, zero_division=0)
                f1 = f1_score(y, y_pred, zero_division=0)
                
                self.training_metrics[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std()
                }
                
                if y_pred_proba is not None:
                    auc = roc_auc_score(y, y_pred_proba)
                    self.training_metrics[name]['auc'] = auc
                    logger.info(
                        f"   {name}: Acc={accuracy:.3f}, F1={f1:.3f}, "
                        f"Prec={precision:.3f}, Rec={recall:.3f}, AUC={auc:.3f}"
                    )
                else:
                    logger.info(
                        f"   {name}: Acc={accuracy:.3f}, F1={f1:.3f}, "
                        f"Prec={precision:.3f}, Rec={recall:.3f}"
                    )
                
            except Exception as e:
                logger.error(f"   {name} training failed: {e}")
                self.training_metrics[name] = {'error': str(e)}
        
        # Build ensemble based on method
        if self.ensemble_method == 'dynamic':
            self.dynamic_ensemble = DynamicWeightedEnsemble(
                self.models,
                lookback_window=20,
                min_accuracy=0.55,
                weighting_method='exponential'
            )
            logger.info("\n✓ Dynamic weighted ensemble initialized")
        
        elif self.ensemble_method == 'voting':
            estimators = [(name, model) for name, model in self.models.items()]
            self.voting_ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
            self.voting_ensemble.fit(X_scaled, y)
            logger.info("\n✓ Voting ensemble trained")
        
        elif self.ensemble_method == 'stacking':
            estimators = [(name, model) for name, model in self.models.items() 
                         if name not in ['lr']]  # Exclude LR as it'll be meta-learner
            self.stacking_ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5,
                n_jobs=-1
            )
            self.stacking_ensemble.fit(X_scaled, y)
            logger.info("\n✓ Stacking ensemble trained")
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        self.is_fitted = True
        self.training_date = datetime.now().isoformat()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Ensemble training complete")
        logger.info("=" * 80)
        self._print_training_summary()
    
    def predict(self, X) -> Dict:
        """
        Generate ensemble prediction
        
        Returns:
            Dict with prediction, probability, confidence, etc.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction based on ensemble method
        if self.ensemble_method == 'dynamic' and self.dynamic_ensemble:
            result = self.dynamic_ensemble.predict(X_scaled)
            result['prediction'] = 1 if result['ensemble_prediction'] > 0.5 else 0
            result['probability'] = result['ensemble_prediction']
        
        elif self.ensemble_method == 'voting' and self.voting_ensemble:
            pred = self.voting_ensemble.predict(X_scaled)[0]
            proba = self.voting_ensemble.predict_proba(X_scaled)[0]
            result = {
                'prediction': int(pred),
                'probability': float(proba[1]),
                'ensemble_prediction': float(proba[1]),
                'confidence': abs(proba[1] - 0.5) * 2  # Scale 0.5-1.0 to 0-1
            }
        
        elif self.ensemble_method == 'stacking' and self.stacking_ensemble:
            pred = self.stacking_ensemble.predict(X_scaled)[0]
            proba = self.stacking_ensemble.predict_proba(X_scaled)[0]
            result = {
                'prediction': int(pred),
                'probability': float(proba[1]),
                'ensemble_prediction': float(proba[1]),
                'confidence': abs(proba[1] - 0.5) * 2
            }
        
        else:
            # Fallback: simple averaging
            predictions = []
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_scaled)[0, 1]
                    else:
                        pred = float(model.predict(X_scaled)[0])
                    predictions.append(pred)
                except:
                    pass
            
            avg_pred = np.mean(predictions) if predictions else 0.5
            result = {
                'prediction': 1 if avg_pred > 0.5 else 0,
                'probability': float(avg_pred),
                'ensemble_prediction': float(avg_pred),
                'confidence': abs(avg_pred - 0.5) * 2
            }
        
        return result
    
    def update_performance(self, X, y_true):
        """
        Update model weights based on actual outcomes
        (Only for dynamic ensemble)
        """
        if not self.is_fitted or self.ensemble_method != 'dynamic' or not self.dynamic_ensemble:
            return
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name in self.dynamic_ensemble.active_models:
            try:
                model = self.models[name]
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_scaled)[:, 1]
                else:
                    pred = model.predict(X_scaled)
                predictions[name] = float(pred[0])
            except:
                predictions[name] = 0.5
        
        self.dynamic_ensemble.update_weights(predictions, float(y_true))
    
    def _calculate_feature_importances(self):
        """Calculate average feature importances from tree-based models"""
        tree_models = ['rf', 'et', 'xgb', 'lgbm', 'catboost', 'gb', 'dt']
        importances = []
        
        for name in tree_models:
            if name in self.models and hasattr(self.models[name], 'feature_importances_'):
                importances.append(self.models[name].feature_importances_)
        
        if importances:
            self.feature_importances_ = np.mean(importances, axis=0)
    
    def get_feature_importances(self, feature_names: List[str] = None, top_k: int = 20) -> pd.DataFrame:
        """Get top feature importance ranking"""
        if self.feature_importances_ is None:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False).head(top_k)
        
        return importance_df
    
    def _print_training_summary(self):
        """Print comprehensive training summary"""
        logger.info("\n📊 TRAINING SUMMARY:")
        logger.info("-" * 80)
        
        for name, metrics in self.training_metrics.items():
            if 'error' in metrics:
                logger.info(f"{name:15} - FAILED: {metrics['error']}")
            else:
                logger.info(
                    f"{name:15} - Acc: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}, "
                    f"CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}"
                )
        
        logger.info("-" * 80)
        
        # Average metrics
        valid_metrics = {k: v for k, v in self.training_metrics.items() if 'error' not in v}
        if valid_metrics:
            avg_acc = np.mean([m['accuracy'] for m in valid_metrics.values()])
            avg_f1 = np.mean([m['f1'] for m in valid_metrics.values()])
            avg_auc = np.mean([m['cv_auc_mean'] for m in valid_metrics.values()])
            
            logger.info(f"Average Accuracy: {avg_acc:.3f}")
            logger.info(f"Average F1:       {avg_f1:.3f}")
            logger.info(f"Average CV AUC:   {avg_auc:.3f}")
    
    def save(self, filepath: str):
        """Save ensemble to disk"""
        ensemble_data = {
            'models': self.models,
            'scaler': self.scaler,
            'dynamic_ensemble': self.dynamic_ensemble,
            'voting_ensemble': self.voting_ensemble,
            'stacking_ensemble': self.stacking_ensemble,
            'feature_importances': self.feature_importances_,
            'training_date': self.training_date,
            'training_metrics': self.training_metrics,
            'cv_scores': self.cv_scores,
            'is_fitted': self.is_fitted,
            'ensemble_method': self.ensemble_method
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"✓ Ensemble saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load ensemble from disk"""
        ensemble_data = joblib.load(filepath)
        
        instance = cls(ensemble_method=ensemble_data.get('ensemble_method', 'dynamic'))
        instance.models = ensemble_data['models']
        instance.scaler = ensemble_data['scaler']
        instance.dynamic_ensemble = ensemble_data.get('dynamic_ensemble')
        instance.voting_ensemble = ensemble_data.get('voting_ensemble')
        instance.stacking_ensemble = ensemble_data.get('stacking_ensemble')
        instance.feature_importances_ = ensemble_data['feature_importances']
        instance.training_date = ensemble_data['training_date']
        instance.training_metrics = ensemble_data.get('training_metrics', {})
        instance.cv_scores = ensemble_data.get('cv_scores', {})
        instance.is_fitted = ensemble_data['is_fitted']
        
        logger.info(f"✓ Ensemble loaded from {filepath}")
        logger.info(f"   Trained: {instance.training_date}")
        logger.info(f"   Method: {instance.ensemble_method}")
        
        return instance
    
    def get_model_summary(self) -> Dict:
        """Get summary of all models and their performance"""
        summary = {
            'ensemble_method': self.ensemble_method,
            'training_date': self.training_date,
            'is_fitted': self.is_fitted,
            'num_models': len(self.models),
            'training_metrics': self.training_metrics,
            'cv_scores': {name: scores.tolist() for name, scores in self.cv_scores.items()}
        }
        
        if self.dynamic_ensemble:
            summary['dynamic_performance'] = self.dynamic_ensemble.get_performance_summary()
        
        return summary
    
    def evaluate(self, X_test, y_test) -> Dict:
        """Evaluate ensemble on test data"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        X_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        result = self.predict(X_test)
        y_pred = result['prediction']
        y_proba = result['probability']
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info("\n📊 TEST SET EVALUATION:")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"AUC:       {metrics['auc']:.4f}")
        
        return metrics


logger.info("✓ Enhanced ML Ensemble module loaded (COMPLETE VERSION)")
