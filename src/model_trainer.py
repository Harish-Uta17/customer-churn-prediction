"""
Model training pipeline for Customer Churn Prediction
Fixed: Encodes Target ('Yes'/'No') to (1/0) to prevent XGBoost and Predictor errors.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os

# Ensure these are imported from your config
from src.config import MODEL_DIR, RANDOM_STATE, TEST_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChurnModelTrainer:

    def __init__(self, df: pd.DataFrame, target_col: str = 'Churn'):
        self.df = df
        self.target_col = target_col
        
        self.X = None
        self.y = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.models = {}
        self.results = {}
        
        self.best_model = None
        self.feature_names = None

    def prepare_data(self):
        logger.info("📊 Preparing data for modeling...")

        # --- FIX: ENCODE TARGET VARIABLE MANUALLY ---
        # This prevents XGBoost errors and predictor crashes
        if self.df[self.target_col].dtype == 'object':
            logger.info("⚙️ Encoding target variable (Yes/No -> 1/0)...")
            self.df[self.target_col] = self.df[self.target_col].map({'Yes': 1, 'No': 0})

        # Drop rows where target might be NaN (safety check)
        self.df = self.df.dropna(subset=[self.target_col])
        self.df[self.target_col] = self.df[self.target_col].astype(int)

        # Columns to exclude
        exclude_cols = [
            self.target_col, 
            'customerID', 
            'Churn_Original',
        ]
        
        categorical_originals = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'TenureGroup', 'ChargeCategory'
        ]
        exclude_cols.extend(categorical_originals)

        feature_cols = [
            col for col in self.df.columns 
            if col not in exclude_cols 
            and (pd.api.types.is_numeric_dtype(self.df[col]) or col.endswith('_Encoded'))
        ]

        self.X = self.df[feature_cols]
        self.y = self.df[self.target_col]
        self.feature_names = feature_cols

        logger.info(f"   ✅ Final features count: {len(feature_cols)}")
        logger.info(f"   📊 Dataset shape: {self.X.shape}")

    def split_data(self, balance=True):
        logger.info("✂️ Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=self.y
        )
        
        if balance:
            logger.info("⚖️ Applying SMOTE balancing...")
            smote = SMOTE(random_state=RANDOM_STATE)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def initialize_models(self):
        logger.info("🤖 Initializing models...")
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000), # Increased iter
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
        }

    def train_models(self):
        logger.info("🎯 Training models...")
        for name, model in self.models.items():
            try:
                logger.info(f"🔄 Training {name}")
                model.fit(self.X_train, self.y_train)
                
                # Check if model has predict_proba
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                    auc = roc_auc_score(self.y_test, y_pred_proba)
                else:
                    # Fallback for models without probability (rare)
                    y_pred = model.predict(self.X_test)
                    auc = roc_auc_score(self.y_test, y_pred)
                
                self.results[name] = {
                    'model': model,
                    'roc_auc': auc
                }
                logger.info(f"   ROC-AUC: {auc:.4f}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")

    def compare_models(self):
        logger.info("\n📊 Model Comparison")
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'ROC-AUC': [self.results[m]['roc_auc'] for m in self.results]
        })
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        print("\nModel Ranking:\n")
        print(comparison_df)
        return comparison_df

    def hyperparameter_tuning(self, model_name):
        logger.info(f"\n🔧 Tuning {model_name}")
        
        param_grids = {
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1]
            }
        }

        if model_name in param_grids:
            # Re-initialize to ensure clean state
            if model_name == 'XGBoost':
                model = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
            elif model_name == 'Random Forest':
                model = RandomForestClassifier(random_state=RANDOM_STATE)
            elif model_name == 'Gradient Boosting':
                model = GradientBoostingClassifier(random_state=RANDOM_STATE)
            elif model_name == 'LightGBM':
                model = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
            
            grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(self.X_train, self.y_train)
            
            self.best_model = grid.best_estimator_
            logger.info(f"✅ Best Params: {grid.best_params_}")
        
        else:
            logger.warning("Tuning not configured for this model, using default trained model.")
            self.best_model = self.results[model_name]['model']

        return self.best_model

    def save_best_model(self):
        if self.best_model is None:
            logger.error("❌ No model available to save!")
            return

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.best_model, MODEL_DIR / 'best_model.pkl')
        joblib.dump(self.feature_names, MODEL_DIR / 'feature_names.pkl')
        
        logger.info(f"💾 Best model and {len(self.feature_names)} feature names saved successfully!")

    def train_complete_pipeline(self):
        logger.info("🚀 Starting Training Pipeline")
        
        self.prepare_data()
        self.split_data(balance=True)
        self.initialize_models()
        self.train_models()
        
        comparison_df = self.compare_models()
        best_model_name = comparison_df.iloc[0]['Model']
        logger.info(f"🏆 Best Model: {best_model_name}")
        
        self.hyperparameter_tuning(best_model_name)
        self.save_best_model()
        logger.info("✅ Training Pipeline Completed Successfully!")

if __name__ == "__main__":
    df = pd.read_csv('data/processed/engineered_data.csv')
    trainer = ChurnModelTrainer(df)
    trainer.train_complete_pipeline()