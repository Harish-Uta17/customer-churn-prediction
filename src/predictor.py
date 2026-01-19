"""
Prediction pipeline for new customers - BATCH SUPPORTED
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from src.config import MODEL_DIR
from src.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ChurnPredictor:
    """Make predictions on new customer data"""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.load_artifacts()

    def load_artifacts(self):
        try:
            self.model = joblib.load(MODEL_DIR / 'best_model.pkl')
            self.feature_names = joblib.load(MODEL_DIR / 'feature_names.pkl')
            logger.info("✅ Model and feature names loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading artifacts: {e}")
            raise

    def clean_input(self, data: dict) -> dict:
        """Clean single user input to match Training Data Categories EXACTLY."""
        cleaned = {}
        
        # Mappings
        yes_no_map = {'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'}
        
        # Apply cleanups (Safe defaults provided)
        cleaned['gender'] = 'Male' if str(data.get('gender', '')).lower().startswith('m') else 'Female'
        cleaned['SeniorCitizen'] = int(data.get('SeniorCitizen', 0))
        cleaned['Partner'] = yes_no_map.get(str(data.get('Partner', '')).lower(), 'No')
        cleaned['Dependents'] = yes_no_map.get(str(data.get('Dependents', '')).lower(), 'No')
        cleaned['PhoneService'] = yes_no_map.get(str(data.get('PhoneService', '')).lower(), 'No')
        cleaned['PaperlessBilling'] = yes_no_map.get(str(data.get('PaperlessBilling', '')).lower(), 'No')
        
        # Specific mappings
        cleaned['MultipleLines'] = 'Yes' if str(data.get('MultipleLines', '')).lower() == 'yes' else 'No'
        
        internet = str(data.get('InternetService', '')).lower()
        cleaned['InternetService'] = 'Fiber optic' if 'fiber' in internet else 'DSL' if 'dsl' in internet else 'No'
        
        cleaned['Contract'] = data.get('Contract', 'Month-to-month') # Default
        
        payment = str(data.get('PaymentMethod', '')).lower()
        if 'electronic' in payment: cleaned['PaymentMethod'] = 'Electronic check'
        elif 'mailed' in payment: cleaned['PaymentMethod'] = 'Mailed check'
        elif 'bank' in payment: cleaned['PaymentMethod'] = 'Bank transfer (automatic)'
        elif 'card' in payment: cleaned['PaymentMethod'] = 'Credit card (automatic)'
        else: cleaned['PaymentMethod'] = 'Electronic check'

        # Services
        for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']:
            val = str(data.get(col, '')).lower()
            cleaned[col] = 'Yes' if val == 'yes' else 'No'

        # Numerics
        cleaned['tenure'] = int(data.get('tenure', 0))
        cleaned['MonthlyCharges'] = float(data.get('MonthlyCharges', 0.0))
        cleaned['TotalCharges'] = float(data.get('TotalCharges', 0.0))

        return cleaned

    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process input EXACTLY like training pipeline"""
        engineer = FeatureEngineer(df)
        processed_df = engineer.engineer_all_features(scale=False)
        
        # Align features
        feature_df = processed_df.reindex(columns=self.feature_names, fill_value=0)
        
        if feature_df.isnull().values.any():
            feature_df = feature_df.fillna(0)
            
        return feature_df

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Batch-capable prediction method.
        Returns LISTS of results.
        """
        logger.info(f"🔮 Making predictions for {len(df)} customers...")
        
        processed_df = self.preprocess_input(df)
        
        predictions = self.model.predict(processed_df)
        probabilities = self.model.predict_proba(processed_df)[:, 1]

        # Return lists (serializable for API)
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'risk_levels': [
                'High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
                for p in probabilities
            ]
        }

    def predict_single_customer(self, customer_data: dict) -> dict:
        """Wrapper for single customer prediction"""
        cleaned_data = self.clean_input(customer_data)
        df = pd.DataFrame([cleaned_data])
        
        results = self.predict(df)
        
        # Extract single result from the lists
        return {
            'prediction': int(results['predictions'][0]),
            'probability': float(results['probabilities'][0]),
            'risk_level': results['risk_levels'][0]
        }