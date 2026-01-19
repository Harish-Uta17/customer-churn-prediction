"""
Configuration settings for the project
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'telco_churn.csv'
PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed'

# PostgreSQL Database Configuration
DB_USER = "churn_user"
DB_PASSWORD = "9913933238"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "churn_db"

# PostgreSQL connection URL
DATABASE_URL = f"postgresql://churn_user:9913933238@localhost:5432/churn_db"

# Model paths
MODEL_DIR = BASE_DIR / 'models'
BEST_MODEL_PATH = MODEL_DIR / 'best_model.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
ENCODER_PATH = MODEL_DIR / 'encoder.pkl'

# Logs
LOG_DIR = BASE_DIR / 'logs'
LOG_FILE = LOG_DIR / 'app.log'

# Create directories if they don't exist
for directory in [PROCESSED_DATA_PATH, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Target variable
TARGET_COLUMN = 'Churn'

# Categorical columns
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Numerical columns
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
