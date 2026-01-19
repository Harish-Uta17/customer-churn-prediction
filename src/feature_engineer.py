import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    
    def __init__(self, df):
        self.df = df.copy()
        
    def create_tenure_features(self):
        # Prevent division by zero
        self.df['tenure'] = self.df['tenure'].replace(0, 1)
        
        # Binning tenure
        self.df['TenureGroup'] = pd.cut(
            self.df['tenure'], 
            bins=[0, 12, 24, 48, 60, 1000], 
            labels=['0-12', '12-24', '24-48', '48-60', '60+']
        )
        return self.df

    def create_charge_features(self):
        # Interaction between charges and tenure
        self.df['AverageChargePerMonth'] = self.df['TotalCharges'] / self.df['tenure']
        self.df['ChargeDifference'] = self.df['MonthlyCharges'] - self.df['AverageChargePerMonth']
        self.df['IncreaseInCharge'] = (self.df['ChargeDifference'] > 0).astype(int)
        
        # Binning Monthly Charges
        self.df['ChargeCategory'] = pd.cut(
            self.df['MonthlyCharges'], 
            bins=[0, 30, 70, 1000], 
            labels=['Low', 'Medium', 'High']
        )
        return self.df
        
    def create_service_features(self):
        # Count total services subscribed
        services = ['PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Clean 'No internet service' to 'No' for counting purposes
        # We perform this on a copy to avoid SettingWithCopy warnings
        temp_df = self.df[services].copy()
        temp_df = temp_df.replace({'No internet service': 'No', 'No phone service': 'No'})
        
        # FIX: Correct way to sum 'Yes' values across columns (axis=1)
        # We convert to boolean (True/False) then sum (True=1, False=0)
        self.df['TotalServices'] = (temp_df.apply(lambda x: x.str.lower()) == 'yes').sum(axis=1)
        
        self.df['HasInternet'] = self.df['InternetService'].apply(
            lambda x: 0 if x == 'No' else 1
        )
        
        self.df['HasPhone'] = self.df['PhoneService'].apply(
            lambda x: 1 if x == 'Yes' else 0
        )
        return self.df

    def engineer_all_features(self, scale=False):
        """
        Main pipeline. 
        scale=False ensures we train and predict on raw readable numbers.
        """
        # 1. Create Features
        self.create_tenure_features()
        self.create_charge_features()
        self.create_service_features()
        
        # 2. One-Hot Encoding
        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod', 'TenureGroup', 
            'ChargeCategory'
        ]
        
        # Only encode columns that actually exist
        existing_cols = [col for col in categorical_cols if col in self.df.columns]
        
        self.df = pd.get_dummies(
            self.df, 
            columns=existing_cols, 
            drop_first=False 
        )
        
        return self.df