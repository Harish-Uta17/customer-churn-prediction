"""
Data cleaning and preprocessing module
"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from src.config import TARGET_COLUMN, CATEGORICAL_COLS, NUMERICAL_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess customer churn data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_report = {}
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Identify and handle missing values
        """
        logger.info("🔍 Checking for missing values...")
        
        missing_before = self.df.isnull().sum().sum()
        self.cleaning_report['missing_values_before'] = missing_before
        
        # Check for ' ' (space) values which represent missing
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                space_count = (self.df[col] == ' ').sum()
                if space_count > 0:
                    logger.info(f"   Found {space_count} whitespace values in '{col}'")
                    self.df[col] = self.df[col].replace(' ', np.nan)
        
        # Handle TotalCharges (known issue in this dataset)
        if 'TotalCharges' in self.df.columns:
            # Convert to numeric (will create NaN for non-numeric)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            
            # Fill NaN with 0 for customers with 0 tenure
            mask = (self.df['TotalCharges'].isna()) & (self.df['tenure'] == 0)
            self.df.loc[mask, 'TotalCharges'] = 0
            
            # For others, fill with median
            median_charges = self.df['TotalCharges'].median()
            self.df['TotalCharges'].fillna(median_charges, inplace=True)
            
            logger.info(f"✅ Handled TotalCharges missing values")
        
        missing_after = self.df.isnull().sum().sum()
        self.cleaning_report['missing_values_after'] = missing_after
        
        logger.info(f"✅ Missing values: {missing_before} → {missing_after}")
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows"""
        logger.info("🔍 Checking for duplicates...")
        
        duplicates_before = self.df.duplicated().sum()
        self.cleaning_report['duplicates_removed'] = duplicates_before
        
        if duplicates_before > 0:
            self.df = self.df.drop_duplicates()
            logger.info(f"✅ Removed {duplicates_before} duplicate rows")
        else:
            logger.info("✅ No duplicates found")
        
        return self.df
    
    def fix_data_types(self) -> pd.DataFrame:
        """
        Convert columns to appropriate data types
        """
        logger.info("🔧 Fixing data types...")
        
        # Convert SeniorCitizen to string for consistency
        if 'SeniorCitizen' in self.df.columns:
            self.df['SeniorCitizen'] = self.df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # Ensure numerical columns are numeric
        for col in NUMERICAL_COLS:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        logger.info("✅ Data types fixed")
        return self.df
    
    def handle_outliers(self, method='iqr', threshold=3) -> pd.DataFrame:
        """
        Detect and handle outliers in numerical columns
        
        Args:
            method: 'iqr' or 'zscore'
            threshold: Threshold for z-score method
        """
        logger.info(f"🔍 Detecting outliers using {method} method...")
        
        outliers_count = 0
        
        for col in NUMERICAL_COLS:
            if col not in self.df.columns:
                continue
            
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outliers_count += outliers
                
                if outliers > 0:
                    logger.info(f"   {col}: {outliers} outliers detected")
                    # Cap outliers instead of removing
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = (z_scores > threshold).sum()
                outliers_count += outliers
                
                if outliers > 0:
                    logger.info(f"   {col}: {outliers} outliers detected")
        
        self.cleaning_report['outliers_handled'] = outliers_count
        logger.info(f"✅ Handled {outliers_count} outliers")
        
        return self.df
    
    def clean_categorical_values(self) -> pd.DataFrame:
        """
        Clean and standardize categorical values
        """
        logger.info("🧹 Cleaning categorical values...")
        
        for col in CATEGORICAL_COLS:
            if col not in self.df.columns:
                continue
            
            # Strip whitespace
            self.df[col] = self.df[col].str.strip()
            
            # Standardize 'No internet service' and 'No phone service'
            self.df[col] = self.df[col].replace({
                'No internet service': 'No',
                'No phone service': 'No'
            })
        
        logger.info("✅ Categorical values cleaned")
        return self.df
    
    def encode_target(self) -> pd.DataFrame:
        """
        Encode target variable (Churn: Yes/No → 1/0)
        """
        logger.info("🎯 Encoding target variable...")
        
        if TARGET_COLUMN in self.df.columns:
            self.df[f'{TARGET_COLUMN}_Original'] = self.df[TARGET_COLUMN]
            self.df[TARGET_COLUMN] = (self.df[TARGET_COLUMN] == 'Yes').astype(int)
            
            logger.info(f"✅ Target encoded: Yes → 1, No → 0")
            logger.info(f"   Churn distribution: {self.df[TARGET_COLUMN].value_counts().to_dict()}")
        
        return self.df
    
    def get_cleaning_report(self):
        """Print cleaning report"""
        print("\n" + "="*50)
        print("🧹 DATA CLEANING REPORT")
        print("="*50)
        print(f"\n📊 Original shape: {self.original_shape}")
        print(f"📊 Final shape: {self.df.shape}")
        print(f"\n📋 Cleaning Summary:")
        for key, value in self.cleaning_report.items():
            print(f"   {key}: {value}")
    
    def clean_all(self) -> pd.DataFrame:
        """
        Execute complete cleaning pipeline
        """
        logger.info("\n🚀 Starting data cleaning pipeline...")
        
        self.handle_missing_values()
        self.remove_duplicates()
        self.fix_data_types()
        self.clean_categorical_values()
        self.handle_outliers()
        self.encode_target()
        
        self.get_cleaning_report()
        
        logger.info("\n✅ Data cleaning completed!")
        return self.df


# Example usage
if __name__ == "__main__":
    from src.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_from_csv()
    
    # Clean data
    cleaner = DataCleaner(df)
    cleaned_df = cleaner.clean_all()
    
    # Save cleaned data
    cleaned_df.to_csv('data/processed/cleaned_data.csv', index=False)
    logger.info("\n💾 Cleaned data saved to 'data/processed/cleaned_data.csv'")