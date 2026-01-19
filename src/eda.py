"""
Exploratory Data Analysis module
Modified to run automatically without freezing the pipeline.
"""
import pandas as pd
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindow backend.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class ChurnEDA:
    """Perform comprehensive EDA on churn data"""
    
    def __init__(self, df: pd.DataFrame, save_dir: Path = None):
        self.df = df
        # Ensure path is a Path object and exists
        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path('data/processed/eda_plots')
            
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_target_distribution(self):
        """Analyze churn distribution"""
        logger.info("📊 Analyzing target variable distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        churn_counts = self.df['Churn'].value_counts()
        axes[0].bar(['No Churn', 'Churn'], churn_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count')
        for i, v in enumerate(churn_counts.values):
            axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # Pie chart
        colors = ['#2ecc71', '#e74c3c']
        axes[1].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                    colors=colors, startangle=90)
        axes[1].set_title('Churn Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / 'target_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # Close to free memory
        
        logger.info(f"   Total customers: {len(self.df)}")
        logger.info(f"   Churned: {(self.df['Churn'] == 1).sum()} ({(self.df['Churn'] == 1).mean()*100:.2f}%)")
        logger.info(f"   Retained: {(self.df['Churn'] == 0).sum()} ({(self.df['Churn'] == 0).mean()*100:.2f}%)")
    
    def analyze_numerical_features(self):
        """Analyze numerical features"""
        logger.info("📊 Analyzing numerical features...")
        
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols):
            # Distribution plot
            axes[idx].hist(self.df[col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col} Distribution', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            
            # Box plot by churn
            axes[idx + 3].boxplot([self.df[self.df['Churn'] == 0][col].dropna(),
                                   self.df[self.df['Churn'] == 1][col].dropna()],
                                  labels=['No Churn', 'Churn'],
                                  patch_artist=True)
            axes[idx + 3].set_title(f'{col} by Churn Status', fontweight='bold')
            axes[idx + 3].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'numerical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_categorical_features(self):
        """Analyze categorical features with churn rates"""
        logger.info("📊 Analyzing categorical features...")
        
        categorical_cols = ['Contract', 'InternetService', 'PaymentMethod', 
                           'gender', 'SeniorCitizen', 'Partner']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(categorical_cols):
            # Calculate churn rate for each category
            churn_rate = self.df.groupby(col)['Churn'].mean().sort_values(ascending=False)
            
            axes[idx].bar(range(len(churn_rate)), churn_rate.values, color='coral')
            axes[idx].set_xticks(range(len(churn_rate)))
            axes[idx].set_xticklabels(churn_rate.index, rotation=45, ha='right')
            axes[idx].set_title(f'Churn Rate by {col}', fontweight='bold')
            axes[idx].set_ylabel('Churn Rate')
            axes[idx].set_ylim(0, 1)
            
            # Add value labels
            for i, v in enumerate(churn_rate.values):
                axes[idx].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def correlation_analysis(self):
        """Analyze correlations"""
        logger.info("📊 Analyzing correlations...")
        
        # Select numerical columns and target
        corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
        corr_matrix = self.df[corr_cols].corr()
        
        plt.figure(figsize=(10, 8))
        # 
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def tenure_analysis(self):
        """Detailed tenure analysis"""
        logger.info("📊 Analyzing tenure patterns...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Tenure distribution by churn
        for churn_status in [0, 1]:
            label = 'Churned' if churn_status == 1 else 'Retained'
            color = '#e74c3c' if churn_status == 1 else '#2ecc71'
            axes[0].hist(self.df[self.df['Churn'] == churn_status]['tenure'],
                         bins=30, alpha=0.6, label=label, color=color)
        
        axes[0].set_xlabel('Tenure (months)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Tenure Distribution by Churn Status', fontweight='bold')
        axes[0].legend()
        
        # Churn rate by tenure groups
        # Ensure we use a copy or check if column exists to avoid warnings
        if 'TenureGroup' not in self.df.columns:
            self.df['TenureGroup'] = pd.cut(self.df['tenure'], 
                                          bins=[0, 12, 24, 36, 48, 72],
                                          labels=['0-12', '12-24', '24-36', '36-48', '48-72'])
            
        tenure_churn = self.df.groupby('TenureGroup', observed=False)['Churn'].mean()
        
        axes[1].bar(range(len(tenure_churn)), tenure_churn.values, color='steelblue')
        axes[1].set_xticks(range(len(tenure_churn)))
        axes[1].set_xticklabels(tenure_churn.index)
        axes[1].set_xlabel('Tenure Group (months)')
        axes[1].set_ylabel('Churn Rate')
        axes[1].set_title('Churn Rate by Tenure Groups', fontweight='bold')
        
        for i, v in enumerate(tenure_churn.values):
            axes[1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'tenure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_insights(self):
        """Generate key insights from data"""
        print("\n" + "="*60)
        print("🔍 KEY INSIGHTS FROM EDA")
        print("="*60)
        
        # Churn rate
        churn_rate = (self.df['Churn'] == 1).mean()
        print(f"\n📌 Overall Churn Rate: {churn_rate:.2%}")
        
        # Churn by contract type
        print("\n📌 Churn Rate by Contract Type:")
        contract_churn = self.df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
        for contract, rate in contract_churn.items():
            print(f"   {contract}: {rate:.2%}")
        
        # Average tenure
        print(f"\n📌 Average Tenure:")
        print(f"   Churned customers: {self.df[self.df['Churn'] == 1]['tenure'].mean():.1f} months")
        print(f"   Retained customers: {self.df[self.df['Churn'] == 0]['tenure'].mean():.1f} months")
        
        # Monthly charges
        print(f"\n📌 Average Monthly Charges:")
        print(f"   Churned customers: ${self.df[self.df['Churn'] == 1]['MonthlyCharges'].mean():.2f}")
        print(f"   Retained customers: ${self.df[self.df['Churn'] == 0]['MonthlyCharges'].mean():.2f}")
    
    def run_complete_eda(self):
        """Run complete EDA pipeline"""
        logger.info("\n🚀 Starting comprehensive EDA...\n")
        
        self.analyze_target_distribution()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        self.correlation_analysis()
        self.tenure_analysis()
        self.generate_insights()
        
        logger.info(f"\n✅ EDA completed! Plots saved to: {self.save_dir}")


# Example usage
if __name__ == "__main__":
    # Load cleaned data
    try:
        df = pd.read_csv('data/processed/cleaned_data.csv')
        # Perform EDA
        eda = ChurnEDA(df)
        eda.run_complete_eda()
    except FileNotFoundError:
        print("Cleaned data not found. Please run the Data Cleaner first.")