"""
Data loading module - Load data from CSV and store in PostgreSQL database
"""
import pandas as pd
from pathlib import Path
import logging
from sqlalchemy import create_engine
from src.config import RAW_DATA_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage customer churn data"""

    def __init__(self, csv_path: Path = RAW_DATA_PATH):
        self.csv_path = csv_path

        # PostgreSQL connection details
        self.db_url = "postgresql://churn_user:9913933238@localhost:5432/churn_db"

        # Create SQLAlchemy engine
        self.engine = create_engine(self.db_url)

        self.df = None

    def load_from_csv(self) -> pd.DataFrame:
        """
        Load data from CSV file

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"✅ Data loaded successfully from {self.csv_path}")
            logger.info(f"📊 Dataset shape: {self.df.shape}")
            logger.info(f"📋 Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            logger.error(f"❌ File not found: {self.csv_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise

    def save_to_database(self, table_name: str = 'customers'):
        """
        Save dataframe to PostgreSQL database

        Args:
            table_name: Name of the table to create
        """
        if self.df is None:
            logger.error("❌ No data loaded. Call load_from_csv() first.")
            return

        try:
            # Save to PostgreSQL using SQLAlchemy engine
            self.df.to_sql(table_name, self.engine, if_exists='replace', index=False)

            logger.info(f"✅ Data saved to PostgreSQL database: churn_db")
            logger.info(f"📊 Table: {table_name}")
            logger.info(f"📝 Records: {len(self.df)}")

        except Exception as e:
            logger.error(f"❌ Error saving to PostgreSQL database: {str(e)}")
            raise

    def load_from_database(self, query: str = None, table_name: str = 'customers') -> pd.DataFrame:
        """
        Load data from PostgreSQL database

        Args:
            query: Custom SQL query (optional)
            table_name: Table name if no custom query

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            if query is None:
                query = f"SELECT * FROM {table_name}"

            self.df = pd.read_sql_query(query, self.engine)

            logger.info(f"✅ Data loaded from PostgreSQL database")
            logger.info(f"📊 Records retrieved: {len(self.df)}")

            return self.df

        except Exception as e:
            logger.error(f"❌ Error loading from PostgreSQL database: {str(e)}")
            raise

    def get_data_info(self):
        """Print detailed information about the dataset"""
        if self.df is None:
            logger.error("❌ No data loaded.")
            return

        print("\n" + "=" * 50)
        print("📊 DATASET INFORMATION")
        print("=" * 50)
        print(f"\n📏 Shape: {self.df.shape}")
        print(f"📋 Columns: {self.df.shape[1]}")
        print(f"📝 Rows: {self.df.shape[0]}")

        print("\n📊 Data Types:")
        print(self.df.dtypes)

        print("\n🔍 First 5 Rows:")
        print(self.df.head())

        print("\n📈 Statistical Summary:")
        print(self.df.describe())

        print("\n❓ Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")

        print("\n🎯 Target Variable Distribution:")
        if 'Churn' in self.df.columns:
            print(self.df['Churn'].value_counts())
            print(f"\nChurn Rate: {(self.df['Churn'] == 'Yes').mean() * 100:.2f}%")


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = DataLoader()

    # Load from CSV
    df = loader.load_from_csv()

    # Get info
    loader.get_data_info()

    # Save to PostgreSQL database
    loader.save_to_database()

    # Load from database with custom query
    churned_customers = loader.load_from_database(
        query="SELECT * FROM customers WHERE \"Churn\" = 'Yes'"
    )

    print(f"\n📊 Churned customers: {len(churned_customers)}")
