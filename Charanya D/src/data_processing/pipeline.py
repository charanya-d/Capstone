"""
Data ingestion and processing pipeline
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import os
import logging
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing class for retail analytics"""
    
    def __init__(self, data_path: str = "data", db_path: str = "retail_analytics.db"):
        self.data_path = Path(data_path)
        self.raw_data_path = self.data_path / "raw"
        self.processed_data_path = self.data_path / "processed"
        self.db_path = db_path
        
        # Create directories if they don't exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self, filename: str = "customer_shopping_data.csv") -> pd.DataFrame:
        """Load raw customer shopping data"""
        try:
            file_path = self.raw_data_path / filename
            
            if not file_path.exists():
                logger.warning(f"Data file not found at {file_path}")
                logger.info("Please download the dataset from: https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset")
                # Create sample data for demonstration
                return self._create_sample_data()
            
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration purposes"""
        np.random.seed(42)
        n_customers = 1000
        n_transactions = 5000
        
        # Generate sample data
        data = {
            'invoice_no': [f'INV{i:06d}' for i in range(1, n_transactions + 1)],
            'customer_id': np.random.choice([f'CUST{i:04d}' for i in range(1, n_customers + 1)], n_transactions),
            'gender': np.random.choice(['Male', 'Female'], n_transactions),
            'age': np.random.normal(35, 12, n_transactions).astype(int),
            'category': np.random.choice(['Clothing', 'Shoes', 'Books', 'Cosmetics', 'Food & Beverage', 'Toys', 'Technology', 'Souvenir'], n_transactions),
            'quantity': np.random.poisson(2, n_transactions) + 1,
            'price': np.random.exponential(50, n_transactions) + 10,
            'payment_method': np.random.choice(['Cash', 'Credit Card', 'Debit Card'], n_transactions),
            'invoice_date': pd.date_range(start='2022-01-01', end='2023-12-31', periods=n_transactions),
            'shopping_mall': np.random.choice(['Mall_A', 'Mall_B', 'Mall_C', 'Mall_D', 'Mall_E'], n_transactions)
        }
        
        df = pd.DataFrame(data)
        
        # Apply realistic business logic
        df.loc[df['age'] < 18, 'age'] = 18
        df.loc[df['age'] > 80, 'age'] = 80
        df['price'] = df['price'].round(2)
        
        # Save sample data
        sample_file_path = self.raw_data_path / "customer_shopping_data.csv"
        df.to_csv(sample_file_path, index=False)
        logger.info(f"Created sample dataset with {len(df)} records at {sample_file_path}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        logger.info("Starting data cleaning...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.dropna()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
        
        # Convert data types
        if 'invoice_date' in df_clean.columns:
            df_clean['invoice_date'] = pd.to_datetime(df_clean['invoice_date'])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Data validation
        df_clean = df_clean[df_clean['quantity'] > 0]
        df_clean = df_clean[df_clean['price'] > 0]
        
        # Feature engineering
        df_clean['total_amount'] = df_clean['quantity'] * df_clean['price']
        df_clean['year'] = df_clean['invoice_date'].dt.year
        df_clean['month'] = df_clean['invoice_date'].dt.month
        df_clean['quarter'] = df_clean['invoice_date'].dt.quarter
        df_clean['day_of_week'] = df_clean['invoice_date'].dt.dayofweek
        df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Data cleaning completed. Records: {len(df_clean)}")
        return df_clean
    
    def calculate_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) scores"""
        logger.info("Calculating RFM scores...")
        
        # Get the latest date in the dataset
        current_date = df['invoice_date'].max()
        
        # Calculate RFM metrics
        rfm = df.groupby('customer_id').agg({
            'invoice_date': lambda x: (current_date - x.max()).days,  # Recency
            'invoice_no': 'nunique',  # Frequency
            'total_amount': 'sum'  # Monetary
        })
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        
        # Calculate RFM scores using quantiles
        rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        
        # Combine RFM scores
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        # Create customer segments
        def segment_customers(row):
            if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['rfm_score'] in ['533', '532', '431', '432', '423', '353', '352', '351']:
                return 'New Customers'
            elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115']:
                return 'Cannot Lose Them'
            else:
                return 'Others'
        
        rfm['segment'] = rfm.apply(segment_customers, axis=1)
        rfm = rfm.reset_index()
        
        logger.info(f"RFM analysis completed for {len(rfm)} customers")
        return rfm
    
    def save_to_database(self, df: pd.DataFrame, table_name: str):
        """Save DataFrame to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            logger.info(f"Data saved to {table_name} table in {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def run_pipeline(self):
        """Run the complete data processing pipeline"""
        logger.info("Starting data processing pipeline...")
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Clean data
        clean_data = self.clean_data(raw_data)
        
        # Calculate RFM scores
        rfm_data = self.calculate_rfm_scores(clean_data)
        
        # Save processed data
        clean_data.to_csv(self.processed_data_path / "clean_customer_data.csv", index=False)
        rfm_data.to_csv(self.processed_data_path / "rfm_analysis.csv", index=False)
        
        # Save to database
        self.save_to_database(clean_data, "customers")
        self.save_to_database(rfm_data, "customer_segments")
        
        logger.info("Data processing pipeline completed successfully!")
        
        return {
            "clean_data": clean_data,
            "rfm_data": rfm_data,
            "summary": {
                "total_customers": clean_data['customer_id'].nunique(),
                "total_transactions": len(clean_data),
                "date_range": f"{clean_data['invoice_date'].min()} to {clean_data['invoice_date'].max()}",
                "total_revenue": clean_data['total_amount'].sum()
            }
        }

if __name__ == "__main__":
    processor = DataProcessor()
    results = processor.run_pipeline()
    print("Pipeline completed successfully!")
    print(f"Summary: {results['summary']}")
