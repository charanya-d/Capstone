"""
Machine Learning models for customer segmentation, profitability prediction, and forecasting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import logging
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CustomerSegmentationModel:
    """ML model for customer segmentation and lifetime value prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.kmeans_model = None
        self.classification_model = None
        self.regression_model = None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        features_df = df.copy()
        
        # Create RFM features
        if 'invoice_date' in features_df.columns:
            reference_date = features_df['invoice_date'].max()
            customer_features = features_df.groupby('customer_id').agg({
                'invoice_date': lambda x: (reference_date - x.max()).days,  # Recency
                'invoice_no': 'nunique',  # Frequency
                'total_amount': ['sum', 'mean', 'std'],  # Monetary features
                'quantity': ['sum', 'mean'],
                'age': 'first',
                'gender': 'first',
                'category': lambda x: x.mode().iloc[0],
                'payment_method': lambda x: x.mode().iloc[0],
                'is_weekend': 'mean'
            }).reset_index()
            
            # Flatten column names
            customer_features.columns = ['customer_id', 'recency', 'frequency', 'total_spent', 
                                       'avg_spend', 'spend_std', 'total_quantity', 'avg_quantity',
                                       'age', 'gender', 'preferred_category', 'preferred_payment', 'weekend_preference']
            
            # Fill missing values
            customer_features['spend_std'] = customer_features['spend_std'].fillna(0)
            
            # Create additional features
            customer_features['spending_consistency'] = 1 / (1 + customer_features['spend_std'])
            customer_features['customer_value_score'] = (
                customer_features['frequency'] * customer_features['total_spent'] / 
                (customer_features['recency'] + 1)
            )
            
            return customer_features
        
        return features_df
    
    def train_segmentation_model(self, df: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
        """Train customer segmentation model using K-means clustering"""
        logger.info("Training customer segmentation model...")
        
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Select numerical features for clustering
        numerical_features = ['recency', 'frequency', 'total_spent', 'avg_spend', 
                            'total_quantity', 'age', 'weekend_preference', 'customer_value_score']
        
        X = features_df[numerical_features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-means model
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        # Add cluster labels
        features_df['cluster'] = clusters
        features_df['segment'] = features_df['cluster'].map(self._get_segment_names())
        
        # Train classification model to predict segments
        categorical_features = ['gender', 'preferred_category', 'preferred_payment']
        for feature in categorical_features:
            features_df[f'{feature}_encoded'] = self.label_encoder.fit_transform(features_df[feature].astype(str))
        
        X_classification = features_df[numerical_features + [f'{f}_encoded' for f in categorical_features]].values
        y_classification = features_df['cluster'].values
        
        self.classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classification_model.fit(X_classification, y_classification)
        
        # Evaluate clustering quality
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        
        self.is_trained = True
        
        return {
            'model': self.kmeans_model,
            'segmented_data': features_df,
            'silhouette_score': silhouette_avg,
            'cluster_centers': self.kmeans_model.cluster_centers_,
            'feature_importance': dict(zip(numerical_features, self.classification_model.feature_importances_[:len(numerical_features)]))
        }
    
    def train_clv_prediction_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train Customer Lifetime Value prediction model"""
        logger.info("Training CLV prediction model...")
        
        features_df = self.prepare_features(df)
        
        # Calculate CLV target
        features_df['clv'] = self._calculate_clv(features_df)
        
        # Prepare features for regression
        feature_cols = ['recency', 'frequency', 'total_spent', 'avg_spend', 'total_quantity', 
                       'age', 'weekend_preference', 'customer_value_score']
        
        X = features_df[feature_cols].values
        y = features_df['clv'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        self.regression_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.regression_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.regression_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': self.regression_model,
            'mse': mse,
            'r2_score': r2,
            'feature_importance': dict(zip(feature_cols, self.regression_model.feature_importances_)),
            'predictions': y_pred[:10].tolist()  # Sample predictions
        }
    
    def predict_customer_segment(self, customer_data: pd.DataFrame) -> np.ndarray:
        """Predict customer segment for new data"""
        if not self.is_trained or self.classification_model is None:
            raise ValueError("Model not trained. Call train_segmentation_model first.")
        
        features = self.prepare_features(customer_data)
        numerical_features = ['recency', 'frequency', 'total_spent', 'avg_spend', 
                            'total_quantity', 'age', 'weekend_preference', 'customer_value_score']
        
        X = features[numerical_features].values
        X_scaled = self.scaler.transform(X)
        
        return self.kmeans_model.predict(X_scaled)
    
    def predict_clv(self, customer_data: pd.DataFrame) -> np.ndarray:
        """Predict Customer Lifetime Value"""
        if self.regression_model is None:
            raise ValueError("CLV model not trained. Call train_clv_prediction_model first.")
        
        features = self.prepare_features(customer_data)
        feature_cols = ['recency', 'frequency', 'total_spent', 'avg_spend', 'total_quantity', 
                       'age', 'weekend_preference', 'customer_value_score']
        
        X = features[feature_cols].values
        return self.regression_model.predict(X)
    
    def _get_segment_names(self) -> Dict[int, str]:
        """Map cluster numbers to meaningful segment names"""
        return {
            0: 'Price Conscious',
            1: 'Loyal Customers',
            2: 'High Value',
            3: 'Potential Loyalists',
            4: 'At Risk'
        }
    
    def _calculate_clv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Customer Lifetime Value"""
        # Simplified CLV calculation
        purchase_frequency = df['frequency'] / 365
        avg_order_value = df['total_spent'] / df['frequency']
        customer_lifespan = 365 / (df['recency'] + 1)
        
        clv = avg_order_value * purchase_frequency * customer_lifespan
        return clv
    
    def save_models(self, models_path: str = "models"):
        """Save trained models"""
        import os
        os.makedirs(models_path, exist_ok=True)
        
        if self.kmeans_model:
            joblib.dump(self.kmeans_model, f"{models_path}/customer_segmentation_kmeans.pkl")
        if self.classification_model:
            joblib.dump(self.classification_model, f"{models_path}/customer_segmentation_classifier.pkl")
        if self.regression_model:
            joblib.dump(self.regression_model, f"{models_path}/customer_clv_model.pkl")
        
        joblib.dump(self.scaler, f"{models_path}/scaler.pkl")
        
        logger.info(f"Models saved to {models_path}")
    
    def load_models(self, models_path: str = "models"):
        """Load trained models"""
        try:
            self.kmeans_model = joblib.load(f"{models_path}/customer_segmentation_kmeans.pkl")
            self.classification_model = joblib.load(f"{models_path}/customer_segmentation_classifier.pkl")
            self.regression_model = joblib.load(f"{models_path}/customer_clv_model.pkl")
            self.scaler = joblib.load(f"{models_path}/scaler.pkl")
            self.is_trained = True
            logger.info(f"Models loaded from {models_path}")
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")

class SalesForecastingModel:
    """ML model for sales forecasting and demand prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series features for forecasting"""
        # Aggregate daily sales
        daily_sales = df.groupby('invoice_date').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        daily_sales = daily_sales.sort_values('invoice_date')
        
        # Create time-based features
        daily_sales['year'] = daily_sales['invoice_date'].dt.year
        daily_sales['month'] = daily_sales['invoice_date'].dt.month
        daily_sales['day'] = daily_sales['invoice_date'].dt.day
        daily_sales['day_of_week'] = daily_sales['invoice_date'].dt.dayofweek
        daily_sales['quarter'] = daily_sales['invoice_date'].dt.quarter
        daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)
        daily_sales['is_month_end'] = daily_sales['invoice_date'].dt.is_month_end.astype(int)
        
        # Create lag features
        for lag in [1, 7, 30]:
            daily_sales[f'sales_lag_{lag}'] = daily_sales['total_amount'].shift(lag)
            daily_sales[f'quantity_lag_{lag}'] = daily_sales['quantity'].shift(lag)
        
        # Create rolling window features
        for window in [7, 30]:
            daily_sales[f'sales_rolling_mean_{window}'] = daily_sales['total_amount'].rolling(window=window).mean()
            daily_sales[f'sales_rolling_std_{window}'] = daily_sales['total_amount'].rolling(window=window).std()
        
        # Fill missing values
        daily_sales = daily_sales.fillna(method='bfill').fillna(0)
        
        return daily_sales
    
    def train_forecasting_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train sales forecasting model"""
        logger.info("Training sales forecasting model...")
        
        # Prepare features
        time_series_df = self.prepare_time_series_features(df)
        
        # Select features
        feature_cols = ['year', 'month', 'day', 'day_of_week', 'quarter', 'is_weekend', 'is_month_end',
                       'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'quantity_lag_1', 'quantity_lag_7', 'quantity_lag_30',
                       'sales_rolling_mean_7', 'sales_rolling_mean_30', 'sales_rolling_std_7', 'sales_rolling_std_30']
        
        X = time_series_df[feature_cols].values
        y = time_series_df['total_amount'].values
        
        # Split data (temporal split)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        self.is_trained = True
        
        return {
            'model': self.model,
            'mse': mse,
            'r2_score': r2,
            'mape': mape,
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
            'test_predictions': y_pred[:10].tolist()
        }
    
    def forecast_sales(self, days_ahead: int = 30) -> pd.DataFrame:
        """Generate sales forecast"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_forecasting_model first.")
        
        # This is a simplified forecast - in practice, you'd need historical data
        # to generate proper lag features for future dates
        
        # Generate future dates
        future_dates = pd.date_range(start=datetime.now(), periods=days_ahead, freq='D')
        
        # Create basic features for future dates
        future_features = pd.DataFrame({
            'invoice_date': future_dates,
            'year': future_dates.year,
            'month': future_dates.month,
            'day': future_dates.day,
            'day_of_week': future_dates.dayofweek,
            'quarter': future_dates.quarter,
            'is_weekend': future_dates.dayofweek.isin([5, 6]).astype(int),
            'is_month_end': future_dates.is_month_end.astype(int)
        })
        
        # For lag features, use historical averages (simplified approach)
        future_features['sales_lag_1'] = 5000  # Placeholder
        future_features['sales_lag_7'] = 5000
        future_features['sales_lag_30'] = 5000
        future_features['quantity_lag_1'] = 100
        future_features['quantity_lag_7'] = 100
        future_features['quantity_lag_30'] = 100
        future_features['sales_rolling_mean_7'] = 5000
        future_features['sales_rolling_mean_30'] = 5000
        future_features['sales_rolling_std_7'] = 1000
        future_features['sales_rolling_std_30'] = 1500
        
        feature_cols = ['year', 'month', 'day', 'day_of_week', 'quarter', 'is_weekend', 'is_month_end',
                       'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'quantity_lag_1', 'quantity_lag_7', 'quantity_lag_30',
                       'sales_rolling_mean_7', 'sales_rolling_mean_30', 'sales_rolling_std_7', 'sales_rolling_std_30']
        
        X_future = future_features[feature_cols].values
        X_future_scaled = self.scaler.transform(X_future)
        
        predictions = self.model.predict(X_future_scaled)
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions,
            'confidence_lower': predictions * 0.9,  # Simplified confidence intervals
            'confidence_upper': predictions * 1.1
        })
        
        return forecast_df
