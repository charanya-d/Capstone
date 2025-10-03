"""
Customer analytics and segmentation module
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class CustomerAnalytics:
    """Customer analytics and segmentation engine"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        
    def perform_rfm_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive RFM analysis"""
        logger.info("Performing RFM analysis...")
        
        # Calculate reference date
        reference_date = df['invoice_date'].max()
        
        # Calculate RFM metrics
        rfm = df.groupby('customer_id').agg({
            'invoice_date': lambda x: (reference_date - x.max()).days,  # Recency
            'invoice_no': 'nunique',  # Frequency
            'total_amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Calculate percentile ranks for scoring
        rfm['recency_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm['monetary_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        
        # Create RFM segments
        rfm['rfm_segment'] = self._create_rfm_segments(rfm)
        
        # Calculate customer lifetime value
        rfm['clv'] = self._calculate_clv(rfm)
        
        return rfm
    
    def _create_rfm_segments(self, rfm: pd.DataFrame) -> pd.Series:
        """Create meaningful customer segments based on RFM scores"""
        segments = []
        
        for _, row in rfm.iterrows():
            r, f, m = int(row['recency_score']), int(row['frequency_score']), int(row['monetary_score'])
            
            if r >= 4 and f >= 4 and m >= 4:
                segments.append('Champions')
            elif r >= 3 and f >= 3 and m >= 3:
                segments.append('Loyal Customers')
            elif r >= 4 and f <= 2 and m <= 2:
                segments.append('New Customers')
            elif r <= 2 and f >= 3 and m >= 3:
                segments.append('At Risk')
            elif r <= 2 and f <= 2 and m >= 3:
                segments.append('Cannot Lose Them')
            elif r >= 3 and f <= 2 and m <= 2:
                segments.append('Potential Loyalists')
            elif r <= 2 and f <= 2 and m <= 2:
                segments.append('Hibernating')
            else:
                segments.append('Others')
                
        return pd.Series(segments)
    
    def _calculate_clv(self, rfm: pd.DataFrame) -> pd.Series:
        """Calculate Customer Lifetime Value"""
        # Simplified CLV calculation
        # CLV = (Average Order Value) × (Purchase Frequency) × (Customer Lifespan)
        avg_order_value = rfm['monetary'] / rfm['frequency']
        purchase_frequency = rfm['frequency'] / 365  # Assume 1 year period
        customer_lifespan = 365 / (rfm['recency'] + 1)  # Estimated lifespan
        
        clv = avg_order_value * purchase_frequency * customer_lifespan
        return clv
    
    def segment_customers_kmeans(self, df: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
        """Perform K-means clustering for customer segmentation"""
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        # Prepare features for clustering
        features = ['recency', 'frequency', 'monetary']
        X = df[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        df_clustered['cluster_name'] = df_clustered['cluster'].map(self._get_cluster_names())
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(df_clustered)
        
        return {
            'clustered_data': df_clustered,
            'cluster_analysis': cluster_analysis,
            'silhouette_score': silhouette_avg,
            'model': self.kmeans_model
        }
    
    def _get_cluster_names(self) -> Dict[int, str]:
        """Get meaningful names for clusters"""
        return {
            0: 'Low Value',
            1: 'Medium Value',
            2: 'High Value',
            3: 'VIP Customers',
            4: 'Potential Customers'
        }
    
    def _analyze_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        analysis = {}
        
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            
            analysis[f'cluster_{cluster}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'avg_recency': cluster_data['recency'].mean(),
                'avg_frequency': cluster_data['frequency'].mean(),
                'avg_monetary': cluster_data['monetary'].mean(),
                'total_value': cluster_data['monetary'].sum(),
                'characteristics': self._get_cluster_characteristics(cluster_data)
            }
        
        return analysis
    
    def _get_cluster_characteristics(self, cluster_data: pd.DataFrame) -> str:
        """Get textual description of cluster characteristics"""
        avg_recency = cluster_data['recency'].mean()
        avg_frequency = cluster_data['frequency'].mean()
        avg_monetary = cluster_data['monetary'].mean()
        
        if avg_recency < 30 and avg_frequency > 10 and avg_monetary > 1000:
            return "High-value, frequent, recent customers"
        elif avg_recency < 30 and avg_monetary > 500:
            return "Recent customers with good spend"
        elif avg_frequency > 5 and avg_monetary > 300:
            return "Loyal customers with regular purchases"
        elif avg_recency > 90:
            return "Inactive customers needing re-engagement"
        else:
            return "Standard customers"
    
    def identify_high_value_customers(self, df: pd.DataFrame, threshold_percentile: float = 0.9) -> pd.DataFrame:
        """Identify top 10% high-value customers"""
        threshold = df['monetary'].quantile(threshold_percentile)
        high_value_customers = df[df['monetary'] >= threshold].copy()
        
        # Add additional metrics
        high_value_customers['percentile_rank'] = df['monetary'].rank(pct=True)
        high_value_customers['value_category'] = 'High Value'
        
        return high_value_customers.sort_values('monetary', ascending=False)
    
    def analyze_customer_behavior(self, df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detailed customer behavior patterns"""
        
        # Merge RFM data with transaction details
        customer_behavior = transactions_df.groupby('customer_id').agg({
            'category': lambda x: x.mode().iloc[0],  # Most frequent category
            'payment_method': lambda x: x.mode().iloc[0],  # Preferred payment method
            'shopping_mall': lambda x: x.mode().iloc[0],  # Preferred mall
            'quantity': 'mean',  # Average quantity per transaction
            'price': 'mean',  # Average price per item
            'is_weekend': 'mean',  # Weekend shopping preference
            'month': lambda x: x.mode().iloc[0]  # Preferred shopping month
        }).reset_index()
        
        # Merge with RFM data
        enriched_data = df.merge(customer_behavior, on='customer_id', how='left')
        
        # Analyze patterns by segment
        segment_analysis = enriched_data.groupby('rfm_segment').agg({
            'monetary': ['mean', 'sum', 'count'],
            'frequency': 'mean',
            'recency': 'mean',
            'category': lambda x: x.mode().iloc[0],
            'payment_method': lambda x: x.mode().iloc[0],
            'is_weekend': 'mean'
        })
        
        return {
            'enriched_customer_data': enriched_data,
            'segment_patterns': segment_analysis,
            'behavior_insights': self._generate_behavior_insights(enriched_data)
        }
    
    def _generate_behavior_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable insights from customer behavior"""
        insights = []
        
        # Segment-specific insights
        for segment in df['rfm_segment'].unique():
            segment_data = df[df['rfm_segment'] == segment]
            
            if segment == 'Champions':
                insights.append(f"Champions prefer {segment_data['category'].mode().iloc[0]} and use {segment_data['payment_method'].mode().iloc[0]}")
            
            elif segment == 'At Risk':
                avg_recency = segment_data['recency'].mean()
                insights.append(f"At Risk customers haven't purchased in {avg_recency:.0f} days on average")
            
            elif segment == 'New Customers':
                insights.append(f"New customers prefer shopping at {segment_data['shopping_mall'].mode().iloc[0]}")
        
        # Payment method preferences
        payment_dist = df['payment_method'].value_counts(normalize=True)
        top_payment = payment_dist.index[0]
        insights.append(f"{top_payment} is the most popular payment method ({payment_dist.iloc[0]:.1%})")
        
        # Weekend shopping patterns
        weekend_shoppers = df['is_weekend'].mean()
        insights.append(f"{weekend_shoppers:.1%} of customers prefer weekend shopping")
        
        return insights
