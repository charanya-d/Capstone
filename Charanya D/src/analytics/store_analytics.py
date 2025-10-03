"""
Store performance and profitability analysis module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StoreAnalytics:
    """Store performance and profitability analysis engine"""
    
    def __init__(self):
        self.seasonal_patterns = {}
        
    def analyze_store_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive store performance analysis"""
        logger.info("Analyzing store performance...")
        
        # Store-level metrics
        store_metrics = df.groupby('shopping_mall').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'customer_id': 'nunique',
            'invoice_no': 'nunique'
        }).round(2)
        
        # Flatten column names
        store_metrics.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in store_metrics.columns]
        
        # Rename for clarity
        store_metrics = store_metrics.rename(columns={
            'sum_total_amount': 'total_revenue',
            'mean_total_amount': 'avg_transaction_value',
            'count_total_amount': 'total_transactions',
            'sum_quantity': 'total_items_sold',
            'nunique_customer_id': 'unique_customers',
            'nunique_invoice_no': 'unique_invoices'
        })
        
        # Calculate additional metrics
        store_metrics['revenue_per_customer'] = store_metrics['total_revenue'] / store_metrics['unique_customers']
        store_metrics['items_per_transaction'] = store_metrics['total_items_sold'] / store_metrics['total_transactions']
        
        # Performance ranking
        store_metrics['revenue_rank'] = store_metrics['total_revenue'].rank(ascending=False)
        store_metrics['customer_rank'] = store_metrics['unique_customers'].rank(ascending=False)
        
        return {
            'store_metrics': store_metrics.reset_index(),
            'summary': self._generate_store_summary(store_metrics),
            'recommendations': self._generate_store_recommendations(store_metrics)
        }
    
    def analyze_profitability(self, df: pd.DataFrame, discount_rate: float = 0.05) -> Dict[str, Any]:
        """Analyze profitability with discount impact"""
        logger.info("Analyzing profitability...")
        
        # Calculate base profitability metrics
        df_profit = df.copy()
        
        # Simulate cost structure (assuming 60% gross margin before discounts)
        df_profit['cost'] = df_profit['price'] * 0.4
        df_profit['gross_profit'] = df_profit['price'] - df_profit['cost']
        
        # Apply discount impact
        df_profit['discount_amount'] = df_profit['price'] * discount_rate
        df_profit['discounted_price'] = df_profit['price'] - df_profit['discount_amount']
        df_profit['net_profit'] = df_profit['discounted_price'] - df_profit['cost']
        df_profit['profit_margin'] = (df_profit['net_profit'] / df_profit['discounted_price']) * 100
        
        # Profitability by category
        category_profit = df_profit.groupby('category').agg({
            'price': 'sum',
            'discount_amount': 'sum',
            'net_profit': 'sum',
            'profit_margin': 'mean',
            'quantity': 'sum'
        }).round(2)
        
        category_profit['total_revenue'] = category_profit['price']
        category_profit['discount_impact'] = (category_profit['discount_amount'] / category_profit['price']) * 100
        
        # Profitability by store
        store_profit = df_profit.groupby('shopping_mall').agg({
            'price': 'sum',
            'discount_amount': 'sum',
            'net_profit': 'sum',
            'profit_margin': 'mean',
            'quantity': 'sum'
        }).round(2)
        
        store_profit['total_revenue'] = store_profit['price']
        store_profit['discount_impact'] = (store_profit['discount_amount'] / store_profit['price']) * 100
        
        return {
            'overall_profitability': {
                'total_revenue': df_profit['price'].sum(),
                'total_discounts': df_profit['discount_amount'].sum(),
                'total_profit': df_profit['net_profit'].sum(),
                'avg_margin': df_profit['profit_margin'].mean(),
                'discount_impact_percent': (df_profit['discount_amount'].sum() / df_profit['price'].sum()) * 100
            },
            'category_profitability': category_profit.reset_index(),
            'store_profitability': store_profit.reset_index(),
            'insights': self._generate_profitability_insights(category_profit, store_profit)
        }
    
    def analyze_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal sales trends"""
        logger.info("Analyzing seasonal trends...")
        
        # Monthly trends
        monthly_sales = df.groupby(['year', 'month']).agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        monthly_sales['period'] = monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str).str.zfill(2)
        
        # Calculate month-over-month growth
        monthly_sales['revenue_growth'] = monthly_sales['total_amount'].pct_change() * 100
        
        # Quarterly trends
        quarterly_sales = df.groupby(['year', 'quarter']).agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        quarterly_sales['period'] = quarterly_sales['year'].astype(str) + '-Q' + quarterly_sales['quarter'].astype(str)
        quarterly_sales['revenue_growth'] = quarterly_sales['total_amount'].pct_change() * 100
        
        # Seasonal indices
        monthly_avg = monthly_sales.groupby('month')['total_amount'].mean()
        overall_avg = monthly_avg.mean()
        seasonal_index = (monthly_avg / overall_avg) * 100
        
        # Day of week patterns
        dow_sales = df.groupby('day_of_week').agg({
            'total_amount': 'sum',
            'quantity': 'sum'
        })
        
        dow_sales.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'monthly_trends': monthly_sales,
            'quarterly_trends': quarterly_sales,
            'seasonal_indices': seasonal_index.to_dict(),
            'day_of_week_patterns': dow_sales.to_dict(),
            'peak_months': seasonal_index.nlargest(3).to_dict(),
            'low_months': seasonal_index.nsmallest(3).to_dict(),
            'insights': self._generate_seasonality_insights(seasonal_index, dow_sales)
        }
    
    def analyze_payment_methods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze payment method preferences and patterns"""
        logger.info("Analyzing payment methods...")
        
        # Overall payment method distribution
        payment_dist = df.groupby('payment_method').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        
        payment_dist.columns = ['total_value', 'avg_transaction_value', 'transaction_count', 'unique_customers']
        payment_dist['percentage_share'] = (payment_dist['total_value'] / payment_dist['total_value'].sum()) * 100
        
        # Payment methods by demographics
        payment_by_gender = pd.crosstab(df['payment_method'], df['gender'], normalize='columns') * 100
        payment_by_age = df.groupby(['payment_method', pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])])['total_amount'].sum().unstack(fill_value=0)
        
        # Payment methods by store
        payment_by_store = pd.crosstab(df['shopping_mall'], df['payment_method'], normalize='index') * 100
        
        return {
            'payment_distribution': payment_dist.reset_index(),
            'payment_by_gender': payment_by_gender,
            'payment_by_age': payment_by_age,
            'payment_by_store': payment_by_store,
            'insights': self._generate_payment_insights(payment_dist, payment_by_gender)
        }
    
    def perform_regional_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform regional performance comparison"""
        logger.info("Performing regional analysis...")
        
        # Assuming shopping_mall represents different regions
        regional_metrics = df.groupby('shopping_mall').agg({
            'total_amount': ['sum', 'mean'],
            'customer_id': 'nunique',
            'quantity': 'sum',
            'invoice_no': 'nunique'
        })
        
        regional_metrics.columns = ['total_revenue', 'avg_transaction_value', 'unique_customers', 'total_items', 'total_transactions']
        
        # Calculate performance indicators
        regional_metrics['revenue_per_customer'] = regional_metrics['total_revenue'] / regional_metrics['unique_customers']
        regional_metrics['items_per_customer'] = regional_metrics['total_items'] / regional_metrics['unique_customers']
        
        # Market share
        regional_metrics['market_share'] = (regional_metrics['total_revenue'] / regional_metrics['total_revenue'].sum()) * 100
        
        # Performance ranking
        regional_metrics['performance_rank'] = regional_metrics['total_revenue'].rank(ascending=False)
        
        return {
            'regional_metrics': regional_metrics.reset_index(),
            'best_performer': regional_metrics['total_revenue'].idxmax(),
            'market_leader_share': regional_metrics['market_share'].max(),
            'insights': self._generate_regional_insights(regional_metrics)
        }
    
    def _generate_store_summary(self, store_metrics: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for stores"""
        return {
            'total_stores': len(store_metrics),
            'total_revenue': store_metrics['total_revenue'].sum(),
            'avg_revenue_per_store': store_metrics['total_revenue'].mean(),
            'best_performing_store': store_metrics['total_revenue'].idxmax(),
            'revenue_concentration': (store_metrics['total_revenue'].max() / store_metrics['total_revenue'].sum()) * 100
        }
    
    def _generate_store_recommendations(self, store_metrics: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations for store performance"""
        recommendations = []
        
        # Performance gap analysis
        top_performer = store_metrics['total_revenue'].max()
        bottom_performer = store_metrics['total_revenue'].min()
        gap = ((top_performer - bottom_performer) / bottom_performer) * 100
        
        if gap > 50:
            recommendations.append(f"Significant performance gap ({gap:.1f}%) between stores - investigate best practices from top performer")
        
        # Customer engagement
        low_engagement_stores = store_metrics[store_metrics['revenue_per_customer'] < store_metrics['revenue_per_customer'].median()]
        if len(low_engagement_stores) > 0:
            recommendations.append(f"{len(low_engagement_stores)} stores have below-median customer engagement - focus on customer retention programs")
        
        # Transaction value optimization
        low_avg_transaction = store_metrics[store_metrics['avg_transaction_value'] < store_metrics['avg_transaction_value'].median()]
        if len(low_avg_transaction) > 0:
            recommendations.append(f"{len(low_avg_transaction)} stores have low average transaction values - implement upselling strategies")
        
        return recommendations
    
    def _generate_profitability_insights(self, category_profit: pd.DataFrame, store_profit: pd.DataFrame) -> List[str]:
        """Generate profitability insights"""
        insights = []
        
        # Most profitable category
        top_profit_category = category_profit['net_profit'].idxmax()
        top_margin_category = category_profit['profit_margin'].idxmax()
        
        insights.append(f"{top_profit_category} generates highest total profit")
        insights.append(f"{top_margin_category} has the highest profit margin")
        
        # Store profitability
        most_profitable_store = store_profit['net_profit'].idxmax()
        insights.append(f"{most_profitable_store} is the most profitable store")
        
        # Discount impact
        high_discount_impact = category_profit[category_profit['discount_impact'] > category_profit['discount_impact'].median()]
        if len(high_discount_impact) > 0:
            insights.append(f"Categories with high discount sensitivity: {', '.join(high_discount_impact.index.tolist())}")
        
        return insights
    
    def _generate_seasonality_insights(self, seasonal_index: pd.Series, dow_sales: pd.DataFrame) -> List[str]:
        """Generate seasonality insights"""
        insights = []
        
        # Peak season
        peak_month = seasonal_index.idxmax()
        peak_value = seasonal_index.max()
        insights.append(f"Month {peak_month} is the peak season with {peak_value:.1f}% above average sales")
        
        # Low season
        low_month = seasonal_index.idxmin()
        low_value = seasonal_index.min()
        insights.append(f"Month {low_month} is the lowest season with {low_value:.1f}% below average sales")
        
        # Weekend vs weekday
        weekend_sales = dow_sales.loc[['Saturday', 'Sunday'], 'total_amount'].sum()
        weekday_sales = dow_sales.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 'total_amount'].sum()
        
        if weekend_sales > weekday_sales:
            insights.append("Weekend sales outperform weekday sales - focus weekend promotions")
        else:
            insights.append("Weekday sales stronger than weekends - consider weekday loyalty programs")
        
        return insights
    
    def _generate_payment_insights(self, payment_dist: pd.DataFrame, payment_by_gender: pd.DataFrame) -> List[str]:
        """Generate payment method insights"""
        insights = []
        
        # Most popular payment method
        top_payment = payment_dist['percentage_share'].idxmax()
        top_share = payment_dist.loc[top_payment, 'percentage_share']
        insights.append(f"{top_payment} is the dominant payment method with {top_share:.1f}% market share")
        
        # Highest value transactions
        highest_avg = payment_dist['avg_transaction_value'].idxmax()
        insights.append(f"{highest_avg} users have the highest average transaction value")
        
        # Gender preferences
        for payment_method in payment_by_gender.index:
            male_pref = payment_by_gender.loc[payment_method, 'Male']
            female_pref = payment_by_gender.loc[payment_method, 'Female']
            
            if abs(male_pref - female_pref) > 10:
                if male_pref > female_pref:
                    insights.append(f"{payment_method} is preferred more by males ({male_pref:.1f}% vs {female_pref:.1f}%)")
                else:
                    insights.append(f"{payment_method} is preferred more by females ({female_pref:.1f}% vs {male_pref:.1f}%)")
        
        return insights
    
    def _generate_regional_insights(self, regional_metrics: pd.DataFrame) -> List[str]:
        """Generate regional analysis insights"""
        insights = []
        
        # Market concentration
        top_region_share = regional_metrics['market_share'].max()
        if top_region_share > 40:
            top_region = regional_metrics['market_share'].idxmax()
            insights.append(f"{top_region} dominates with {top_region_share:.1f}% market share")
        
        # Performance gaps
        revenue_std = regional_metrics['total_revenue'].std()
        revenue_mean = regional_metrics['total_revenue'].mean()
        cv = (revenue_std / revenue_mean) * 100
        
        if cv > 30:
            insights.append(f"High regional performance variation (CV: {cv:.1f}%) - standardize operations")
        
        # Customer efficiency
        best_efficiency = regional_metrics['revenue_per_customer'].idxmax()
        best_efficiency_value = regional_metrics.loc[best_efficiency, 'revenue_per_customer']
        insights.append(f"{best_efficiency} has the highest revenue per customer ({best_efficiency_value:.2f})")
        
        return insights
