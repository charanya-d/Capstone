"""
Analytics service for handling analytics-related business logic
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..core.database import Customer, CustomerSegment, StorePerformance
from ..schemas.analytics import RFMAnalysisResponse, ProfitabilityResponse, SeasonalityResponse
from ..analytics.customer_analytics import CustomerAnalytics
from ..analytics.store_analytics import StoreAnalytics
import logging

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service class for analytics operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.customer_analytics = CustomerAnalytics()
        self.store_analytics = StoreAnalytics()
    
    async def get_rfm_analysis(self, customer_id: Optional[str] = None, segment: Optional[str] = None) -> RFMAnalysisResponse:
        """Get RFM (Recency, Frequency, Monetary) analysis"""
        try:
            query = self.db.query(CustomerSegment)
            
            if customer_id:
                query = query.filter(CustomerSegment.customer_id == customer_id)
            elif segment:
                query = query.filter(CustomerSegment.segment == segment)
            
            rfm_data = query.all()
            
            if not rfm_data:
                return self._get_mock_rfm_analysis()
            
            # Calculate segment distribution
            segment_distribution = {}
            total_customers = len(rfm_data)
            
            for customer in rfm_data:
                seg = customer.segment
                segment_distribution[seg] = segment_distribution.get(seg, 0) + 1
            
            # Calculate average scores
            avg_recency = np.mean([c.recency for c in rfm_data])
            avg_frequency = np.mean([c.frequency for c in rfm_data])
            avg_monetary = np.mean([c.monetary for c in rfm_data])
            
            # Get top customers
            top_customers = sorted(rfm_data, key=lambda x: x.monetary, reverse=True)[:10]
            top_customers_data = [
                {
                    'customer_id': c.customer_id,
                    'segment': c.segment,
                    'rfm_score': c.rfm_score,
                    'monetary': c.monetary,
                    'frequency': c.frequency,
                    'recency': c.recency
                }
                for c in top_customers
            ]
            
            return RFMAnalysisResponse(
                total_customers=total_customers,
                segment_distribution=segment_distribution,
                avg_scores={
                    'recency': int(avg_recency),
                    'frequency': int(avg_frequency),
                    'monetary': float(avg_monetary),
                    'rfm_score': '333',  # Representative score
                    'segment': 'Mixed'
                },
                top_customers=top_customers_data
            )
            
        except Exception as e:
            logger.error(f"Error getting RFM analysis: {e}")
            return self._get_mock_rfm_analysis()
    
    async def get_profitability_analysis(self, store_id: Optional[str] = None, category: Optional[str] = None, period: str = "monthly") -> ProfitabilityResponse:
        """Get profitability analysis with discount impact"""
        try:
            # Mock profitability analysis (in real implementation, this would query actual transaction data)
            overall_metrics = {
                'gross_revenue': 150000.0,
                'total_discounts': 7500.0,
                'net_revenue': 142500.0,
                'profit_margin': 35.2,
                'discount_impact': 5.0
            }
            
            by_category = {
                'Clothing': {
                    'gross_revenue': 60000.0,
                    'total_discounts': 3000.0,
                    'net_revenue': 57000.0,
                    'profit_margin': 38.5,
                    'discount_impact': 5.0
                },
                'Electronics': {
                    'gross_revenue': 45000.0,
                    'total_discounts': 2250.0,
                    'net_revenue': 42750.0,
                    'profit_margin': 32.1,
                    'discount_impact': 5.0
                },
                'Books': {
                    'gross_revenue': 25000.0,
                    'total_discounts': 1250.0,
                    'net_revenue': 23750.0,
                    'profit_margin': 28.9,
                    'discount_impact': 5.0
                }
            }
            
            by_store = {
                'Mall_A': {
                    'gross_revenue': 50000.0,
                    'total_discounts': 2500.0,
                    'net_revenue': 47500.0,
                    'profit_margin': 36.8,
                    'discount_impact': 5.0
                },
                'Mall_B': {
                    'gross_revenue': 40000.0,
                    'total_discounts': 2000.0,
                    'net_revenue': 38000.0,
                    'profit_margin': 34.2,
                    'discount_impact': 5.0
                },
                'Mall_C': {
                    'gross_revenue': 35000.0,
                    'total_discounts': 1750.0,
                    'net_revenue': 33250.0,
                    'profit_margin': 33.1,
                    'discount_impact': 5.0
                }
            }
            
            recommendations = [
                "Clothing category shows highest profit margin - increase inventory",
                "Electronics category has good volume but lower margins - optimize pricing",
                "Mall_A outperforms others - replicate successful strategies",
                "Consider reducing discount rates in high-margin categories"
            ]
            
            return ProfitabilityResponse(
                period=period,
                overall_metrics=overall_metrics,
                by_category=by_category,
                by_store=by_store,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error getting profitability analysis: {e}")
            return self._get_mock_profitability_response(period)
    
    async def get_seasonality_analysis(self, granularity: str = "monthly", years: int = 2) -> SeasonalityResponse:
        """Get seasonal sales trends analysis"""
        try:
            # Generate mock seasonal data
            trends = []
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Simulate seasonal patterns
            base_sales = 100000
            seasonal_multipliers = [0.8, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.0, 0.9, 1.4, 1.5]  # Holiday boost in Nov/Dec
            
            for i, month in enumerate(months):
                sales_volume = base_sales * seasonal_multipliers[i] + np.random.normal(0, 5000)
                revenue = sales_volume * np.random.uniform(45, 55)  # Average price variation
                
                # Calculate growth rate (mock)
                growth_rate = np.random.uniform(-10, 15) if i > 0 else 0
                seasonality_index = seasonal_multipliers[i] * 100
                
                trends.append({
                    'period': month,
                    'sales_volume': sales_volume,
                    'revenue': revenue,
                    'growth_rate': growth_rate,
                    'seasonality_index': seasonality_index
                })
            
            # Identify peak and low seasons
            peak_seasons = ['Nov', 'Dec', 'Jul']  # Holiday and summer peaks
            low_seasons = ['Feb', 'Jan', 'Mar']   # Post-holiday lows
            
            # Generate forecast (simplified)
            forecast = []
            for i in range(6):  # Next 6 months
                future_month = months[(datetime.now().month + i) % 12]
                multiplier = seasonal_multipliers[(datetime.now().month + i) % 12]
                forecast.append({
                    'period': future_month,
                    'predicted_revenue': base_sales * multiplier * 50,
                    'confidence_interval': [base_sales * multiplier * 45, base_sales * multiplier * 55]
                })
            
            return SeasonalityResponse(
                granularity=granularity,
                trends=trends,
                peak_seasons=peak_seasons,
                low_seasons=low_seasons,
                forecast=forecast
            )
            
        except Exception as e:
            logger.error(f"Error getting seasonality analysis: {e}")
            return self._get_mock_seasonality_response(granularity)
    
    async def get_payment_method_analysis(self, region: Optional[str] = None, period: str = "all") -> Dict[str, Any]:
        """Get payment method preference analysis"""
        try:
            # Mock payment method data
            payment_methods = [
                {
                    'method': 'Credit Card',
                    'transaction_count': 1250,
                    'total_value': 75000.0,
                    'avg_transaction_value': 60.0,
                    'percentage_share': 45.2
                },
                {
                    'method': 'Debit Card',
                    'transaction_count': 800,
                    'total_value': 40000.0,
                    'avg_transaction_value': 50.0,
                    'percentage_share': 24.1
                },
                {
                    'method': 'Cash',
                    'transaction_count': 650,
                    'total_value': 32500.0,
                    'avg_transaction_value': 50.0,
                    'percentage_share': 19.6
                },
                {
                    'method': 'Digital Wallet',
                    'transaction_count': 300,
                    'total_value': 18000.0,
                    'avg_transaction_value': 60.0,
                    'percentage_share': 10.8
                }
            ]
            
            # Regional preferences (mock)
            regional_preferences = {
                'Mall_A': {'Credit Card': 50, 'Debit Card': 25, 'Cash': 15, 'Digital Wallet': 10},
                'Mall_B': {'Credit Card': 40, 'Debit Card': 30, 'Cash': 20, 'Digital Wallet': 10},
                'Mall_C': {'Credit Card': 45, 'Debit Card': 20, 'Cash': 25, 'Digital Wallet': 10}
            }
            
            # Age group preferences (mock)
            age_preferences = {
                '18-25': {'Credit Card': 35, 'Debit Card': 25, 'Cash': 10, 'Digital Wallet': 30},
                '26-35': {'Credit Card': 45, 'Debit Card': 25, 'Cash': 15, 'Digital Wallet': 15},
                '36-50': {'Credit Card': 50, 'Debit Card': 30, 'Cash': 15, 'Digital Wallet': 5},
                '50+': {'Credit Card': 40, 'Debit Card': 35, 'Cash': 20, 'Digital Wallet': 5}
            }
            
            insights = [
                "Credit Card is the dominant payment method with 45.2% share",
                "Digital Wallet usage is highest among 18-25 age group (30%)",
                "Cash usage decreases with age but remains significant in all groups",
                "Mall_A shows highest digital payment adoption"
            ]
            
            return {
                'payment_methods': payment_methods,
                'regional_preferences': regional_preferences,
                'age_preferences': age_preferences,
                'trends': {
                    'digital_growth': 25.5,  # YoY growth %
                    'cash_decline': -8.2,    # YoY decline %
                    'card_stable': 2.1       # YoY growth %
                },
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error getting payment method analysis: {e}")
            return {'error': str(e)}
    
    async def get_category_insights(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get category-wise insights and customer segments"""
        try:
            categories = ['Clothing', 'Electronics', 'Books', 'Home & Garden', 'Sports', 'Beauty']
            
            category_data = []
            for cat in categories:
                if category and cat != category:
                    continue
                    
                data = {
                    'category': cat,
                    'total_revenue': np.random.uniform(50000, 200000),
                    'total_transactions': np.random.randint(500, 2000),
                    'avg_transaction_value': np.random.uniform(30, 150),
                    'profit_margin': np.random.uniform(25, 45),
                    'top_customer_segment': np.random.choice(['Champions', 'Loyal Customers', 'High Value']),
                    'growth_rate': np.random.uniform(-5, 20),
                    'seasonal_index': np.random.uniform(80, 120)
                }
                category_data.append(data)
            
            # Cross-category analysis
            cross_category_insights = [
                "Clothing customers often purchase Beauty products (35% overlap)",
                "Electronics buyers have highest average transaction value",
                "Books category shows strong loyalty but lower margins",
                "Home & Garden peaks during spring season"
            ]
            
            return {
                'category_performance': category_data,
                'cross_category_analysis': cross_category_insights,
                'recommendations': [
                    "Bundle complementary categories for increased basket size",
                    "Focus marketing on high-margin categories",
                    "Develop seasonal campaigns for peak categories"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting category insights: {e}")
            return {'error': str(e)}
    
    async def simulate_campaign(self, discount_percentage: float, target_segment: str, campaign_budget: float) -> Dict[str, Any]:
        """Simulate marketing campaign targeting specific customer segments"""
        try:
            # Get segment data
            if target_segment == "high_value":
                estimated_reach = 500
                avg_customer_value = 1500
                response_rate = 0.15
            elif target_segment == "loyal_customers":
                estimated_reach = 800
                avg_customer_value = 800
                response_rate = 0.20
            elif target_segment == "at_risk":
                estimated_reach = 300
                avg_customer_value = 600
                response_rate = 0.10
            else:
                estimated_reach = 1000
                avg_customer_value = 500
                response_rate = 0.12
            
            # Calculate campaign metrics
            expected_responders = int(estimated_reach * response_rate)
            
            # Calculate revenue impact
            baseline_revenue = expected_responders * avg_customer_value
            discount_amount = baseline_revenue * (discount_percentage / 100)
            projected_revenue = baseline_revenue - discount_amount
            
            # Campaign costs (mock)
            campaign_cost = campaign_budget
            
            # ROI calculation
            net_profit = projected_revenue - campaign_cost
            roi = (net_profit / campaign_cost) * 100 if campaign_cost > 0 else 0
            
            recommendations = []
            if roi > 200:
                recommendations.append("Excellent ROI - proceed with campaign")
            elif roi > 100:
                recommendations.append("Good ROI - consider scaling up")
            elif roi > 0:
                recommendations.append("Positive ROI but room for optimization")
            else:
                recommendations.append("Negative ROI - reconsider campaign parameters")
            
            if discount_percentage > 20:
                recommendations.append("High discount rate - consider reducing to preserve margins")
            
            return {
                'campaign_type': f"{discount_percentage}% discount campaign",
                'target_segment': target_segment,
                'estimated_reach': estimated_reach,
                'expected_responders': expected_responders,
                'projected_revenue': projected_revenue,
                'campaign_cost': campaign_cost,
                'estimated_roi': roi,
                'break_even_point': campaign_cost / (avg_customer_value * (1 - discount_percentage/100)),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error simulating campaign: {e}")
            return {'error': str(e)}
    
    async def get_cohort_analysis(self, period: str = "monthly") -> Dict[str, Any]:
        """Get customer cohort analysis"""
        try:
            # Mock cohort analysis data
            cohorts = []
            base_date = datetime(2023, 1, 1)
            
            for i in range(12):  # 12 months of cohorts
                cohort_date = base_date + timedelta(days=30*i)
                cohort_size = np.random.randint(80, 200)
                
                # Generate retention rates for subsequent months
                retention_rates = [100]  # Month 0 is always 100%
                for month in range(1, min(12-i, 12)):
                    # Simulate declining retention with some recovery
                    base_retention = max(20, 100 - (month * 12))
                    retention = base_retention + np.random.randint(-10, 10)
                    retention_rates.append(max(0, min(100, retention)))
                
                cohorts.append({
                    'cohort_period': cohort_date.strftime('%Y-%m'),
                    'cohort_size': cohort_size,
                    'retention_rates': retention_rates
                })
            
            # Calculate average retention rates
            avg_retention = {
                'month_1': np.mean([c['retention_rates'][1] if len(c['retention_rates']) > 1 else 0 for c in cohorts]),
                'month_3': np.mean([c['retention_rates'][3] if len(c['retention_rates']) > 3 else 0 for c in cohorts]),
                'month_6': np.mean([c['retention_rates'][6] if len(c['retention_rates']) > 6 else 0 for c in cohorts]),
                'month_12': np.mean([c['retention_rates'][11] if len(c['retention_rates']) > 11 else 0 for c in cohorts])
            }
            
            insights = [
                f"Average 1-month retention: {avg_retention['month_1']:.1f}%",
                f"Average 6-month retention: {avg_retention['month_6']:.1f}%",
                "Retention rates show seasonal patterns with Q4 cohorts performing better",
                "Focus on improving early-stage retention (first 3 months)"
            ]
            
            return {
                'period': period,
                'cohorts': cohorts,
                'average_retention': avg_retention,
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error getting cohort analysis: {e}")
            return {'error': str(e)}
    
    def _get_mock_rfm_analysis(self) -> RFMAnalysisResponse:
        """Generate mock RFM analysis"""
        return RFMAnalysisResponse(
            total_customers=1000,
            segment_distribution={
                'Champions': 150,
                'Loyal Customers': 200,
                'Potential Loyalists': 250,
                'At Risk': 180,
                'New Customers': 220
            },
            avg_scores={
                'recency': 45,
                'frequency': 12,
                'monetary': 850.0,
                'rfm_score': '333',
                'segment': 'Mixed Analysis'
            },
            top_customers=[
                {
                    'customer_id': 'CUST0001',
                    'segment': 'Champions',
                    'rfm_score': '555',
                    'monetary': 5000.0,
                    'frequency': 45,
                    'recency': 5
                }
            ]
        )
    
    def _get_mock_profitability_response(self, period: str) -> ProfitabilityResponse:
        """Generate mock profitability response"""
        return ProfitabilityResponse(
            period=period,
            overall_metrics={
                'gross_revenue': 100000.0,
                'total_discounts': 5000.0,
                'net_revenue': 95000.0,
                'profit_margin': 32.5,
                'discount_impact': 5.0
            },
            by_category={},
            by_store={},
            recommendations=["Optimize discount strategy", "Focus on high-margin products"]
        )
    
    def _get_mock_seasonality_response(self, granularity: str) -> SeasonalityResponse:
        """Generate mock seasonality response"""
        return SeasonalityResponse(
            granularity=granularity,
            trends=[],
            peak_seasons=['Nov', 'Dec'],
            low_seasons=['Jan', 'Feb'],
            forecast=[]
        )
