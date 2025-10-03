"""
Customer service for handling customer-related business logic
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from ..core.database import Customer, CustomerSegment
from ..schemas.customers import CustomerSegmentResponse, CustomerInsightResponse
from ..analytics.customer_analytics import CustomerAnalytics
import logging

logger = logging.getLogger(__name__)

class CustomerService:
    """Service class for customer operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.analytics = CustomerAnalytics()
    
    async def get_customer_segments(self, segment: Optional[str] = None, limit: int = 100) -> List[CustomerSegmentResponse]:
        """Get customer segmentation data"""
        try:
            query = self.db.query(CustomerSegment)
            
            if segment:
                query = query.filter(CustomerSegment.segment == segment)
            
            segments = query.limit(limit).all()
            
            return [CustomerSegmentResponse.from_orm(seg) for seg in segments]
            
        except Exception as e:
            logger.error(f"Error getting customer segments: {e}")
            # Return mock data if database is empty
            return self._get_mock_segments(limit)
    
    async def get_high_value_customers(self, limit: int = 100, threshold: float = 1000.0) -> List[Dict[str, Any]]:
        """Get high-value customers (top 10% by purchase value)"""
        try:
            # Query customers with high monetary value
            high_value_customers = (
                self.db.query(CustomerSegment)
                .filter(CustomerSegment.monetary >= threshold)
                .order_by(CustomerSegment.monetary.desc())
                .limit(limit)
                .all()
            )
            
            result = []
            for customer in high_value_customers:
                result.append({
                    'customer_id': customer.customer_id,
                    'total_spent': customer.monetary,
                    'segment': customer.segment,
                    'rfm_score': customer.rfm_score,
                    'recency': customer.recency,
                    'frequency': customer.frequency
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting high-value customers: {e}")
            return self._get_mock_high_value_customers(limit, threshold)
    
    async def get_customer_insights(self, customer_id: str) -> CustomerInsightResponse:
        """Get detailed insights for a specific customer"""
        try:
            # Get customer segment data
            segment_data = (
                self.db.query(CustomerSegment)
                .filter(CustomerSegment.customer_id == customer_id)
                .first()
            )
            
            if not segment_data:
                raise ValueError(f"Customer {customer_id} not found")
            
            # Get customer transaction data
            customer_data = (
                self.db.query(Customer)
                .filter(Customer.customer_id == customer_id)
                .all()
            )
            
            # Calculate insights
            total_transactions = len(customer_data)
            avg_transaction_value = segment_data.monetary / segment_data.frequency if segment_data.frequency > 0 else 0
            
            # Get favorite categories (mock data for now)
            favorite_categories = ['Clothing', 'Shoes']
            preferred_payment_method = 'Credit Card'
            last_purchase_date = segment_data.created_at
            
            # Calculate CLV (simplified)
            customer_lifetime_value = segment_data.monetary * 1.5
            loyalty_score = min(100, (segment_data.frequency * 10) + ((365 - segment_data.recency) / 365 * 50))
            
            return CustomerInsightResponse(
                customer_id=customer_id,
                total_spent=segment_data.monetary,
                total_transactions=segment_data.frequency,
                avg_transaction_value=avg_transaction_value,
                favorite_categories=favorite_categories,
                preferred_payment_method=preferred_payment_method,
                last_purchase_date=last_purchase_date,
                customer_lifetime_value=customer_lifetime_value,
                segment=segment_data.segment,
                loyalty_score=loyalty_score
            )
            
        except Exception as e:
            logger.error(f"Error getting customer insights: {e}")
            return self._get_mock_customer_insights(customer_id)
    
    async def get_loyalty_analysis(self, segment: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get customer loyalty analysis"""
        try:
            query = self.db.query(CustomerSegment)
            
            if segment:
                query = query.filter(CustomerSegment.segment == segment)
            
            segments_data = query.all()
            
            # Group by segment and calculate metrics
            segment_analysis = {}
            for customer in segments_data:
                seg = customer.segment
                if seg not in segment_analysis:
                    segment_analysis[seg] = {
                        'customer_count': 0,
                        'total_recency': 0,
                        'total_frequency': 0,
                        'total_monetary': 0
                    }
                
                segment_analysis[seg]['customer_count'] += 1
                segment_analysis[seg]['total_recency'] += customer.recency
                segment_analysis[seg]['total_frequency'] += customer.frequency
                segment_analysis[seg]['total_monetary'] += customer.monetary
            
            # Calculate averages and additional metrics
            result = []
            for seg, data in segment_analysis.items():
                count = data['customer_count']
                avg_recency = data['total_recency'] / count
                avg_frequency = data['total_frequency'] / count
                avg_monetary = data['total_monetary'] / count
                
                # Calculate retention rate (simplified)
                retention_rate = max(0, min(100, (365 - avg_recency) / 365 * 100))
                
                # Determine churn risk
                churn_risk = 'High' if avg_recency > 90 else 'Medium' if avg_recency > 30 else 'Low'
                
                result.append({
                    'segment': seg,
                    'customer_count': count,
                    'avg_recency': avg_recency,
                    'avg_frequency': avg_frequency,
                    'avg_monetary': avg_monetary,
                    'retention_rate': retention_rate,
                    'churn_risk': churn_risk
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting loyalty analysis: {e}")
            return self._get_mock_loyalty_analysis()
    
    async def get_retention_analysis(self, period: str = "monthly") -> Dict[str, Any]:
        """Get customer retention analysis"""
        try:
            # This would typically involve cohort analysis
            # For now, providing mock retention data
            
            retention_data = {
                'period': period,
                'overall_retention_rate': 75.5,
                'new_customer_retention': 65.2,
                'returning_customer_retention': 85.8,
                'retention_by_segment': {
                    'Champions': 92.5,
                    'Loyal Customers': 88.3,
                    'Potential Loyalists': 72.1,
                    'At Risk': 45.2,
                    'New Customers': 68.7
                },
                'cohort_analysis': self._generate_cohort_data(),
                'insights': [
                    'Champions have the highest retention rate at 92.5%',
                    'At Risk customers have only 45.2% retention - immediate action needed',
                    'New customer retention is below average - improve onboarding'
                ]
            }
            
            return retention_data
            
        except Exception as e:
            logger.error(f"Error getting retention analysis: {e}")
            return {'error': str(e)}
    
    def _get_mock_segments(self, limit: int) -> List[CustomerSegmentResponse]:
        """Generate mock customer segments for demonstration"""
        mock_segments = []
        segments = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'New Customers']
        
        for i in range(min(limit, 50)):
            segment = segments[i % len(segments)]
            mock_segments.append(CustomerSegmentResponse(
                customer_id=f'CUST{i+1:04d}',
                segment=segment,
                rfm_score=f'{np.random.randint(1,6)}{np.random.randint(1,6)}{np.random.randint(1,6)}',
                recency=np.random.randint(1, 365),
                frequency=np.random.randint(1, 50),
                monetary=np.random.uniform(100, 5000),
                created_at=pd.Timestamp.now()
            ))
        
        return mock_segments
    
    def _get_mock_high_value_customers(self, limit: int, threshold: float) -> List[Dict[str, Any]]:
        """Generate mock high-value customers"""
        mock_customers = []
        segments = ['Champions', 'Loyal Customers', 'VIP']
        
        for i in range(min(limit, 20)):
            mock_customers.append({
                'customer_id': f'VIP{i+1:04d}',
                'total_spent': np.random.uniform(threshold, threshold * 5),
                'segment': segments[i % len(segments)],
                'rfm_score': f'{np.random.randint(4,6)}{np.random.randint(4,6)}{np.random.randint(4,6)}',
                'recency': np.random.randint(1, 30),
                'frequency': np.random.randint(10, 100)
            })
        
        return mock_customers
    
    def _get_mock_customer_insights(self, customer_id: str) -> CustomerInsightResponse:
        """Generate mock customer insights"""
        return CustomerInsightResponse(
            customer_id=customer_id,
            total_spent=2500.75,
            total_transactions=15,
            avg_transaction_value=166.72,
            favorite_categories=['Clothing', 'Shoes', 'Accessories'],
            preferred_payment_method='Credit Card',
            last_purchase_date=pd.Timestamp.now() - pd.Timedelta(days=5),
            customer_lifetime_value=3750.0,
            segment='Loyal Customers',
            loyalty_score=82.5
        )
    
    def _get_mock_loyalty_analysis(self) -> List[Dict[str, Any]]:
        """Generate mock loyalty analysis"""
        return [
            {
                'segment': 'Champions',
                'customer_count': 150,
                'avg_recency': 15.2,
                'avg_frequency': 25.8,
                'avg_monetary': 2500.50,
                'retention_rate': 92.5,
                'churn_risk': 'Low'
            },
            {
                'segment': 'Loyal Customers',
                'customer_count': 200,
                'avg_recency': 25.1,
                'avg_frequency': 18.3,
                'avg_monetary': 1800.25,
                'retention_rate': 85.2,
                'churn_risk': 'Low'
            },
            {
                'segment': 'At Risk',
                'customer_count': 100,
                'avg_recency': 120.5,
                'avg_frequency': 8.2,
                'avg_monetary': 950.75,
                'retention_rate': 45.8,
                'churn_risk': 'High'
            }
        ]
    
    def _generate_cohort_data(self) -> List[Dict[str, Any]]:
        """Generate mock cohort analysis data"""
        cohorts = []
        months = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
        
        for i, month in enumerate(months):
            retention_rates = [100]  # Month 0 is always 100%
            
            # Generate declining retention rates
            for j in range(1, 6):
                if i + j < len(months):
                    retention_rate = max(20, 100 - (j * 15) + np.random.randint(-5, 5))
                    retention_rates.append(retention_rate)
            
            cohorts.append({
                'cohort_month': month,
                'customer_count': np.random.randint(50, 200),
                'retention_rates': retention_rates
            })
        
        return cohorts
