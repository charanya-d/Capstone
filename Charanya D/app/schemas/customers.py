"""
Customer-related Pydantic schemas
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class CustomerSegmentResponse(BaseModel):
    customer_id: str
    segment: str
    rfm_score: str
    recency: int
    frequency: int
    monetary: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class CustomerInsightResponse(BaseModel):
    customer_id: str
    total_spent: float
    total_transactions: int
    avg_transaction_value: float
    favorite_categories: List[str]
    preferred_payment_method: str
    last_purchase_date: datetime
    customer_lifetime_value: float
    segment: str
    loyalty_score: float

class HighValueCustomerResponse(BaseModel):
    customer_id: str
    total_spent: float
    percentile_rank: float
    segment: str
    lifetime_value: float

class LoyaltyAnalysisResponse(BaseModel):
    segment: str
    customer_count: int
    avg_recency: float
    avg_frequency: float
    avg_monetary: float
    retention_rate: float
    churn_risk: str
