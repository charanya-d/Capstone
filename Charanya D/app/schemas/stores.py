"""
Store-related Pydantic schemas
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class StorePerformanceResponse(BaseModel):
    store_id: str
    region: str
    total_sales: float
    total_transactions: int
    avg_transaction_value: float
    profit_margin: float
    period: str
    growth_rate: Optional[float] = None
    
    class Config:
        from_attributes = True

class StoreComparisonResponse(BaseModel):
    comparison_metric: str
    stores: List[Dict[str, Any]]
    best_performer: str
    worst_performer: str
    average_performance: float
    
class RegionalAnalysisResponse(BaseModel):
    region: str
    total_stores: int
    total_sales: float
    avg_sales_per_store: float
    top_performing_store: str
    performance_ranking: int
