"""
Analytics-related Pydantic schemas
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class RFMScores(BaseModel):
    recency: int
    frequency: int
    monetary: float
    rfm_score: str
    segment: str

class RFMAnalysisResponse(BaseModel):
    total_customers: int
    segment_distribution: Dict[str, int]
    avg_scores: RFMScores
    top_customers: List[Dict[str, Any]]
    
class ProfitabilityMetrics(BaseModel):
    gross_revenue: float
    total_discounts: float
    net_revenue: float
    profit_margin: float
    discount_impact: float

class ProfitabilityResponse(BaseModel):
    period: str
    overall_metrics: ProfitabilityMetrics
    by_category: Dict[str, ProfitabilityMetrics]
    by_store: Dict[str, ProfitabilityMetrics]
    recommendations: List[str]

class SeasonalTrend(BaseModel):
    period: str
    sales_volume: float
    revenue: float
    growth_rate: float
    seasonality_index: float

class SeasonalityResponse(BaseModel):
    granularity: str
    trends: List[SeasonalTrend]
    peak_seasons: List[str]
    low_seasons: List[str]
    forecast: List[Dict[str, Any]]

class PaymentMethodStats(BaseModel):
    method: str
    transaction_count: int
    total_value: float
    avg_transaction_value: float
    percentage_share: float

class CampaignSimulationResponse(BaseModel):
    campaign_type: str
    target_segment: str
    estimated_reach: int
    projected_revenue: float
    campaign_cost: float
    estimated_roi: float
    recommendations: List[str]
