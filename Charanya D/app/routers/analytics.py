"""
Analytics API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..core.database import get_db
from ..schemas.analytics import RFMAnalysisResponse, ProfitabilityResponse, SeasonalityResponse
from ..services.analytics_service import AnalyticsService

router = APIRouter()

@router.get("/rfm", response_model=RFMAnalysisResponse)
async def get_rfm_analysis(
    customer_id: Optional[str] = None,
    segment: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get RFM (Recency, Frequency, Monetary) analysis"""
    try:
        analytics_service = AnalyticsService(db)
        return await analytics_service.get_rfm_analysis(customer_id, segment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profitability", response_model=ProfitabilityResponse)
async def get_profitability_analysis(
    store_id: Optional[str] = None,
    category: Optional[str] = None,
    period: str = "monthly",
    db: Session = Depends(get_db)
):
    """Get profitability analysis with discount impact"""
    try:
        analytics_service = AnalyticsService(db)
        return await analytics_service.get_profitability_analysis(store_id, category, period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/seasonality", response_model=SeasonalityResponse)
async def get_seasonality_analysis(
    granularity: str = "monthly",
    years: int = 2,
    db: Session = Depends(get_db)
):
    """Get seasonal sales trends analysis"""
    try:
        analytics_service = AnalyticsService(db)
        return await analytics_service.get_seasonality_analysis(granularity, years)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/payment-methods")
async def get_payment_method_analysis(
    region: Optional[str] = None,
    period: str = "all",
    db: Session = Depends(get_db)
):
    """Get payment method preference analysis"""
    try:
        analytics_service = AnalyticsService(db)
        return await analytics_service.get_payment_method_analysis(region, period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/category-insights")
async def get_category_insights(
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get category-wise insights and customer segments"""
    try:
        analytics_service = AnalyticsService(db)
        return await analytics_service.get_category_insights(category)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/campaign-simulation")
async def simulate_campaign(
    discount_percentage: float = 10.0,
    target_segment: str = "high_value",
    campaign_budget: float = 10000.0,
    db: Session = Depends(get_db)
):
    """Simulate marketing campaign targeting high-value customers"""
    try:
        analytics_service = AnalyticsService(db)
        return await analytics_service.simulate_campaign(
            discount_percentage, 
            target_segment, 
            campaign_budget
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cohort-analysis")
async def get_cohort_analysis(
    period: str = "monthly",
    db: Session = Depends(get_db)
):
    """Get customer cohort analysis"""
    try:
        analytics_service = AnalyticsService(db)
        return await analytics_service.get_cohort_analysis(period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
