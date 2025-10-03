"""
Store performance API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd

from ..core.database import get_db, StorePerformance
from ..schemas.stores import StorePerformanceResponse, StoreComparisonResponse
from ..services.store_service import StoreService

router = APIRouter()

@router.get("/performance", response_model=List[StorePerformanceResponse])
async def get_store_performance(
    store_id: Optional[str] = None,
    region: Optional[str] = None,
    period: Optional[str] = "monthly",
    db: Session = Depends(get_db)
):
    """Get store performance metrics"""
    try:
        store_service = StoreService(db)
        return await store_service.get_performance_metrics(store_id, region, period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/comparison", response_model=StoreComparisonResponse)
async def compare_stores(
    store_ids: List[str],
    metric: str = "revenue",
    db: Session = Depends(get_db)
):
    """Compare performance between stores"""
    try:
        store_service = StoreService(db)
        return await store_service.compare_stores(store_ids, metric)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top-performers")
async def get_top_performing_stores(
    limit: int = 10,
    metric: str = "revenue",
    period: str = "monthly",
    db: Session = Depends(get_db)
):
    """Get top performing stores"""
    try:
        store_service = StoreService(db)
        return await store_service.get_top_performers(limit, metric, period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/regional-analysis")
async def get_regional_analysis(
    region: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get regional performance analysis"""
    try:
        store_service = StoreService(db)
        return await store_service.get_regional_analysis(region)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
