"""
Customer segmentation API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from ..core.database import get_db
from ..schemas.customers import CustomerSegmentResponse, CustomerInsightResponse
from ..services.customer_service import CustomerService

router = APIRouter()

@router.get("/segments", response_model=List[CustomerSegmentResponse])
async def get_customer_segments(
    segment: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get customer segmentation data"""
    try:
        customer_service = CustomerService(db)
        return await customer_service.get_customer_segments(segment, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/high-value")
async def get_high_value_customers(
    limit: int = 100,
    threshold: float = 1000.0,
    db: Session = Depends(get_db)
):
    """Get high-value customers (top 10% by purchase value)"""
    try:
        customer_service = CustomerService(db)
        return await customer_service.get_high_value_customers(limit, threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/{customer_id}", response_model=CustomerInsightResponse)
async def get_customer_insights(
    customer_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed insights for a specific customer"""
    try:
        customer_service = CustomerService(db)
        return await customer_service.get_customer_insights(customer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loyalty-analysis")
async def get_loyalty_analysis(
    segment: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get customer loyalty analysis"""
    try:
        customer_service = CustomerService(db)
        return await customer_service.get_loyalty_analysis(segment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/retention-analysis")
async def get_retention_analysis(
    period: str = "monthly",
    db: Session = Depends(get_db)
):
    """Get customer retention analysis"""
    try:
        customer_service = CustomerService(db)
        return await customer_service.get_retention_analysis(period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
