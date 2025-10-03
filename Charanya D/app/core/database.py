"""
Database configuration and models
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from .config import settings

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

class SalesTransaction(Base):
    """Sales transaction model"""
    __tablename__ = "sales_transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    invoice_no = Column(String, index=True)
    stock_code = Column(String, index=True)
    description = Column(String)
    quantity = Column(Integer)
    invoice_date = Column(DateTime)
    unit_price = Column(Float)
    customer_id = Column(String, index=True)
    country = Column(String)
    
class Customer(Base):
    """Customer model"""
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, unique=True, index=True)
    age = Column(Integer)
    gender = Column(String)
    item = Column(String)
    quantity = Column(Integer)
    category = Column(String)
    price = Column(Float)
    payment_method = Column(String)
    invoice_date = Column(DateTime)
    shopping_mall = Column(String)
    
class CustomerSegment(Base):
    """Customer segmentation model"""
    __tablename__ = "customer_segments"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    segment = Column(String)
    rfm_score = Column(String)
    recency = Column(Integer)
    frequency = Column(Integer)
    monetary = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class StorePerformance(Base):
    """Store performance model"""
    __tablename__ = "store_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(String, index=True)
    region = Column(String)
    total_sales = Column(Float)
    total_transactions = Column(Integer)
    avg_transaction_value = Column(Float)
    profit_margin = Column(Float)
    period = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
