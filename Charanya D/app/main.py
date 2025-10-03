"""
FastAPI application for Retail Store & Customer Insights
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from routers import stores, customers, analytics
from core.config import settings
from core.database import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Retail Store & Customer Insights API",
    description="MLOps Capstone Project - Analytics pipeline for retail data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(stores.router, prefix="/api/v1/stores", tags=["stores"])
app.include_router(customers.router, prefix="/api/v1/customers", tags=["customers"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])

@app.get("/", response_class=HTMLResponse)
async def root():
    """Home page with API information"""
    return """
    <html>
        <head>
            <title>Retail Analytics API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #2c3e50; }
                .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
                a { color: #3498db; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1 class="header">🏪 Retail Store & Customer Insights API</h1>
            <p>Welcome to the MLOps Capstone Project API for retail analytics.</p>
            
            <h2>📊 Available Endpoints:</h2>
            <div class="endpoint">
                <strong>Stores:</strong> <a href="/api/v1/stores/performance">/api/v1/stores/performance</a>
            </div>
            <div class="endpoint">
                <strong>Customers:</strong> <a href="/api/v1/customers/segments">/api/v1/customers/segments</a>
            </div>
            <div class="endpoint">
                <strong>Analytics:</strong> <a href="/api/v1/analytics/rfm">/api/v1/analytics/rfm</a>
            </div>
            
            <h2>📖 Documentation:</h2>
            <p><a href="/docs">Interactive API Documentation (Swagger)</a></p>
            <p><a href="/redoc">Alternative Documentation (ReDoc)</a></p>
            
            <p><em>Built with FastAPI for high-performance retail analytics</em></p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
