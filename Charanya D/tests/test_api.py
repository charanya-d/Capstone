"""
Test cases for API endpoints
"""

import pytest
from fastapi.testclient import TestClient

def test_health_check(client: TestClient):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data

def test_root_endpoint(client: TestClient):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_store_performance_endpoint(client: TestClient):
    """Test store performance endpoint"""
    response = client.get("/api/v1/stores/performance")
    assert response.status_code == 200
    # The response might be empty or mock data, so just check it's a list
    data = response.json()
    assert isinstance(data, list)

def test_customer_segments_endpoint(client: TestClient):
    """Test customer segments endpoint"""
    response = client.get("/api/v1/customers/segments")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_rfm_analysis_endpoint(client: TestClient):
    """Test RFM analysis endpoint"""
    response = client.get("/api/v1/analytics/rfm")
    assert response.status_code == 200
    data = response.json()
    assert "total_customers" in data
    assert "segment_distribution" in data
    assert "avg_scores" in data

def test_profitability_analysis_endpoint(client: TestClient):
    """Test profitability analysis endpoint"""
    response = client.get("/api/v1/analytics/profitability")
    assert response.status_code == 200
    data = response.json()
    assert "period" in data
    assert "overall_metrics" in data

def test_seasonality_analysis_endpoint(client: TestClient):
    """Test seasonality analysis endpoint"""
    response = client.get("/api/v1/analytics/seasonality")
    assert response.status_code == 200
    data = response.json()
    assert "granularity" in data
    assert "trends" in data

def test_payment_methods_endpoint(client: TestClient):
    """Test payment methods analysis endpoint"""
    response = client.get("/api/v1/analytics/payment-methods")
    assert response.status_code == 200
    data = response.json()
    assert "payment_methods" in data or "error" not in data

def test_campaign_simulation_endpoint(client: TestClient):
    """Test campaign simulation endpoint"""
    response = client.post("/api/v1/analytics/campaign-simulation", 
                          params={"discount_percentage": 10.0, "target_segment": "high_value"})
    assert response.status_code == 200
    data = response.json()
    assert "campaign_type" in data
    assert "estimated_reach" in data

@pytest.mark.parametrize("endpoint", [
    "/api/v1/stores/performance",
    "/api/v1/customers/segments", 
    "/api/v1/analytics/rfm",
    "/api/v1/analytics/profitability",
    "/api/v1/analytics/seasonality"
])
def test_api_endpoints_return_200(client: TestClient, endpoint: str):
    """Test that all main endpoints return 200 status"""
    response = client.get(endpoint)
    assert response.status_code == 200
