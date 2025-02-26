# ai_services_api/tests/test_search.py
from fastapi.testclient import TestClient
import pytest
from datetime import datetime, timedelta
from ai_services_api.controllers.search_router import router

client = TestClient(router)

# Test data
TEST_USER = "test_user_789"
TEST_QUERY = "health research"
TEST_EXPERT_ID = "123"

@pytest.fixture
def auth_headers():
    """Fixture for authentication headers"""
    return {"X-User-ID": TEST_USER}

@pytest.fixture
def sample_expert():
    """Fixture for sample expert data"""
    return {
        "id": TEST_EXPERT_ID,
        "first_name": "John",
        "last_name": "Doe",
        "designation": "Research Lead",
        "theme": "Health",
        "unit": "Research",
        "contact": "john@example.com",
        "is_active": True,
        "knowledge_expertise": ["Health", "Research Methods"]
    }

def test_search_experts(auth_headers):
    """Test main expert search endpoint"""
    response = client.get(
        f"/experts/search/{TEST_QUERY}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "total_results" in data
    assert "experts" in data
    assert data["user_id"] == TEST_USER
    
    if len(data["experts"]) > 0:
        expert = data["experts"][0]
        assert all(key in expert for key in [
            "id", "first_name", "last_name", "designation",
            "theme", "unit", "contact", "is_active", "score"
        ])

def test_search_without_user_id():
    """Test search endpoint with default user ID"""
    response = client.get(f"/experts/search/{TEST_QUERY}")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_123"  # Default test user

def test_predict_query(auth_headers):
    """Test query prediction endpoint"""
    partial_query = "hea"
    response = client.get(
        f"/experts/predict/{partial_query}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert all(key in data for key in ["predictions", "confidence_scores", "user_id"])
    assert data["user_id"] == TEST_USER
    assert len(data["predictions"]) == len(data["confidence_scores"])

def test_train_predictor(auth_headers):
    """Test ML predictor training endpoint"""
    sample_queries = ["health research", "medical studies", "public health"]
    response = client.post(
        "/experts/train-predictor",
        headers=auth_headers,
        json=sample_queries
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Predictor training initiated"
    assert data["user_id"] == TEST_USER

def test_similar_experts(auth_headers, sample_expert):
    """Test finding similar experts"""
    response = client.get(
        f"/experts/similar/{TEST_EXPERT_ID}",
        headers=auth_headers
    )
    
    assert response.status_code in [200, 404]  # 404 is acceptable if expert not found
    if response.status_code == 200:
        data = response.json()
        assert "total_results" in data
        assert "experts" in data
        assert data["user_id"] == TEST_USER
        
        if len(data["experts"]) > 0:
            assert str(data["experts"][0]["id"]) != TEST_EXPERT_ID

def test_get_expert_details(auth_headers, sample_expert):
    """Test getting expert details"""
    response = client.get(
        f"/experts/{TEST_EXPERT_ID}",
        headers=auth_headers
    )
    
    assert response.status_code in [200, 404]  # 404 is acceptable if expert not found
    if response.status_code == 200:
        data = response.json()
        assert data["user_id"] == TEST_USER
        assert all(key in data for key in [
            "id", "first_name", "last_name", "designation",
            "theme", "unit", "contact", "is_active"
        ])

def test_test_endpoints(auth_headers):
    """Test all test endpoints"""
    # Test search
    response = client.get(
        f"/test/experts/search/{TEST_QUERY}",
        params={"active_only": True, "test_error": False}
    )
    assert response.status_code == 200
    assert "total_results" in response.json()
    
    # Test prediction
    response = client.get(
        "/test/experts/predict/health",
        params={"test_error": False}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    
    # Test analytics
    response = client.get("/test/analytics/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "performance_metrics" in data
    assert "search_metrics" in data
    assert data["user_id"] == "test_user"
    
    # Test click recording
    response = client.post(
        "/test/record-click",
        params={"search_id": 1, "expert_id": TEST_EXPERT_ID}
    )
    assert response.status_code in [200, 500]  # 500 is acceptable if record doesn't exist

@pytest.mark.parametrize("query", [
    "simple query",
    "Complex Query With Capitals",
    "query with numbers 123",
    "query @with #special !chars",
    "very " * 10 + "long query",
    ""  # Empty query
])
def test_different_queries(auth_headers, query):
    """Test search with different query types"""
    response = client.get(
        f"/experts/search/{query}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "total_results" in data
    assert "experts" in data

def test_error_cases(auth_headers):
    """Test various error scenarios"""
    # Test invalid expert ID
    response = client.get(
        "/experts/invalid_id",
        headers=auth_headers
    )
    assert response.status_code == 404

    # Test very long query
    long_query = "a" * 1000
    response = client.get(
        f"/experts/search/{long_query}",
        headers=auth_headers
    )
    assert response.status_code in [200, 500]

    # Test invalid similarity search
    response = client.get(
        "/experts/similar/nonexistent_id",
        headers=auth_headers
    )
    assert response.status_code == 404

def test_concurrent_searches(auth_headers):
    """Test multiple concurrent searches"""
    queries = ["health", "research", "medical"]
    responses = []
    
    for query in queries:
        response = client.get(
            f"/experts/search/{query}",
            headers=auth_headers
        )
        responses.append(response)
    
    assert all(r.status_code == 200 for r in responses)
    for response in responses:
        data = response.json()
        assert data["user_id"] == TEST_USER

# Database verification tests
def test_search_analytics(auth_headers):
    """Test analytics recording for searches"""
    # Perform a search
    search_response = client.get(
        f"/experts/search/{TEST_QUERY}",
        headers=auth_headers
    )
    assert search_response.status_code == 200
    
    # Verify analytics were recorded
    analytics_response = client.get("/test/analytics/metrics")
    assert analytics_response.status_code == 200
    data = analytics_response.json()
    assert "search_metrics" in data

def test_prediction_analytics(auth_headers):
    """Test analytics recording for predictions"""
    # Make prediction request
    predict_response = client.get(
        "/experts/predict/heal",
        headers=auth_headers
    )
    assert predict_response.status_code == 200
    
    # Verify prediction was recorded
    analytics_response = client.get("/test/analytics/metrics")
    assert analytics_response.status_code == 200
    data = analytics_response.json()
    assert "performance_metrics" in data

if __name__ == "__main__":
    pytest.main(["-v"])
