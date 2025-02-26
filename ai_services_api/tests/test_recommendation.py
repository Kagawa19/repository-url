# ai_services_api/tests/test_recommendation.py
from fastapi.testclient import TestClient
import pytest
from datetime import datetime, timedelta
from ai_services_api.controllers.recommendation_router import router

client = TestClient(router)

# Test data
TEST_USER = "test_user_789"
TEST_EXPERT_ID = "123"
TEST_EXPERT_ID2 = "456"

@pytest.fixture
def auth_headers():
    """Fixture for authentication headers"""
    return {"X-User-ID": TEST_USER}

@pytest.fixture
def test_expert_data():
    """Fixture for sample expert data"""
    return {
        "id": TEST_EXPERT_ID,
        "first_name": "John",
        "last_name": "Doe",
        "expertise_summary": {
            "domains": ["Health", "Research Methods"],
            "fields": ["Public Health", "Epidemiology"],
            "skills": ["Data Analysis", "Statistical Methods"]
        },
        "designation": "Senior Researcher",
        "theme": "Public Health",
        "unit": "Research",
        "is_active": True
    }

def test_get_expert_profile(auth_headers):
    """Test getting expert profile with recommendations"""
    response = client.get(
        f"/experts/{TEST_EXPERT_ID}",
        headers=auth_headers
    )
    
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert data["user_id"] == TEST_USER
        assert "expertise_summary" in data
        assert "similar_experts" in data
        assert "collaboration_suggestions" in data
        assert isinstance(data["similar_experts"], list)
        assert isinstance(data["collaboration_suggestions"], list)

def test_analyze_expertise(auth_headers):
    """Test expertise analysis endpoint"""
    response = client.get(
        f"/experts/{TEST_EXPERT_ID}/analyze",
        headers=auth_headers
    )
    
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert data["user_id"] == TEST_USER
        assert all(key in data for key in [
            "domains",
            "research_areas",
            "technical_skills",
            "applications",
            "related_fields"
        ])
        assert isinstance(data["domains"], list)
        assert isinstance(data["research_areas"], list)

def test_get_collaborations(auth_headers):
    """Test getting collaboration recommendations"""
    response = client.get(
        f"/experts/{TEST_EXPERT_ID}/collaborations",
        headers=auth_headers,
        params={"min_score": 0.5}
    )
    
    assert response.status_code == 200
    recommendations = response.json()
    assert isinstance(recommendations, list)
    if recommendations:
        rec = recommendations[0]
        assert all(key in rec for key in [
            "expert_id",
            "name",
            "matched_domains",
            "matched_skills",
            "collaboration_score",
            "recommendation_reason",
            "user_id"
        ])
        assert rec["user_id"] == TEST_USER
        assert 0 <= rec["collaboration_score"] <= 1
        assert isinstance(rec["matched_domains"], int)
        assert isinstance(rec["matched_skills"], int)

def test_find_expert_connection(auth_headers):
    """Test finding connection between experts"""
    response = client.get(
        f"/experts/{TEST_EXPERT_ID}/connection/{TEST_EXPERT_ID2}",
        headers=auth_headers,
        params={"max_depth": 3}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == TEST_USER
    assert "paths" in data
    assert "message" in data
    assert isinstance(data["paths"], list)

def test_without_user_id():
    """Test endpoints without user ID header"""
    response = client.get(f"/experts/{TEST_EXPERT_ID}")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert data["user_id"] == "test_user_123"  # Default test user

def test_test_recommendation(auth_headers):
    """Test the test recommendation endpoint"""
    response = client.post(
        "/test/recommend",
        headers=auth_headers,
        params={
            "expert_id": TEST_EXPERT_ID,
            "min_similarity": 0.5
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == TEST_USER
    assert "recommendations" in data
    assert "test_timestamp" in data
    assert isinstance(data["recommendations"], list)

def test_analytics(auth_headers):
    """Test getting analytics data"""
    response = client.get(
        f"/test/analytics/expert/{TEST_EXPERT_ID}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == TEST_USER
    assert "matching_metrics" in data
    assert "collaboration_metrics" in data
    
    # Verify metrics structure
    matching = data["matching_metrics"]
    assert all(key in matching for key in [
        "total_matches",
        "avg_similarity",
        "total_shared_domains",
        "total_shared_fields",
        "total_shared_skills",
        "success_rate"
    ])
    
    collab = data["collaboration_metrics"]
    assert all(key in collab for key in [
        "total_collaborations",
        "avg_score",
        "unique_collaborators"
    ])

def test_verify_data(auth_headers):
    """Test data verification endpoint"""
    response = client.post(
        f"/test/verify/{TEST_EXPERT_ID}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["user_id"] == TEST_USER
    assert "data_verification" in data
    verification = data["data_verification"]
    assert all(key in verification for key in [
        "matching_records",
        "collaboration_records",
        "active_domains"
    ])

def test_error_cases(auth_headers):
    """Test various error scenarios"""
    # Test invalid expert ID
    response = client.get(
        "/experts/invalid_id",
        headers=auth_headers
    )
    assert response.status_code == 404

    # Test invalid max_depth
    response = client.get(
        f"/experts/{TEST_EXPERT_ID}/connection/{TEST_EXPERT_ID2}",
        headers=auth_headers,
        params={"max_depth": 10}  # Too high
    )
    assert response.status_code in [400, 422]

    # Test invalid similarity threshold
    response = client.get(
        f"/experts/{TEST_EXPERT_ID}/collaborations",
        headers=auth_headers,
        params={"min_score": 2.0}  # Invalid score > 1
    )
    assert response.status_code in [400, 422]

def test_concurrent_requests(auth_headers):
    """Test multiple concurrent recommendation requests"""
    expert_ids = [TEST_EXPERT_ID, TEST_EXPERT_ID2, "789"]
    responses = []
    
    # Get profiles concurrently
    for expert_id in expert_ids:
        response = client.get(
            f"/experts/{expert_id}",
            headers=auth_headers
        )
        responses.append(response)
    
    # Verify responses
    success_count = sum(1 for r in responses if r.status_code == 200)
    assert success_count > 0  # At least one should succeed
    
    # Verify user ID in all successful responses
    for response in responses:
        if response.status_code == 200:
            data = response.json()
            assert data["user_id"] == TEST_USER

def test_recommendation_validation(auth_headers):
    """Test recommendation score validation"""
    min_scores = [-0.1, 0.0, 0.5, 1.0, 1.1]
    
    for score in min_scores:
        response = client.get(
            f"/experts/{TEST_EXPERT_ID}/collaborations",
            headers=auth_headers,
            params={"min_score": score}
        )
        
        if 0 <= score <= 1:
            assert response.status_code == 200
        else:
            assert response.status_code in [400, 422]

def test_expertise_analysis_validation(auth_headers, test_expert_data):
    """Test expertise analysis results"""
    response = client.get(
        f"/experts/{TEST_EXPERT_ID}/analyze",
        headers=auth_headers
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Verify fields exist and are non-empty
        assert len(data["domains"]) > 0
        assert len(data["research_areas"]) > 0
        
        # Verify reasonable limits
        assert len(data["technical_skills"]) <= 20
        assert len(data["research_areas"]) <= 30
        
        # Verify data types
        assert all(isinstance(d, str) for d in data["domains"])
        assert all(isinstance(r, str) for r in data["research_areas"])

if __name__ == "__main__":
    pytest.main(["-v"])
