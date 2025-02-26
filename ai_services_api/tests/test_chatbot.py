from fastapi.testclient import TestClient
import pytest
from datetime import datetime, timedelta
from ai_services_api.controllers.chatbot_router import api_router as chatbot_router
from fastapi import FastAPI
from ai_services_api.tests.db_utils import DatabaseConnector

client = TestClient(chatbot_router)

# Initialize the database connector
db_connector = DatabaseConnector()

# Test data
TEST_USER = "test_user_789"
TEST_QUERY = "where are the health publications?"

@pytest.fixture
def auth_headers():
    return {"X-User-ID": TEST_USER}

def test_chat_endpoint(auth_headers):
    """Test main chat endpoint"""
    response = client.get(
        f"/conversation/chat/{TEST_QUERY}",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == TEST_USER
    assert "response" in data
    assert "session_id" in data
    assert "metrics" in data
    return data["session_id"]  # Return for other tests

def test_chat_without_user_id():
    """Test chat endpoint without user ID"""
    response = client.get(f"/conversation/chat/{TEST_QUERY}")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] != TEST_USER  # Expect a different user ID
    assert "response" in data

def test_chat_metrics(auth_headers):
    """Test getting chat metrics"""
    # First create a chat session
    chat_response = client.get(
        f"/conversation/chat/{TEST_QUERY}",
        headers=auth_headers
    )
    session_id = chat_response.json()["session_id"]
    
    # Then get metrics
    response = client.get(
        f"/conversation/chat/metrics/{session_id}",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_interactions" in data
    assert "avg_response_time" in data
    assert "success_rate" in data
    assert "content_matches" in data

def test_content_click(auth_headers):
    """Test content click recording"""
    # First create a chat session and get content
    chat_response = client.get(
        f"/conversation/chat/{TEST_QUERY}",
        headers=auth_headers
    )
    
    # Then simulate a content click
    response = client.post(
        "/conversation/chat/content-click",
        headers=auth_headers,
        json={
            "interaction_id": 1,  # Use first interaction
            "content_id": "1",
            "content_type": "navigation"
        }
    )
    assert response.status_code in [200, 404]  # 404 is acceptable if no content match found

def test_chat_analytics(auth_headers):
    """Test getting chat analytics"""
    # Set up date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1)
    
    response = client.get(
        "/conversation/chat/analytics",
        headers=auth_headers,
        params={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "interactions" in data
    assert "content_matching" in data
    assert "intents" in data

def test_end_session(auth_headers):
    """Test ending a chat session"""
    # First create a session
    chat_response = client.get(
        f"/conversation/chat/{TEST_QUERY}",
        headers=auth_headers
    )
    session_id = chat_response.json()["session_id"]
    
    # Then end it
    response = client.post(
        f"/conversation/chat/end-session/{session_id}",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_messages" in data
    assert "duration_seconds" in data
    assert data["session_id"] == session_id

def test_unauthorized_session_access(auth_headers):
    """Test unauthorized access to session data"""
    response = client.get(
        "/conversation/chat/metrics/nonexistent_session",
        headers=auth_headers
    )
    assert response.status_code == 404

def test_rate_limiting():
    """Test rate limiting"""
    headers = {"X-User-ID": "test_rate_limit_user"}
    # Make 6 requests (limit is 5/minute)
    responses = []
    for _ in range(6):
        response = client.get(
            f"/conversation/chat/{TEST_QUERY}",
            headers=headers
        )
        responses.append(response.status_code)
    
    # At least one should be rate limited (429)
    assert 429 in responses

@pytest.mark.parametrize("query", [
    "Simple query",
    "Complex query with multiple words",
    "Query with special !@#$ characters",
    "Very " * 50 + "long query",  # Long query
    ""  # Empty query
])
def test_different_queries(auth_headers, query):
    """Test different types of queries"""
    response = client.get(
        f"/conversation/chat/{query}",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data

# Test the development/debug endpoints
def test_test_endpoints(auth_headers):
    """Test the test/debug endpoints"""
    # Test chat endpoint
    response = client.get(
        f"/conversation/test/chat/{TEST_QUERY}",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "test_matches" in data
    session_id = data["session_id"]
    
    # Verify session data
    verify_response = client.get(
        f"/conversation/test/verify/{session_id}",
        headers=auth_headers
    )
    assert verify_response.status_code == 200
    verify_data = verify_response.json()
    assert verify_data["verification_status"] == "success"
    assert "session_data" in verify_data

if __name__ == "__main__":
    pytest.main(["-v"])
