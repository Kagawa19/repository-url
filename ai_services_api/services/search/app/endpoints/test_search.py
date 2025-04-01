import requests
import json

def test_expert_search():
    # URL of the expert search endpoint 
    query = "machine learning"
    url = f"http://localhost:8000/search/search/experts/search/{query}"
    
    # Headers including the required X-User-ID
    headers = {
        "X-User-ID": "1"
    }
    
    try:
        # Send GET request to the expert search endpoint
        response = requests.get(url, headers=headers)
        
        # Check response status
        response.raise_for_status()
        
        # Parse and pretty print the JSON response
        response_data = response.json()
        print("Expert Search Response:")
        print(json.dumps(response_data, indent=2))
        
        # Optional: Additional response validation
        assert 'total_results' in response_data, "Response is missing 'total_results' field"
        assert 'experts' in response_data, "Response is missing 'experts' field"
        assert 'user_id' in response_data, "Response is missing 'user_id' field"
        assert 'session_id' in response_data, "Response is missing 'session_id' field"
        
        # Print number of experts found
        print(f"\nTotal experts found: {response_data['total_results']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        if hasattr(e, 'response'):
            print("Response content:", e.response.text)

# Optional: Test query prediction
def test_query_prediction():
    # URL for query prediction endpoint
    partial_query = "machine"
    url = f"http://localhost:8000/search/search/experts/predict/{partial_query}"
    
    # Headers including the required X-User-ID
    headers = {
        "X-User-ID": "1"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        print("\nQuery Prediction Response:")
        response_data = response.json()
        print(json.dumps(response_data, indent=2))
        
        # Validate response structure
        assert 'predictions' in response_data, "Response is missing 'predictions' field"
        assert 'confidence_scores' in response_data, "Response is missing 'confidence_scores' field"
        assert 'user_id' in response_data, "Response is missing 'user_id' field"
        
        # Print predictions
        print("\nPredicted Queries:")
        for pred, conf in zip(response_data['predictions'], response_data['confidence_scores']):
            print(f"- {pred} (Confidence: {conf:.2f})")
        
    except requests.exceptions.RequestException as e:
        print(f"Error checking query prediction: {e}")

# Test health check endpoint
def test_health_check():
    # URL for health check endpoint
    url = "http://localhost:8000/search/search/health"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print("\nHealth Check Response:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"Error performing health check: {e}")

# Run the tests
if __name__ == "__main__":
    test_expert_search()
    test_query_prediction()
    test_health_check()