import requests
import json

def test_expert_recommendations():
    """
    Comprehensive test for the expert recommendations endpoint
    """
    # Test cases with different user IDs
    test_user_ids = [
        "1"  # Use user ID 1 as specified
    ]
    
    # Base URL for recommendations
    base_url = "http://localhost:8000/recommendation/recommendation/recommend"
    
    # Headers including the required X-User-ID
    headers = {
        "X-User-ID": "1"
    }
    
    # Comprehensive test for each user ID
    for user_id in test_user_ids:
        try:
            # Construct full URL
            url = f"{base_url}/{user_id}"
            
            # Send GET request to the recommendations endpoint
            response = requests.get(url, headers=headers)
            
            # Check response status
            response.raise_for_status()
            
            # Parse the JSON response
            response_data = response.json()
            
            # Print detailed response
            print(f"\nRecommendations for User ID '{user_id}':")
            print("=" * 40)
            print(json.dumps(response_data, indent=2))
            
            # Validate response structure
            assert 'user_id' in response_data, f"Response for '{user_id}' is missing 'user_id' field"
            assert 'recommendations' in response_data, f"Response for '{user_id}' is missing 'recommendations' field"
            assert 'total_matches' in response_data, f"Response for '{user_id}' is missing 'total_matches' field"
            assert 'timestamp' in response_data, f"Response for '{user_id}' is missing 'timestamp' field"
            
            # Validate recommendations
            recommendations = response_data['recommendations']
            
            # Print summary
            print(f"\nTotal Recommendations: {len(recommendations)}")
            if recommendations:
                print("Sample Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"{i}. {rec.get('name', 'Unnamed Expert')}")
            
            print("\nValidation Passed ✓")
        
        except requests.exceptions.RequestException as e:
            print(f"\nError occurred for user ID '{user_id}':")
            print(f"URL: {url}")
            if hasattr(e, 'response'):
                print("Response content:", e.response.text)
            print(f"Error details: {e}")

def test_test_recommendations():
    """
    Test the test endpoint for recommendations
    """
    # Base URL for test recommendations
    url = "http://localhost:8000/recommendation/recommendation/test/recommend"
    
    # Headers 
    headers = {
        "X-User-ID": "test_user_123"
    }
    
    try:
        # Send GET request to the test recommendations endpoint
        response = requests.get(url, headers=headers)
        
        # Check response status
        response.raise_for_status()
        
        # Parse the JSON response
        response_data = response.json()
        
        # Print detailed response
        print("\nTest Recommendations:")
        print("=" * 25)
        print(json.dumps(response_data, indent=2))
        
        # Validate response structure
        assert 'user_id' in response_data, "Test response is missing 'user_id' field"
        assert 'recommendations' in response_data, "Test response is missing 'recommendations' field"
        
        print("\nTest Recommendations Validation Passed ✓")
    
    except requests.exceptions.RequestException as e:
        print("\nError in test recommendations:")
        if hasattr(e, 'response'):
            print("Response content:", e.response.text)
        print(f"Error details: {e}")

# Run the tests
if __name__ == "__main__":
    print("Starting Recommendation Endpoint Testing")
    print("=" * 40)
    
    # Run recommendation tests
    test_expert_recommendations()
    
    # Run test recommendations endpoint
    test_test_recommendations()
    
    print("\nTesting Complete ✓")