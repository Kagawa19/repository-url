import requests
import json
import urllib.parse

def test_autocomplete():
    """
    Comprehensive test for the autocomplete endpoint
    """
    # Test cases with different queries
    test_queries = [
        "search",
        "find",
        "machine learning",
        "research",
        "data"
    ]
    
    # Base URL for autocomplete
    base_url = "http://localhost:8000/autocomplete/autocomplete/experts/autocomplete"
    
    # Headers including the required X-User-ID (if needed)
    headers = {
        "X-User-ID": "1"
    }
    
    # Comprehensive test for each query
    for query in test_queries:
        try:
            # URL encode the query parameter
            encoded_query = urllib.parse.quote(query)
            
            # Construct URL with query parameter
            url = f"{base_url}?q={encoded_query}"
            
            # Send GET request to the autocomplete endpoint
            response = requests.get(url, headers=headers)
            
            # Check response status
            response.raise_for_status()
            
            # Parse the JSON response
            response_data = response.json()
            
            # Print detailed response
            print(f"\nAutocomplete for '{query}':")
            print("=" * 40)
            print(json.dumps(response_data, indent=2))
            
            # Validate response
            assert isinstance(response_data, list), f"Response for '{query}' should be a list"
            
            # Check suggestions
            if len(response_data) > 0:
                # Validate each suggestion
                for suggestion in response_data:
                    assert isinstance(suggestion, str), "Each suggestion should be a string"
                    assert len(suggestion) > 0, "Suggestions should not be empty"
                    # Allow suggestions that contain the query but are not exactly the same
                    assert not (query.lower() == suggestion.lower()), "Suggestion should not be identical to the query"
            
            print("\nValidation Passed ✓")
        
        except requests.exceptions.RequestException as e:
            print(f"\nError occurred for query '{query}':")
            print(f"URL: {url}")
            if hasattr(e, 'response'):
                print("Response content:", e.response.text)
            print(f"Error details: {e}")

def test_edge_cases():
    """
    Test edge cases and error scenarios for autocomplete
    """
    # Test cases
    edge_cases = [
        "",             # Empty query
        "a",            # Very short query
        "a" * 200,      # Extremely long query
        "!@#$%^&*"      # Special characters
    ]
    
    base_url = "http://localhost:8000/autocomplete/autocomplete/experts/autocomplete"
    headers = {"X-User-ID": "1"}
    
    print("\nTesting Autocomplete Edge Cases:")
    print("=" * 35)
    
    for query in edge_cases:
        try:
            # URL encode the query parameter
            encoded_query = urllib.parse.quote(query)
            url = f"{base_url}?q={encoded_query}"
            
            response = requests.get(url, headers=headers)
            
            print(f"\nQuery: '{query}'")
            print(f"Status Code: {response.status_code}")
            
            # If successful, print response
            if response.ok:
                response_data = response.json()
                print("Suggestions:", json.dumps(response_data, indent=2))
                
                # Additional validation
                assert isinstance(response_data, list), "Response should be a list"
                # For edge cases, it's okay to have no suggestions
            else:
                print("Error Response:", response.text)
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed for query '{query}': {e}")

def test_health_check():
    """
    Test the health check endpoint
    """
    url = "http://localhost:8000/autocomplete/autocomplete/health"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print("\nHealth Check Response:")
        health_data = response.json()
        print(json.dumps(health_data, indent=2))
        
        # Validate health check response
        assert 'status' in health_data, "Health check should return status"
        assert health_data['status'] == 'healthy', "Service should be healthy"
        
        print("\nHealth Check Passed ✓")
    
    except requests.exceptions.RequestException as e:
        print("Health check failed:", e)

# Run the tests
if __name__ == "__main__":
    print("Starting Autocomplete Testing")
    print("=" * 30)
    
    # Run comprehensive autocomplete test
    test_autocomplete()
    
    # Test edge cases
    test_edge_cases()
    
    # Perform health check
    test_health_check()
    
    print("\nTesting Complete ✓")