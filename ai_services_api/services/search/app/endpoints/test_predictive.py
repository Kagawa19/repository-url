import requests
import json

def test_predictive_search():
    """
    Test various scenarios for predictive search endpoint
    """
    # Test cases with different partial queries
    test_queries = [
        "machine",
        "data",
        "ai",
        "learning",
        "natural language",  # Add more diverse queries
        "computer vision"
    ]
    
    # Base URL for predictive search
    base_url = "http://localhost:8000/search/search/experts/predict"
    
    # Headers including the required X-User-ID
    headers = {
        "X-User-ID": "1"
    }
    
    # Comprehensive test for each query
    for query in test_queries:
        try:
            # Construct full URL
            url = f"{base_url}/{query}"
            
            # Send GET request to the predictive search endpoint
            response = requests.get(url, headers=headers)
            
            # Check response status
            response.raise_for_status()
            
            # Parse the JSON response
            response_data = response.json()
            
            # Print detailed response
            print(f"\nPredictive Search for '{query}':")
            print("=" * 40)
            print(json.dumps(response_data, indent=2))
            
            # Validate response structure
            assert 'predictions' in response_data, f"Response for '{query}' is missing 'predictions' field"
            assert 'confidence_scores' in response_data, f"Response for '{query}' is missing 'confidence_scores' field"
            assert 'user_id' in response_data, f"Response for '{query}' is missing 'user_id' field"
            assert 'refinements' in response_data, f"Response for '{query}' is missing 'refinements' field"  # Add this line
            
            # Validate predictions and confidence scores
            predictions = response_data['predictions']
            confidence_scores = response_data['confidence_scores']
            
            # Check that we have predictions
            assert len(predictions) > 0, f"No predictions found for query '{query}'"
            
            # Validate predictions match confidence scores
            assert len(predictions) == len(confidence_scores), \
                f"Mismatch between predictions and confidence scores for '{query}'"
            
            # Print out predictions
            print("\nPredicted Queries:")
            for pred, conf in zip(predictions, confidence_scores):
                print(f"- {pred} (Confidence: {conf:.2f})")
            
            # Validate refinements
            refinements = response_data['refinements']
            assert isinstance(refinements, dict), f"Refinements for '{query}' should be a dictionary"
            assert 'filters' in refinements, f"Refinements for '{query}' should contain 'filters'"
            assert 'related_queries' in refinements, f"Refinements for '{query}' should contain 'related_queries'"
            assert 'expertise_areas' in refinements, f"Refinements for '{query}' should contain 'expertise_areas'"
            
            # Print out refinement suggestions
            print("\nRefinement Suggestions:")
            print(f"Filters: {refinements['filters']}")
            print(f"Related Queries: {refinements['related_queries']}")
            print(f"Expertise Areas: {refinements['expertise_areas']}")
            
            print("\nValidation Passed ✓")
        
        except requests.exceptions.RequestException as e:
            print(f"\nError occurred for query '{query}':")
            print(f"URL: {url}")
            if hasattr(e, 'response'):
                print("Response content:", e.response.text)
            print(f"Error details: {e}")

...


def test_edge_cases():
    """
    Test edge cases and error scenarios
    """
    # Test cases
    edge_cases = [
        "",             # Empty query
        "x",            # Very short query
        "a" * 1000,     # Extremely long query
        "!@#$%^&*"      # Special characters
    ]
    
    base_url = "http://localhost:8000/search/search/experts/predict"
    headers = {"X-User-ID": "1"}
    
    print("\nTesting Edge Cases:")
    print("=" * 30)
    
    for query in edge_cases:
        try:
            url = f"{base_url}/{query}"
            response = requests.get(url, headers=headers)
            
            print(f"\nQuery: '{query}'")
            print(f"Status Code: {response.status_code}")
            
            # If successful, print response
            if response.ok:
                print("Response:", json.dumps(response.json(), indent=2))
            else:
                print("Error Response:", response.text)
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed for query '{query}': {e}")

# Run the tests
# Run the tests
if __name__ == "__main__":
    print("Starting Predictive Search Testing")
    print("=" * 30)
    
    # Run comprehensive predictive search test
    test_predictive_search()
    
    # Test edge cases
    test_edge_cases()
    
    print("\nTesting Complete ✓")