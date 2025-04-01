import requests
import json
import urllib.parse

def test_expert_search():
    """
    Comprehensive test for the expert search endpoint
    """
    # Test cases with different queries
    test_queries = [
        "machine learning",
        "nutrition",
        "artificial intelligence",
        "research",
        "computer vision"
    ]
    
    # Base URL for search
    base_url = "http://localhost:8000/search/search/experts/search"
    
    # Headers including the required X-User-ID
    headers = {
        "X-User-ID": "1"
    }
    
    # Comprehensive test for each query
    for query in test_queries:
        try:
            # URL encode the query
            encoded_query = urllib.parse.quote(query)
            
            # Construct full URL
            url = f"{base_url}/{encoded_query}"
            
            # Send GET request to the search endpoint
            response = requests.get(url, headers=headers)
            
            # Check response status
            response.raise_for_status()
            
            # Parse the JSON response
            response_data = response.json()
            
            # Print detailed response
            print(f"\nSearch Results for '{query}':")
            print("=" * 40)
            print(json.dumps(response_data, indent=2))
            
            # Validate response structure
            assert 'total_results' in response_data, f"Response for '{query}' is missing 'total_results' field"
            assert 'experts' in response_data, f"Response for '{query}' is missing 'experts' field"
            assert 'user_id' in response_data, f"Response for '{query}' is missing 'user_id' field"
            assert 'session_id' in response_data, f"Response for '{query}' is missing 'session_id' field"
            
            # Validate experts
            experts = response_data['experts']
            assert isinstance(experts, list), f"Experts should be a list for query '{query}'"
            
            # Print summary
            print(f"\nTotal Experts Found: {response_data['total_results']}")
            
            # Detailed expert validation
            if experts:
                print("\nSample Experts:")
                for i, expert in enumerate(experts[:3], 1):
                    print(f"{i}. {expert.get('first_name', 'Unknown')} {expert.get('last_name', '')}")
                    
                    # Validate each expert has required fields
                    assert 'id' in expert, f"Expert is missing 'id' field for query '{query}'"
                    assert 'first_name' in expert, f"Expert is missing 'first_name' field for query '{query}'"
                    assert 'last_name' in expert, f"Expert is missing 'last_name' field for query '{query}'"
            
            # Validate refinements (optional)
            if 'refinements' in response_data:
                refinements = response_data['refinements']
                print("\nRefinement Suggestions:")
                
                # Validate refinement structure
                assert isinstance(refinements.get('filters', []), list), "Filters should be a list"
                assert isinstance(refinements.get('related_queries', []), list), "Related queries should be a list"
                assert isinstance(refinements.get('expertise_areas', []), list), "Expertise areas should be a list"
                
                # Print refinement details
                print("Filters:", refinements.get('filters', []))
                print("Related Queries:", refinements.get('related_queries', []))
                print("Expertise Areas:", refinements.get('expertise_areas', []))
            
            print("\nValidation Passed ✓")
        
        except requests.exceptions.RequestException as e:
            print(f"\nError occurred for query '{query}':")
            print(f"URL: {url}")
            if hasattr(e, 'response'):
                print("Response content:", e.response.text)
            print(f"Error details: {e}")
            raise

def test_search_edge_cases():
    """
    Test edge cases and error scenarios for search
    """
    # Test cases
    edge_cases = [
        "",             # Empty query
        "a",            # Very short query
        "a" * 200,      # Extremely long query
        "!@#$%^&*"      # Special characters
    ]
    
    base_url = "http://localhost:8000/search/search/experts/search"
    headers = {"X-User-ID": "1"}
    
    print("\nTesting Search Edge Cases:")
    print("=" * 35)
    
    for query in edge_cases:
        try:
            # URL encode the query parameter
            encoded_query = urllib.parse.quote(query)
            url = f"{base_url}/{encoded_query}"
            
            response = requests.get(url, headers=headers)
            
            print(f"\nQuery: '{query}'")
            print(f"Status Code: {response.status_code}")
            
            # If successful, print response
            if response.ok:
                response_data = response.json()
                print("Search Results:", json.dumps(response_data, indent=2))
                
                # Basic validations
                assert 'experts' in response_data, "Response should have 'experts' field"
                assert isinstance(response_data['experts'], list), "Experts should be a list"
            else:
                print("Error Response:", response.text)
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed for query '{query}': {e}")

# Run the tests
if __name__ == "__main__":
    print("Starting Expert Search Endpoint Testing")
    print("=" * 40)
    
    # Run comprehensive search tests
    test_expert_search()
    
    # Test edge cases
    test_search_edge_cases()
    
    print("\nTesting Complete ✓")