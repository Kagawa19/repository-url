import requests
import json
import urllib.parse

def test_message_draft():
    """
    Comprehensive test for the message draft endpoint
    """
    # Test cases with different receiver IDs and content
    test_cases = [
        {"receiver_id": "1", "content": "Discussing research publication"},
        {"receiver_id": "2", "content": "Collaboration opportunity in machine learning"},
        {"receiver_id": "3", "content": "Interested in your recent work on AI ethics"}
    ]
    
    # Base URL for message draft
    base_url = "http://localhost:8000/message/message/draft"
    
    # Headers including the required X-User-ID
    headers = {
        "X-User-ID": "1"
    }
    
    # Comprehensive test for each test case
    for case in test_cases:
        try:
            # URL encode the content
            encoded_content = urllib.parse.quote(case['content'])
            
            # Construct full URL
            url = f"{base_url}/{case['receiver_id']}/{encoded_content}"
            
            # Send GET request to the message draft endpoint
            response = requests.get(url, headers=headers)
            
            # Check response status
            response.raise_for_status()
            
            # Parse the JSON response
            response_data = response.json()
            
            # Print detailed response
            print(f"\nMessage Draft for Receiver {case['receiver_id']}:")
            print("=" * 40)
            print(json.dumps(response_data, indent=2))
            
            # Validate response structure
            assert 'id' in response_data, f"Response is missing 'id' field"
            assert 'content' in response_data, f"Response is missing 'content' field"
            assert 'sender_id' in response_data, f"Response is missing 'sender_id' field"
            assert 'receiver_id' in response_data, f"Response is missing 'receiver_id' field"
            assert 'created_at' in response_data, f"Response is missing 'created_at' field"
            assert 'receiver_name' in response_data, f"Response is missing 'receiver_name' field"
            
            # Validate content
            content = response_data['content']
            assert len(content) > 0, "Draft content should not be empty"
            assert len(content) < 1000, "Draft content should not be excessively long"
            
            print("\nContent Preview:")
            print(content[:200] + "..." if len(content) > 200 else content)
            
            print("\nValidation Passed ✓")
        
        except requests.exceptions.RequestException as e:
            print(f"\nError occurred for receiver {case['receiver_id']}:")
            print(f"URL: {url}")
            if hasattr(e, 'response'):
                print("Response content:", e.response.text)
            print(f"Error details: {e}")

def test_test_message_draft():
    """
    Test the test endpoint for message draft
    """
    # Test cases with different receiver IDs and content
    test_cases = [
        {"receiver_id": "1", "content": "Test publication discussion"},
        {"receiver_id": "2", "content": "Exploring research collaboration"}
    ]
    
    # Base URL for test message draft
    base_url = "http://localhost:8000/message/message/test/draft"
    
    # Headers 
    headers = {
        "X-User-ID": "1"
    }
    
    # Test each case
    for case in test_cases:
        try:
            # URL encode the content
            encoded_content = urllib.parse.quote(case['content'])
            
            # Construct full URL
            url = f"{base_url}/{case['receiver_id']}/{encoded_content}"
            
            # Send GET request to the test message draft endpoint
            response = requests.get(url, headers=headers)
            
            # Check response status
            response.raise_for_status()
            
            # Parse the JSON response
            response_data = response.json()
            
            # Print detailed response
            print(f"\nTest Message Draft for Receiver {case['receiver_id']}:")
            print("=" * 40)
            print(json.dumps(response_data, indent=2))
            
            # Validate response structure
            assert 'id' in response_data, "Test response is missing 'id' field"
            assert 'content' in response_data, "Test response is missing 'content' field"
            
            print("\nTest Message Draft Validation Passed ✓")
        
        except requests.exceptions.RequestException as e:
            print("\nError in test message draft:")
            if hasattr(e, 'response'):
                print("Response content:", e.response.text)
            print(f"Error details: {e}")

# Run the tests
if __name__ == "__main__":
    print("Starting Message Draft Endpoint Testing")
    print("=" * 40)
    
    # Run message draft tests
    test_message_draft()
    
    # Run test message draft endpoint
    test_test_message_draft()
    
    print("\nTesting Complete ✓")