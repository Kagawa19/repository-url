# phrc_interactive_search_tester.py
import requests
import json
from typing import Dict, Any, Optional, List

BASE_URL = "http://localhost:8000"  # Change to your API's base URL
TEST_USER = "test_user_789"

def print_header():
    print("\n" + "=" * 30)
    print("PHRC Interactive Search Tester")
    print("=" * 30)

def make_request(endpoint: str, params: Optional[Dict[str, Any]] = None, 
                json_data: Optional[Dict[str, Any]] = None, method: str = "GET"):
    """Make a request to the API with the specified parameters."""
    headers = {"X-User-ID": TEST_USER}
    url = f"{BASE_URL}/{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, params=params, json=json_data)
        else:
            print(f"Unsupported method: {method}")
            return None
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
        
        return response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def display_results(data: Dict[str, Any]):
    """Display search results in a readable format."""
    print("\nResults:")
    print(f"Total results: {data.get('total_results', 0)}")
    
    experts = data.get("experts", [])
    if not experts:
        print("No experts found.")
        return
    
    for i, expert in enumerate(experts, 1):
        print(f"\n{i}. {expert.get('first_name', '')} {expert.get('last_name', '')}")
        print(f"   Designation: {expert.get('designation', '')}")
        print(f"   Theme: {expert.get('theme', '')}")
        print(f"   Score: {expert.get('score', '')}")
        
        expertise = expert.get('knowledge_expertise', [])
        if expertise:
            print(f"   Expertise: {', '.join(expertise)}")

def get_prediction(partial_query: str, search_type: Optional[str] = None):
    """Get predictions for a partial query."""
    params = {}
    if search_type:
        params["search_type"] = search_type
        
    data = make_request(f"experts/advanced_predict/{partial_query}", params=params)
    
    if not data:
        return []
        
    return data.get("predictions", [])

def display_predictions(predictions: list):
    """Display predictions in a numbered list."""
    if not predictions:
        print("No predictions available.")
        return
    
    print("\nPredictions:")
    for i, prediction in enumerate(predictions, 1):
        print(f"{i}. {prediction}")

def track_selection(partial_query: str, selected_suggestion: str):
    """Track the user's selection."""
    payload = {
        "partial_query": partial_query,
        "selected_suggestion": selected_suggestion
    }
    
    result = make_request("experts/track-suggestion", json_data=payload, method="POST")
    
    if result and result.get("status") == "success":
        print(f"Selection '{selected_suggestion}' tracked successfully.")
    else:
        print("Failed to track selection.")

def normal_search():
    """Perform a normal search."""
    print("\n=== Normal Search ===")
    
    # Get partial query and show predictions
    partial_query = input("Start typing your query: ")
    
    if not partial_query:
        print("Query cannot be empty.")
        return
    
    predictions = get_prediction(partial_query)
    display_predictions(predictions)
    
    # Let user select a prediction or type full query
    choice = input("\nSelect a prediction number or press Enter to type full query: ")
    
    if choice.isdigit() and 1 <= int(choice) <= len(predictions):
        selected_prediction = predictions[int(choice) - 1]
        print(f"Selected: {selected_prediction}")
        
        # Track the selection
        track_selection(partial_query, selected_prediction)
        
        # Use selected prediction as query
        query = selected_prediction
    else:
        query = input("Enter full search query: ")
        
    if not query:
        print("Search query cannot be empty.")
        return
    
    data = make_request(f"experts/search/{query}")
    if data:
        display_results(data)

def combined_search():
    """Search with multiple parameters combined in a single request."""
    print("\n=== Combined Parameters Search ===")
    print("This search finds experts matching MULTIPLE criteria at once")
    
    # Create parameters dictionary
    params = {}
    
    # Select parameter combination from pre-defined options
    print("\nChoose a parameter combination to test:")
    print("1. Name + Theme (experts named 'X' working on theme 'Y')")
    print("2. Theme + Designation (experts with theme 'X' and job title 'Y')")
    print("3. Name + Designation + Theme (most specific)")
    print("4. Query + Search Type (contextual search)")
    print("5. Custom combination (specify your own parameters)")
    
    option = input("\nSelect option (1-5): ")
    
    if option == "1":
        # Name + Theme
        name = input("\nEnter expert name: ")
        theme = input("Enter research theme: ")
        
        if not name or not theme:
            print("Both parameters are required for this option.")
            return
            
        params["name"] = name
        params["theme"] = theme
        
    elif option == "2":
        # Theme + Designation
        theme = input("\nEnter research theme: ")
        designation = input("Enter job title/designation: ")
        
        if not theme or not designation:
            print("Both parameters are required for this option.")
            return
            
        params["theme"] = theme
        params["designation"] = designation
        
    elif option == "3":
        # Name + Designation + Theme (triple combination)
        name = input("\nEnter expert name: ")
        designation = input("Enter job title/designation: ")
        theme = input("Enter research theme: ")
        
        if not name or not designation or not theme:
            print("All three parameters are required for this option.")
            return
            
        params["name"] = name
        params["designation"] = designation
        params["theme"] = theme
        
    elif option == "4":
        # Query + Search Type
        query = input("\nEnter search query: ")
        print("Search type options: name, theme, designation, publication")
        search_type = input("Enter search type: ")
        
        if not query or not search_type or search_type not in ["name", "theme", "designation", "publication"]:
            print("Valid query and search type are required for this option.")
            return
            
        params["query"] = query
        params["search_type"] = search_type
        
    elif option == "5":
        # Custom combination
        print("\nEnter values for any parameters you want to include (leave blank to skip):")
        
        query = input("Query: ")
        if query:
            params["query"] = query
            
        print("\nSearch type options: name, theme, designation, publication")
        search_type = input("Search type: ")
        if search_type and search_type in ["name", "theme", "designation", "publication"]:
            params["search_type"] = search_type
            
        name = input("\nExpert name: ")
        if name:
            params["name"] = name
            
        theme = input("Research theme: ")
        if theme:
            params["theme"] = theme
            
        designation = input("Designation/job title: ")
        if designation:
            params["designation"] = designation
            
        publication = input("Publication: ")
        if publication:
            params["publication"] = publication
            
        if not params:
            print("At least one parameter must be provided.")
            return
    else:
        print("Invalid option.")
        return
        
    # Show final parameter combination
    print("\n=== Final Combined Parameters ===")
    for key, value in params.items():
        print(f"{key}: {value}")
        
    print("\nThis will search for experts matching ALL these criteria simultaneously.")
    
    # Execute the search
    print("\nExecuting combined search...")
    data = make_request("advanced_search", params=params)
    
    if data:
        print(f"\nExperts matching ALL criteria: {data.get('total_results', 0)}")
        display_results(data)

def main():
    while True:
        print_header()
        print("1. Normal Search")
        print("2. Combined Search")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            normal_search()
        elif choice == "2":
            combined_search()
        elif choice == "3":
            print("Exiting tester. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()