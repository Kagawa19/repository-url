import requests
import json
import time
import logging
import os
import sys
import readline
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your API base URL
USER_ID = "test_user_123"
MAX_SUGGESTIONS = 8

class InteractiveSearchTester:
    """Interactive search tester with context-aware predictions for advanced search."""
    
    def __init__(self, base_url: str, user_id: str):
        self.base_url = base_url
        self.user_id = user_id
        self.headers = {
            "X-User-ID": user_id,
            "Content-Type": "application/json"
        }
    
    def get_predictions(self, partial_query: str, context_type: Optional[str] = None) -> List[str]:
        """
        Get search predictions for a partial query with context filtering.
        
        Args:
            partial_query: Partial query to get predictions for
            context_type: Optional context type for filtering (name, theme, designation)
            
        Returns:
            List of prediction strings
        """
        # Make sure we're explicitly passing the context parameter to the API
        url = f"{self.base_url}/search/search/experts/predict/{partial_query}"
        
        # Include context parameter in the request if provided
        params = {}
        if context_type:
            params["context"] = context_type
            logger.info(f"Getting predictions with context: {context_type}")
            
        try:
            # Make the API call with the context parameter
            response = requests.get(url, headers=self.headers, params=params)
            
            # Log the full request URL for debugging
            logger.info(f"Prediction request URL: {response.request.url}")
            
            if response.status_code == 200:
                data = response.json()
                # Log the response to see what's being returned
                logger.info(f"Got predictions: {data.get('predictions', [])}")
                return data.get("predictions", [])
            else:
                logger.error(f"Error getting predictions: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Exception getting predictions: {str(e)}")
            return []
    
    def normal_search(self, query: str) -> Dict[str, Any]:
        """Perform a normal search."""
        url = f"{self.base_url}/search/search/experts/search/{query}"
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error in normal search: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Exception in normal search: {str(e)}")
            return {}
    
    def advanced_search(self, name: Optional[str] = None, theme: Optional[str] = None, 
                        designation: Optional[str] = None) -> Dict[str, Any]:
        """Perform an advanced search."""
        url = f"{self.base_url}/search/search/advanced_search"
        
        params = {}
        if name:
            params["name"] = name
        if theme:
            params["theme"] = theme
        if designation:
            params["designation"] = designation
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error in advanced search: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Exception in advanced search: {str(e)}")
            return {}
    
    def display_search_results(self, results: Dict[str, Any]):
        """Display search results."""
        if not results:
            print("No results found.")
            return
            
        total = results.get("total_results", 0)
        experts = results.get("experts", [])
        
        print(f"\nFound {total} experts:")
        
        for i, expert in enumerate(experts, 1):
            name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}"
            designation = expert.get('designation', '')
            score = expert.get('score', 0.0)
            
            print(f"{i}. {name} - {designation} (Score: {score:.2f})")
            
            # Show expertise (limited)
            expertise = expert.get('knowledge_expertise', [])
            if expertise:
                if isinstance(expertise, list):
                    expertise_text = ", ".join(expertise[:3])
                    if len(expertise) > 3:
                        expertise_text += f" (+ {len(expertise) - 3} more)"
                    print(f"   Expertise: {expertise_text}")
                elif isinstance(expertise, str):
                    print(f"   Expertise: {expertise}")
                
            print()
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def run_interactive(self):
        """Run the interactive search experience."""
        while True:
            self.clear_screen()
            print("==== APHRC Interactive Search Tester ====\n")
            print("1. Normal Search")
            print("2. Advanced Search")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == "1":
                self.run_normal_search()
            elif choice == "2":
                self.run_advanced_search()
            elif choice == "3":
                print("Exiting. Goodbye!")
                break
            else:
                print("Invalid choice. Press Enter to continue...")
                input()
    
    def run_normal_search(self):
        """Run a normal search with predictive suggestions."""
        self.clear_screen()
        print("==== Normal Search Mode ====\n")
        print("Type your search query (then press Enter to see predictions)")
        print("Press Ctrl+C to go back to the main menu\n")
        
        try:
            query = ""
            while True:
                query = input("Search: ")
                if not query:
                    continue
                    
                # Show predictions - pass None for context to get normal predictions
                print("\nGetting predictions...")
                predictions = self.get_predictions(query, context_type=None)
                
                if predictions:
                    print("\nPredictions:")
                    for i, pred in enumerate(predictions[:MAX_SUGGESTIONS], 1):
                        print(f"{i}. {pred}")
                    
                    # Ask if user wants to select a prediction
                    selection = input("\nSelect a prediction (1-8) or press Enter to search with your query: ")
                    
                    if selection.strip() and selection.isdigit():
                        idx = int(selection) - 1
                        if 0 <= idx < len(predictions) and idx < MAX_SUGGESTIONS:
                            query = predictions[idx]
                            print(f"Selected: {query}")
                else:
                    print("No predictions found.")
                
                # Perform the search
                print(f"\nSearching for: {query}")
                results = self.normal_search(query)
                
                # Display results
                self.display_search_results(results)
                
                # Ask if user wants to do another search
                another = input("\nDo another search? (y/n): ")
                if another.lower() != 'y':
                    break
                    
                self.clear_screen()
                print("==== Normal Search Mode ====\n")
                
        except KeyboardInterrupt:
            print("\nSearch interrupted. Returning to the main menu.")
            time.sleep(1)
    
    def run_advanced_search(self):
        """Run an advanced search with choice of search type."""
        self.clear_screen()
        print("==== Advanced Search Mode ====\n")
        print("Which fields would you like to search?")
        print("1. Name")
        print("2. Theme")
        print("3. Designation")
        print("4. Multiple Fields")
        print("5. Go Back")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "5":
            return
        
        name_query = None
        theme_query = None
        designation_query = None
        
        if choice == "1" or choice == "4":
            name_query = self.get_search_field("name")
        
        if choice == "2" or choice == "4":
            theme_query = self.get_search_field("theme")
        
        if choice == "3" or choice == "4":
            designation_query = self.get_search_field("designation")
        
        # Perform the search
        self.clear_screen()
        print("Performing advanced search with parameters:")
        if name_query:
            print(f"- Name: {name_query}")
        if theme_query:
            print(f"- Theme: {theme_query}")
        if designation_query:
            print(f"- Designation: {designation_query}")
        
        results = self.advanced_search(name_query, theme_query, designation_query)
        
        # Display results
        self.display_search_results(results)
        
        print("\nPress Enter to continue...")
        input()
    
    def get_search_field(self, field_type: str) -> Optional[str]:
        """Get a search field value with context-aware predictive suggestions."""
        self.clear_screen()
        print(f"==== Enter {field_type.title()} ====\n")
        print(f"Type your {field_type} search query and press Enter")
        print("Press Ctrl+C to cancel\n")
        
        try:
            partial_query = ""
            while True:
                partial_query = input(f"{field_type.title()} query: ")
                if not partial_query:
                    continue
                
                # Get context-specific predictions - explicitly pass the field_type as context
                print("\nGetting predictions...")
                predictions = self.get_predictions(partial_query, context_type=field_type)
                
                if predictions:
                    print("\nPredictions:")
                    for i, pred in enumerate(predictions[:MAX_SUGGESTIONS], 1):
                        print(f"{i}. {pred}")
                    
                    # Ask if user wants to select a prediction
                    selection = input("\nSelect a prediction (1-8) or press Enter to use your query: ")
                    
                    if selection.strip() and selection.isdigit():
                        idx = int(selection) - 1
                        if 0 <= idx < len(predictions) and idx < MAX_SUGGESTIONS:
                            partial_query = predictions[idx]
                            print(f"Selected: {partial_query}")
                    break
                else:
                    print("No predictions found.")
                    use_query = input("\nUse this query anyway? (y/n): ")
                    if use_query.lower() == 'y':
                        break
            
            return partial_query if partial_query else None
            
        except KeyboardInterrupt:
            print("\nInput canceled.")
            time.sleep(1)
            return None

# Main execution
if __name__ == "__main__":
    tester = InteractiveSearchTester(BASE_URL, USER_ID)
    tester.run_interactive()