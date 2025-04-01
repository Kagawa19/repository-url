import requests
import json
import time
import logging
import readline
from typing import List, Dict, Any, Optional
import os

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

class SearchTester:
    """Interactive Google-style search tester."""
    
    def __init__(self, base_url: str, user_id: str):
        self.base_url = base_url
        self.user_id = user_id
        self.headers = {
            "X-User-ID": user_id,
            "Content-Type": "application/json"
        }
        
    def get_suggestions(self, partial_query: str) -> List[Dict[str, Any]]:
        """Get search suggestions for a partial query."""
        # Fixed endpoint to match your API structure
        url = f"{self.base_url}/search/search/experts/suggest/{partial_query}"
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("suggestions", [])
            else:
                logger.error(f"Error getting suggestions: {response.status_code}")
                logger.error(response.text)
                return []
        except Exception as e:
            logger.error(f"Exception getting suggestions: {str(e)}")
            return []
            
    def record_suggestion_click(self, suggestion: str, resulting_query: str, source_type: Optional[str] = None):
        """Record that a suggestion was clicked."""
        # Fixed endpoint to match your API structure
        url = f"{self.base_url}/search/search/experts/suggestion/click"
        params = {
            "suggestion": suggestion,
            "resulting_query": resulting_query
        }
        
        if source_type:
            params["source_type"] = source_type
            
        try:
            response = requests.post(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                logger.info("Suggestion click recorded successfully")
            else:
                logger.error(f"Error recording suggestion click: {response.status_code}")
                logger.error(response.text)
        except Exception as e:
            logger.error(f"Exception recording suggestion click: {str(e)}")
            
    def search_experts(self, query: str, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """Search for experts using the query."""
        # Fixed endpoint to match your API structure
        url = f"{self.base_url}/search/search/experts/search/{query}"
        params = {
            "page": page,
            "page_size": page_size
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error searching experts: {response.status_code}")
                logger.error(response.text)
                return {}
        except Exception as e:
            logger.error(f"Exception searching experts: {str(e)}")
            return {}
            
    def display_suggestions(self, suggestions: List[Dict[str, Any]]):
        """Display suggestions with numbering."""
        if not suggestions:
            print("No suggestions found.")
            return
            
        print("\nSuggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            display = suggestion.get("display_text") or suggestion.get("text")
            # Remove HTML tags
            display = display.replace("<b>", "").replace("</b>", "")
            source = suggestion.get("type", "unknown")
            confidence = suggestion.get("confidence", 0.0)
            
            print(f"{i}. {display} [{source}] ({confidence:.2f})")
            
    def display_search_results(self, results: Dict[str, Any]):
        """Display search results."""
        if not results:
            print("No results found.")
            return
            
        total = results.get("total_results", 0)
        experts = results.get("experts", [])
        pagination = results.get("pagination", {})
        
        print(f"\nFound {total} experts (Page {pagination.get('page', 1)} of {pagination.get('total_pages', 1)}):")
        
        for i, expert in enumerate(experts, 1):
            name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}"
            designation = expert.get('designation', '')
            score = expert.get('score', 0.0)
            
            print(f"{i}. {name} - {designation} (Score: {score:.2f})")
            
            # Show expertise (limited)
            expertise = expert.get('knowledge_expertise', [])
            if expertise:
                expertise_text = ", ".join(expertise[:3])
                if len(expertise) > 3:
                    expertise_text += f" (+ {len(expertise) - 3} more)"
                print(f"   Expertise: {expertise_text}")
                
            print()
            
    def interactive_search(self):
        """Run interactive search with suggestions and results."""
        print("\n=== APHRC Expert Search Tester ===\n")
        print("Type a search query (type slowly to see suggestions)")
        print("Enter 'q' to quit")
        
        while True:
            try:
                # Get input with prompt and readline for better UX
                query_input = input("\nSearch: ")
                
                if query_input.lower() == 'q':
                    print("Goodbye!")
                    break
                    
                if not query_input.strip():
                    continue
                
                # Show suggestions as user is typing (simulate with partial query)
                partial_query = query_input
                suggestions = self.get_suggestions(partial_query)
                self.display_suggestions(suggestions)
                
                # Ask if user wants to select a suggestion
                if suggestions:
                    selection = input("\nSelect a suggestion (1-8) or press Enter to use your query: ")
                    
                    if selection.strip() and selection.isdigit():
                        idx = int(selection) - 1
                        if 0 <= idx < len(suggestions):
                            # Use the selected suggestion
                            selected = suggestions[idx]
                            final_query = selected.get("text")
                            print(f"Selected: {final_query}")
                            
                            # Record the click
                            self.record_suggestion_click(
                                final_query,
                                final_query,
                                selected.get("type")
                            )
                        else:
                            # Use original query
                            final_query = query_input
                            print(f"Invalid selection. Using: {final_query}")
                    else:
                        # Use original query
                        final_query = query_input
                else:
                    final_query = query_input
                
                # Now perform the search with the final query
                print(f"\nSearching for: {final_query}")
                search_start = time.time()
                results = self.search_experts(final_query)
                search_time = time.time() - search_start
                
                print(f"Search completed in {search_time:.2f} seconds")
                self.display_search_results(results)
                
            except KeyboardInterrupt:
                print("\nSearch interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error during interactive search: {str(e)}")
                print(f"Error: {str(e)}")
            
    def demo_search(self, initial_query: str = "health"):
        """Run a demonstration search with predefined steps."""
        print("\n=== APHRC Expert Search Demo ===\n")
        
        print(f"Starting search demo with query: '{initial_query}'")
        
        # 1. Type query character by character to show suggestions
        partial_query = ""
        for char in initial_query:
            partial_query += char
            print(f"\nTyping: {partial_query}")
            
            suggestions = self.get_suggestions(partial_query)
            self.display_suggestions(suggestions)
            time.sleep(0.5)  # Simulate typing delay
        
        # 2. Pick a suggestion
        print("\nFinal suggestions:")
        suggestions = self.get_suggestions(initial_query)
        self.display_suggestions(suggestions)
        
        if suggestions:
            # Pick the first suggestion
            selected = suggestions[0]
            final_query = selected.get("text")
            print(f"\nSelected suggestion: {final_query}")
            
            # Record the click
            self.record_suggestion_click(
                final_query,
                final_query,
                selected.get("type")
            )
            
            # 3. Perform the search
            print(f"\nSearching for: {final_query}")
            search_start = time.time()
            results = self.search_experts(final_query)
            search_time = time.time() - search_start
            
            print(f"Search completed in {search_time:.2f} seconds")
            self.display_search_results(results)
        else:
            print("No suggestions found. Demo cannot continue.")

def main():
    """Main function for testing."""
    tester = SearchTester(BASE_URL, USER_ID)
    
    # Get mode from environment or ask user
    mode = os.environ.get("SEARCH_TEST_MODE", "").lower()
    
    if not mode:
        print("\nSelect test mode:")
        print("1. Interactive search")
        print("2. Demo search")
        choice = input("Enter choice (1 or 2): ")
        mode = "interactive" if choice == "1" else "demo"
    
    if mode == "interactive":
        tester.interactive_search()
    else:
        query = os.environ.get("SEARCH_DEMO_QUERY", "health")
        tester.demo_search(query)

if __name__ == "__main__":
    main()