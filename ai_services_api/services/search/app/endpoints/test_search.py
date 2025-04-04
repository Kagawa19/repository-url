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
        
    def get_suggestions(self, partial_query: str) -> Dict[str, Any]:
        """Get search suggestions for a partial query."""
        # Updated endpoint to match your API structure
        url = f"{self.base_url}/search/search/experts/predict/{partial_query}"
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting suggestions: {response.status_code}")
                logger.error(response.text)
                return {}
        except Exception as e:
            logger.error(f"Exception getting suggestions: {str(e)}")
            return {}
            
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
        
    def display_refinements(self, refinements: Dict[str, Any]):
        """Display search refinements."""
        if not refinements:
            print("No refinements available.")
            return
            
        # Display related queries
        related_queries = refinements.get("related_queries", [])
        if related_queries:
            print("\nRelated Queries:")
            for i, query in enumerate(related_queries, 1):
                print(f"  {i}. {query}")
        
        # Display expertise areas
        expertise_areas = refinements.get("expertise_areas", [])
        if expertise_areas:
            print("\nExpertise Areas:")
            for i, area in enumerate(expertise_areas, 1):
                print(f"  {i}. {area}")
        
        # Display filters
        filters = refinements.get("filters", [])
        if filters:
            print("\nFilters:")
            for filter_group in filters:
                label = filter_group.get("label", "Unknown")
                values = filter_group.get("values", [])
                print(f"  {label}:")
                for i, value in enumerate(values, 1):
                    print(f"    {i}. {value}")
            
    def display_suggestions(self, response: Dict[str, Any]):
        """Display suggestions with numbering."""
        suggestions = response.get("predictions", [])
        
        if not suggestions:
            print("No suggestions found.")
            return
            
        print("\nSuggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            confidence = response.get("confidence_scores", [])[i-1] if i <= len(response.get("confidence_scores", [])) else 0.0
            print(f"{i}. {suggestion} ({confidence:.2f})")
        
        # Display refinements if available
        refinements = response.get("refinements")
        if refinements:
            self.display_refinements(refinements)
            
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

    def advanced_search_experts(self, query: str, search_type: str, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """
        Perform an advanced search for experts with specific search type.
        
        Args:
            query: Search term
            search_type: Type of search ('name', 'theme', or 'designation')
            page: Page number for pagination
            page_size: Number of results per page
        
        Returns:
            Dictionary of search results
        """
        # Updated endpoint to match your API structure for advanced search
        url = f"{self.base_url}/search/search/experts/advanced_search/{query}"
        params = {
            "search_type": search_type,
            "page": page,
            "page_size": page_size
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error in advanced search: {response.status_code}")
                logger.error(response.text)
                return {}
        except Exception as e:
            logger.error(f"Exception in advanced search: {str(e)}")
            return {}

    def advanced_search_demo(self, initial_query: str = "health"):
        """
        Demonstration of advanced search across different search types.
        """
        print("\n=== APHRC Advanced Search Demo ===\n")
        
        # List of search types to demonstrate
        search_types = ["name", "theme", "designation"]
        
        for search_type in search_types:
            print(f"\nPerforming advanced search with query: '{initial_query}', Type: {search_type}")
            
            # Perform advanced search
            search_start = time.time()
            results = self.advanced_search_experts(initial_query, search_type)
            search_time = time.time() - search_start
            
            print(f"Advanced Search completed in {search_time:.2f} seconds")
            
            # Display results
            if results:
                print(f"\nAdvanced Search Results (Type: {search_type}):")
                self.display_search_results(results)
                
                # Display refinements if available
                refinements = results.get("refinements")
                if refinements:
                    print("\nRefinements:")
                    self.display_refinements(refinements)
            else:
                print(f"No results found for {search_type} search.")
            
            # Add a small pause between searches
            time.sleep(1)

# Modify the main function to include advanced search demo option
def main():
    """Main function for testing."""
    tester = SearchTester(BASE_URL, USER_ID)
    
    # Get mode from environment or ask user
    mode = os.environ.get("SEARCH_TEST_MODE", "").lower()
    
    if not mode:
        print("\nSelect test mode:")
        print("1. Interactive search")
        print("2. Demo search")
        print("3. Advanced search demo")
        choice = input("Enter choice (1, 2, or 3): ")
        mode = "interactive" if choice == "1" else "demo" if choice == "2" else "advanced"
    
    if mode == "interactive":
        tester.interactive_search()
    elif mode == "demo":
        query = os.environ.get("SEARCH_DEMO_QUERY", "health")
        tester.demo_search(query)
    else:
        query = os.environ.get("SEARCH_DEMO_QUERY", "health")
        tester.advanced_search_demo(query)

