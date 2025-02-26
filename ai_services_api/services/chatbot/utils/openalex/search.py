# OpenAlex Search Retriever

import requests
import os

class OpenAlexSearch():
    """
    OpenAlex Search Retriever
    """
    def __init__(self, query):
        """
        Initializes the OpenAlexSearch object
        Args:
            query: The search query (topic of interest)
        """
        self.query = query

    def search(self, max_results=10):
        """
        Searches the OpenAlex API for works related to the query
        Args:
            max_results: The maximum number of results to retrieve
        Returns:
            A list of dictionary objects containing information about each work
        """
        print(f"Searching OpenAlex for '{self.query}'...")
        url = "https://api.openalex.org/works"
        params = {
            'filter': f'title.search:{self.query}',
            'per-page': max_results
        }

        resp = requests.get(url, params=params)

        if resp.status_code != 200:
            print(f"Failed to retrieve data: HTTP {resp.status_code}")
            return None

        try:
            search_results = resp.json()
        except Exception as e:
            print(f"Failed to parse the response: {e}")
            return None

        results = search_results.get("results", [])
        search_response = []

        for result in results:
            # Collect all keyword and concepts strings
            keywords_list = [kw['keyword'] for kw in result.get('keywords', [])]
            concepts_list = [cl['display_name'] for cl in result.get('concepts', [])]
            
            # Assembling the content/body part
            content = {
                "title": result['display_name'],
                "year": result.get('publication_year', "No year provided"),
                "author": result['authorships'][0]['raw_author_name'] if result.get('authorships') else "No author provided",
                "concepts": concepts_list,
                "keywords": keywords_list
            }
            
            # Final transformation
            transformed_result = {
                "href": result.get('doi', "No DOI provided"),
                "body": content
            }
            
            search_response.append(transformed_result)

        return search_response