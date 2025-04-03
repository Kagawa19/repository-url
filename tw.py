import os
import google.generativeai as genai
from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion

# Configure Gemini API
try:
    # Ensure you replace this with your actual Gemini API key
    API_KEY = os.getenv('GEMINI_API_KEY')
    if not API_KEY:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")
    
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit(1)

class GeminiAutocomplete(Completer):
    def __init__(self, model_name='gemini-1.5-flash'):
        """
        Initialize the Gemini Autocomplete Completer
        
        :param model_name: Gemini model to use for generating suggestions
        """
        self.model = genai.GenerativeModel(model_name)
    
    def get_completions(self, document, complete_event):
        """
        Generate autocomplete suggestions based on current input
        
        :param document: Current input document
        :param complete_event: Completion event details
        :yield: Completion suggestions
        """
        user_input = document.text
        
        if not user_input or len(user_input) < 2:
            return
        
        try:
            # Generate suggestions based on the current input
            prompt_text = f"Provide 5-10 autocomplete suggestions for a search query starting with: {user_input}"
            
            # Configure generation to focus on suggestions
            generation_config = {
                'temperature': 0.7,  # Slightly creative but not too random
                'max_output_tokens': 100  # Limit output length
            }
            
            response = self.model.generate_content(
                prompt_text, 
                generation_config=generation_config
            )
            
            # Parse and clean suggestions
            suggestions = [
                suggestion.strip() 
                for suggestion in response.text.split('\n') 
                if suggestion.strip() and user_input.lower() in suggestion.lower()
            ]
            
            # Yield completions
            for suggestion in suggestions:
                yield Completion(suggestion, start_position=-len(user_input))
        
        except Exception as e:
            print(f"Error generating suggestions: {e}")

def main():
    """
    Main function to run the predictive search interface
    """
    print("Gemini Predictive Search")
    print("Start typing to get search suggestions (Ctrl+C to exit)")
    
    # Create the Gemini Autocomplete completer
    completer = GeminiAutocomplete()
    
    try:
        while True:
            # Prompt with autocomplete suggestions
            user_input = prompt('Search: ', completer=completer)
            
            # Option to exit
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            # Optional: Perform search or show selected suggestion
            print(f"You selected: {user_input}")
    
    except KeyboardInterrupt:
        print("\nExiting predictive search...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()