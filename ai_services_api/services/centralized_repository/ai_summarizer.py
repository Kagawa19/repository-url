import os
import logging
import google.generativeai as genai
from typing import Optional, List, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
import json

# Simple logging setup like in the second version
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self):
        """Initialize the TextSummarizer with Gemini model."""
        self.model = self._setup_gemini()
        self.content_types = ["articles", "publications", "blogs", "multimedia"]
        
        # Property to store dynamically generated fields and their subfields
        self.corpus_fields = {}
        self.has_analyzed_corpus = False
        
        logger.info("TextSummarizer initialized successfully")

    def _setup_gemini(self):
        """Set up and configure the Gemini model."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set")
            
            # Simple configuration without specifying API version
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            logger.info("Gemini model setup completed")
            return model
            
        except Exception as e:
            logger.error(f"Error setting up Gemini model: {e}")
            raise

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=7, max=14)
    )
    def summarize(self, title: str, abstract: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate a summary of the title and abstract using Gemini and classify the content type.
        """
        try:
            if not title:
                logger.error("Title is required for summarization")
                return ("Cannot generate summary: title is missing", None)

            if not abstract or abstract.strip() == "N/A":
                abstract = title  # Use title as content if no abstract available
            
            prompt = self._create_combined_prompt(title, abstract)
            
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            if not result:
                logger.warning("Generated content is empty")
                return ("Failed to generate meaningful content", None)
            
            summary, content_type = self._parse_response(result)
            cleaned_summary = self._clean_summary(summary)
            logger.info(f"Successfully generated content for: {title[:100]}...")
            
            return (cleaned_summary, content_type)
        
        except Exception as e:
            logger.error(f"Error in content generation: {e}")
            return ("Failed to generate content due to technical issues", None)
    
    def analyze_content_corpus(self, publications: List[Dict]) -> Dict[str, List[str]]:
        """
        Comprehensive corpus analysis for publication classification.
        """
        # Prepare publications data for analysis
        publication_data = [{
            'title': p.get('title', ''),
            'abstract': p.get('abstract', '')[:200],  # Truncate abstract
            'domains': p.get('domains', [])
        } for p in publications[:50]]  # Sample first 50 for analysis
        
        # Prompt for generating fields with constraint
        corpus_prompt = f"""
        Analyze this collection of {len(publication_data)} academic publications and identify exactly 5 natural groupings:

        Publications:
        {json.dumps(publication_data, indent=2)}

        Task:
        1. Identify EXACTLY 5 major thematic fields that emerge from this content
        2. Choose fields that are distinct but collectively cover the entire corpus
        3. For each field, identify 3-5 common specialized subfields
        4. Consider interdisciplinary areas and emerging fields

        Return the classification structure as:
        FIELDS:
        [Field 1]
        - [Subfield 1.1]
        - [Subfield 1.2]
        - [Subfield 1.3]
        [Field 2]
        - [Subfield 2.1]
        - [Subfield 2.2]
        - [Subfield 2.3]
        ... and so on for exactly 5 fields
        """
        
        try:
            # Generate classification using Gemini
            response = self.model.generate_content(corpus_prompt)
            fields = {}
            current_field = None
            
            # Parse the response
            for line in response.text.strip().split('\n'):
                if line.startswith('-'):
                    if current_field:
                        subfield = line.replace('-', '').strip()
                        if current_field in fields:
                            fields[current_field].append(subfield)
                        else:
                            fields[current_field] = [subfield]
                else:
                    potential_field = line.strip()
                    if potential_field and not potential_field.startswith('FIELDS:'):
                        current_field = potential_field
                        if current_field not in fields:
                            fields[current_field] = []
            
            # Limit to exactly 5 fields if we got more
            if len(fields) > 5:
                fields = {k: fields[k] for k in list(fields.keys())[:5]}
            
            # Store and return fields
            self.corpus_fields = fields
            self.has_analyzed_corpus = True
            
            logger.info(f"Corpus analysis complete. Identified {len(fields)} fields: {', '.join(fields.keys())}")
            
            return fields
        
        except Exception as e:
            logger.error(f"Error in corpus analysis: {e}")
            return {}

    def classify_field_and_subfield(self, title: str, abstract: str, domains: List[str]) -> Tuple[str, str]:
        """
        Classify content using the dynamically generated fields from corpus analysis.
        """
        # Check if we have analyzed the corpus
        if self.has_analyzed_corpus and self.corpus_fields:
            # Format the fields for the prompt
            fields_list = list(self.corpus_fields.keys())
            fields_formatted = "\n".join([f"{i+1}. {field}" for i, field in enumerate(fields_list)])
            
            prompt = f"""
            Analyze this academic content and classify it into one of the following fields 
            that were derived from corpus analysis:
            
            {fields_formatted}
            
            Title: {title}
            Abstract: {abstract}
            Domains: {', '.join(domains)}
            
            Instructions:
            1. Choose ONE of the fields listed above that best matches this content.
            2. Then, determine a specific subfield that best describes the specialized area within that field.
            3. The subfield should be specific and meaningful.
            
            Return ONLY:
            FIELD: [one of the fields listed above]
            SUBFIELD: [specific subfield within that field]
            """
        else:
            # Fall back to dynamic classification
            prompt = f"""
            Analyze this academic content and create a natural field classification:

            Title: {title}
            Abstract: {abstract}
            Domains: {', '.join(domains)}

            Instructions:
            1. First, determine the broad field this content belongs to, considering the overall theme and academic discipline.
            2. Then, determine a more specific subfield that best describes the specialized area within that field.
            3. The classification should emerge naturally from the content rather than fitting into predefined categories.
            4. Your field should be broad enough to group similar content but specific enough to be meaningful.
            5. Your subfield should capture the specific focus area within that field.

            Return ONLY:
            FIELD: [naturally derived field]
            SUBFIELD: [specific subfield within that field]
            """

        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            field = None
            subfield = None
            
            for line in result.split('\n'):
                if line.startswith('FIELD:'):
                    field = line.replace('FIELD:', '').strip()
                elif line.startswith('SUBFIELD:'):
                    subfield = line.replace('SUBFIELD:', '').strip()
            
            if field and subfield:
                return field, subfield
            else:
                return "Unclassified", "General"
                
        except Exception as e:
            logger.error(f"Error in field classification: {e}")
            return "Unclassified", "General"

    def _create_combined_prompt(self, title: str, abstract: str) -> str:
        """Create a prompt for both summarization and content type classification."""
        return f"""
        Please analyze the following content and provide:
        1. A concise summary
        2. Classification of the content type (strictly choose one: articles, publications, blogs, multimedia)
        
        Title: {title}
        
        Content: {abstract}
        
        Instructions:
        1. Provide a clear and concise summary in 2-3 sentences
        2. Focus on the main points and implications
        3. Use appropriate language for the content type
        4. Keep the summary under 200 words
        5. Retain technical terms and key concepts
        6. Begin directly with the summary, do not include phrases like "This paper" or "This content"
        7. After the summary, on a new line, write "CONTENT_TYPE:" followed by one of: articles, publications, blogs, multimedia
        
        Example format:
        [Your summary here]
        CONTENT_TYPE: publications
        """

    def _create_title_only_prompt(self, title: str) -> str:
        """Create a prompt for generating a brief description and assigning a content genre."""
        return f"""
        Please analyze the following title and determine:
        1. A brief description of the content.
        2. A suitable **genre tag** (maximum of two words) that best represents the subject matter, selected from a predefined list.

        **Title:** {title}

        **Instructions:**
        - Provide a concise, single-sentence summary of what this content likely discusses.
        - Use phrases like "This content appears to discuss..." or "This work likely explores..."
        - Keep the description under 50 words.
        - Assign a **genre tag** (strictly one or two words) that best fits the content, choosing from the following:
        **"Reproductive Health," "Public Health," "Education," "Policy," "Research," "Nutrition," "Urbanization," "Gender Equity," "Climate Change," "Demography"**.
        - If the title does not clearly fit one of these, return "Uncategorized."

        **Example format:**
        [Your description here]  
        GENRE: [One of the 10 predefined genre tags]
        """

    def _parse_response(self, response: str) -> Tuple[str, Optional[str]]:
        """Parse the response to separate summary and content type."""
        try:
            parts = response.split('CONTENT_TYPE:', 1)
            summary = parts[0].strip()
            
            content_type = None
            if len(parts) > 1:
                content_type = parts[1].strip().lower()
                if content_type not in self.content_types:
                    logger.warning(f"Invalid content type detected: {content_type}")
                    content_type = None
            
            return summary, content_type
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return response, None

    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary."""
        try:
            # Basic cleaning
            cleaned = summary.strip()
            cleaned = ' '.join(cleaned.split())  # Normalize whitespace
            
            # Remove common prefixes if present
            prefixes = [
                'Summary:', 
                'Here is a summary:', 
                'The summary is:', 
                'Here is a concise summary:',
                'This paper',
                'This research',
                'This study'
            ]
            
            lower_cleaned = cleaned.lower()
            for prefix in prefixes:
                if lower_cleaned.startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            
            # Ensure the summary starts with a capital letter
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]
            
            # Add a period at the end if missing
            if cleaned and cleaned[-1] not in ['.', '!', '?']:
                cleaned += '.'
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning summary: {e}")
            return summary

    def __del__(self):
        """Cleanup any resources."""
        try:
            # No specific cleanup needed
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")