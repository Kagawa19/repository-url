from typing import List, Dict, Tuple, Any, Optional, Set
from ai_services_api.services.centralized_repository.expert_matching.models import Expert, Resource
from ai_services_api.services.centralized_repository.expert_matching.logger import Logger
from ai_services_api.services.centralized_repository.database_setup import get_db_cursor
import json
import re
from Levenshtein import distance as levenshtein_distance
from difflib import SequenceMatcher
import numpy as np

class EnhancedMatcher:
    """Enhanced matcher that combines name matching with expertise similarity"""
    
    def __init__(self):
        self.name_cache = {}
        self.logger = Logger(__name__)
        self.name_similarity_threshold = 0.8  # Threshold for fuzzy name matching
        self.expertise_similarity_threshold = 0.3  # Threshold for expertise similarity
        self.aphrc_expert_ids = set() 


    def _normalize_name(self, name: str) -> str:
        """Normalize author name for comparison"""
        if not name or not isinstance(name, str):
            return ""
        
        # Convert to lowercase and remove extra spaces
        normalized = ' '.join(name.lower().split())
        
        # Remove common suffixes and prefixes
        prefixes = ['dr.', 'dr ', 'prof.', 'prof ', 'professor ', 'mr.', 'mr ', 'mrs.', 'mrs ', 'ms.', 'ms ']
        suffixes = [' phd', ' md', ' jr', ' sr', ' jr.', ' sr.', ' ii', ' iii', ' iv']
        
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        return normalized

    def _get_name_parts(self, full_name: str) -> Tuple[str, str]:
        """Split a name into first and last name"""
        if not full_name:
            return ("", "")
            
        parts = full_name.split()
        if len(parts) == 1:
            return ("", parts[0])  # Only last name available
        elif len(parts) == 2:
            return (parts[0], parts[1])  # Standard first+last
        else:
            # Multiple parts - assume first part is first name, last part is last name
            return (parts[0], parts[-1])

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names using multiple metrics.
        Returns a score between 0 and 1, where 1 is exact match.
        """
        name1 = self._normalize_name(name1)
        name2 = self._normalize_name(name2)
        
        if not name1 or not name2:
            return 0.0
            
        # Exact match check
        if name1 == name2:
            return 1.0
            
        # Split into parts
        first1, last1 = self._get_name_parts(name1)
        first2, last2 = self._get_name_parts(name2)
        
        # Last name exact match is a strong signal
        if last1 and last2 and last1 == last2:
            # If last names match exactly, check first names
            if first1 and first2:
                # Check for first initial match when full first names don't match
                if first1[0] == first2[0] and (len(first1) == 1 or len(first2) == 1):
                    return 0.9  # First initial matches
                
                # Calculate first name similarity
                first_similarity = SequenceMatcher(None, first1, first2).ratio()
                # Weight heavily toward last name but consider first name
                return 0.7 + (0.3 * first_similarity)
            return 0.8  # Last name matches but can't compare first names
            
        # Calculate overall name similarity
        overall_similarity = SequenceMatcher(None, name1, name2).ratio()
        
        # Calculate Levenshtein distance and normalize
        max_len = max(len(name1), len(name2))
        if max_len > 0:
            levenshtein_similarity = 1 - (levenshtein_distance(name1, name2) / max_len)
        else:
            levenshtein_similarity = 0
            
        # Combine metrics (equal weight)
        combined_similarity = (overall_similarity + levenshtein_similarity) / 2
        
        return combined_similarity

    def _get_expert_expertise(self, expert: Any) -> Set[str]:
        """Extract expertise keywords from expert data"""
        expertise_terms = set()
        
        if isinstance(expert, tuple):
            # Try to extract expertise from knowledge_expertise field if available
            if len(expert) > 3 and expert[3]:  # Position 3 might hold knowledge_expertise
                try:
                    if isinstance(expert[3], str):
                        expertise_data = json.loads(expert[3])
                    else:
                        expertise_data = expert[3]
                        
                    if isinstance(expertise_data, dict):
                        # Extract all values
                        for values in expertise_data.values():
                            if isinstance(values, list):
                                for item in values:
                                    expertise_terms.add(str(item).lower())
                            elif values:
                                expertise_terms.add(str(values).lower())
                except (json.JSONDecodeError, TypeError):
                    pass
        
            # Try to extract domains if available
            if len(expert) > 4 and expert[4]:  # Position 4 might hold domains
                if isinstance(expert[4], list):
                    for domain in expert[4]:
                        expertise_terms.add(str(domain).lower())
                elif isinstance(expert[4], str):
                    try:
                        domains = json.loads(expert[4])
                        if isinstance(domains, list):
                            for domain in domains:
                                expertise_terms.add(str(domain).lower())
                    except json.JSONDecodeError:
                        expertise_terms.add(expert[4].lower())
        
        return expertise_terms

    def _get_resource_keywords(self, resource: Any) -> Set[str]:
        """Extract keywords from resource data"""
        keywords = set()
        
        if isinstance(resource, tuple):
            # Resource is assumed to be (id, authors, [other fields])
            # Add additional fields if available
            for i in range(2, len(resource)):
                field = resource[i]
                if field:
                    # Try to extract keywords from text fields
                    if isinstance(field, str):
                        # Add individual words for text fields
                        words = re.findall(r'\b\w{4,}\b', field.lower())
                        keywords.update(words)
                    # Handle list fields
                    elif isinstance(field, list):
                        for item in field:
                            keywords.add(str(item).lower())
                    # Try to handle JSON fields
                    elif isinstance(field, dict):
                        for value in field.values():
                            if isinstance(value, str):
                                keywords.add(value.lower())
        
        return keywords

    def _calculate_expertise_similarity(self, expert_expertise: Set[str], resource_keywords: Set[str]) -> float:
        """Calculate similarity between expert expertise and resource keywords"""
        if not expert_expertise or not resource_keywords:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = expert_expertise.intersection(resource_keywords)
        union = expert_expertise.union(resource_keywords)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)

    def match_experts_to_resources(self, experts: List[Any], resources: List[Any]) -> Dict[int, List[Tuple[int, float, bool]]]:
        """
        Match experts to resources with confidence score and source flag.
        Returns: {expert_id: [(resource_id, confidence, is_aphrc_publication)]}
        """
        # Cache APHRC experts and their normalized names
        self.aphrc_expert_ids = {expert[0] for expert in experts}  # expert[0] = expert_id
        self.name_cache = {
            self._normalize_name(f"{expert[1]} {expert[2]}"): expert[0] 
            for expert in experts
        }

        # Create expert_lookup for expertise matching
        expert_lookup = {
            expert[0]: {
                'expertise': self._get_expert_expertise(expert)
            } for expert in experts
        }

        matches = {}

        for resource in resources:
            resource_id = resource[0]
            authors = resource[1]

            # Parse authors (handle JSON strings or lists)
            if isinstance(authors, str):
                try:
                    author_list = json.loads(authors)
                except json.JSONDecodeError:
                    author_list = [authors]
            else:
                author_list = authors

            # Identify APHRC and global authors
            aphrc_authors, global_authors = self._identify_aphrc_authors(author_list)

            # Match APHRC authors
            for author in aphrc_authors:
                normalized = self._normalize_name(author)
                expert_id = self.name_cache.get(normalized)
                if expert_id is not None:
                    matches.setdefault(expert_id, []).append((resource_id, 0.9, True))

            # Match global authors based on expertise similarity
            if global_authors:
                resource_keywords = self._get_resource_keywords(resource)
                for expert_id, expert_data in expert_lookup.items():
                    similarity = self._calculate_expertise_similarity(
                        expert_data['expertise'], resource_keywords)
                    if similarity > 0.3:
                        matches.setdefault(expert_id, []).append(
                            (resource_id, 0.6 * similarity, False)
                        )

        return matches

    def link_matched_experts_to_db(self) -> None:
        """Link matched experts to resources in database with confidence scores"""
        try:
            with get_db_cursor() as (cur, conn):
                # Get all experts with additional fields for better matching
                cur.execute("""
                    SELECT id, first_name, last_name, knowledge_expertise, domains
                    FROM experts_expert
                    WHERE is_active = TRUE
                """)
                experts = cur.fetchall()

                # Get all resources with additional context fields
                cur.execute("""
                    SELECT id, authors, title, abstract, summary, domains, topics
                    FROM resources_resource 
                    WHERE authors IS NOT NULL
                """)
                resources = cur.fetchall()

                # Get matches using enhanced matching
                matches = self.match_experts_to_resources(experts, resources)

                # Store matches in database with confidence scores
                for expert_id, resource_matches in matches.items():
                    for resource_id, confidence in resource_matches:
                        cur.execute("""
                            INSERT INTO expert_resource_links 
                                (expert_id, resource_id, confidence_score)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (expert_id, resource_id) 
                            DO UPDATE SET confidence_score = %s
                        """, (expert_id, resource_id, confidence, confidence))

                conn.commit()
                
                # Count total links created
                total_links = sum(len(resource_matches) for resource_matches in matches.values())
                self.logger.info(f"Successfully linked {len(matches)} experts to {total_links} resources")

        except Exception as e:
            self.logger.error(f"Error linking experts to resources: {e}")
            raise
    def _identify_aphrc_authors(self, author_list: List[str]) -> Tuple[List[str], List[str]]:
        """
        Split authors into APHRC vs global based on name matching
        Returns: (aphrc_authors, global_authors)
        """
        aphrc_authors = []
        global_authors = []
        
        for author in author_list:
            is_aphrc = False
            normalized_author = self._normalize_name(author)
            
            # Check against cached APHRC experts
            for expert_name, expert_id in self.name_cache.items():
                if self._calculate_name_similarity(normalized_author, expert_name) >= 0.9:
                    aphrc_authors.append(author)
                    is_aphrc = True
                    break
                    
            if not is_aphrc:
                global_authors.append(author)
                
        return aphrc_authors, global_authors

    

# Keeping the original matcher for backward compatibility
class Matcher:
    """Matches experts with resources based on author names"""
    
    def __init__(self):
        self.name_cache = {}
        self.logger = Logger(__name__)
        # Create enhanced matcher instance for use when requested
        self.enhanced_matcher = EnhancedMatcher()
        
    def _normalize_name(self, name: str) -> str:
        """Normalize author name for comparison"""
        return ' '.join(str(name).lower().split())
    
    def match_experts_to_resources(self, experts: List[Dict], resources: List[Dict], 
                             use_enhanced: bool = False) -> Dict[int, List[int]]:
        """
        Match experts to resources based on author names.
        
        Args:
            experts: List of expert data
            resources: List of resource data
            use_enhanced: Whether to use enhanced matching algorithm
            
        Returns:
            Dictionary mapping expert IDs to lists of resource IDs
        """
        if use_enhanced:
            # Use enhanced matching but convert to original format
            enhanced_matches = self.enhanced_matcher.match_experts_to_resources(experts, resources)
            
            # Convert the enhanced format back to the original format
            original_format = {}
            for expert_id, resource_matches in enhanced_matches.items():
                original_format[expert_id] = [resource_id for resource_id, confidence, _ in resource_matches]
                
            return original_format
        
        # Original matching logic
        try:
            matches = {}
            
            # Build expert name lookup
            expert_lookup = {}
            for expert in experts:
                if isinstance(expert, tuple):
                    # Get id and name fields from tuple, accounting for different lengths
                    expert_id = expert[0]  # ID is always first
                    # First and last name are in positions 1 and 2
                    first_name = expert[1] if len(expert) > 1 else ''
                    last_name = expert[2] if len(expert) > 2 else ''
                    full_name = self._normalize_name(f"{first_name} {last_name}")
                    expert_lookup[full_name] = expert_id
                else:
                    # Handle dictionary input
                    expert_lookup[self._normalize_name(expert['name'])] = expert['id']
            
            # Match resources to experts
            for resource in resources:
                try:
                    if isinstance(resource, tuple):
                        # Handle database tuple result - expect (id, authors)
                        resource_id = resource[0]
                        authors = resource[1]
                        
                        # Handle different author data formats
                        if isinstance(authors, str):
                            if not authors.strip():  # Handle empty strings
                                author_list = []
                            else:
                                try:
                                    author_list = json.loads(authors)
                                except json.JSONDecodeError:
                                    # If JSON parsing fails, try treating as single author
                                    author_list = [authors]
                        elif isinstance(authors, list):
                            author_list = authors
                        elif authors is None:
                            author_list = []
                        # NEW: Handle float and other non-iterable types
                        elif isinstance(authors, (float, int)):
                            # Convert to string and use as a single author name
                            author_list = [str(authors)]
                        else:
                            # Try converting to string if other type
                            author_list = [str(authors)]
                    else:
                        # Handle dictionary input
                        resource_id = resource['id']
                        author_list = resource.get('authors', [])
                        if isinstance(author_list, str):
                            try:
                                author_list = json.loads(author_list)
                            except json.JSONDecodeError:
                                author_list = [author_list]
                        # NEW: Handle float and other non-iterable types
                        elif isinstance(author_list, (float, int)):
                            author_list = [str(author_list)]
                    
                    # Skip empty author lists
                    if not author_list:
                        continue
                        
                    # Ensure author_list is always a list (even for non-iterable types)
                    if not isinstance(author_list, list):
                        author_list = [str(author_list)]
                        
                    # Process each author
                    for author in author_list:
                        if not author:  # Skip empty author names
                            continue
                        normalized_name = self._normalize_name(author)
                        if normalized_name in expert_lookup:
                            expert_id = expert_lookup[normalized_name]
                            if expert_id not in matches:
                                matches[expert_id] = []
                            matches[expert_id].append(resource_id)
                                
                except (KeyError, TypeError) as e:
                    self.logger.warning(f"Skipping malformed resource {resource_id if 'resource_id' in locals() else 'unknown'}: {e}")
                    continue
            
            self.logger.info(f"Found {len(matches)} expert-resource matches")
            return matches
            
        except Exception as e:
            self.logger.error(f"Error matching experts to resources: {e}")
            raise

    

    

    def link_matched_experts_to_db(self, use_enhanced: bool = False) -> None:
        """
        Link matched experts to resources in database
        
        Args:
            use_enhanced: Whether to use enhanced matching algorithm
        """
        try:
            with get_db_cursor() as (cur, conn):
                # Get fields based on matching method
                if use_enhanced:
                    # For enhanced matching, get additional fields
                    cur.execute("""
                        SELECT id, first_name, last_name, knowledge_expertise, domains
                        FROM experts_expert
                        WHERE is_active = TRUE
                    """)
                    experts = cur.fetchall()

                    # Get resources with additional context fields
                    cur.execute("""
                        SELECT id, authors, title, abstract, summary, domains, topics
                        FROM resources_resource 
                        WHERE authors IS NOT NULL
                    """)
                    resources = cur.fetchall()
                    
                    # Use enhanced matcher directly
                    enhanced_matches = self.enhanced_matcher.match_experts_to_resources(experts, resources)
                    
                    # Store matches with confidence scores
                    for expert_id, resource_matches in enhanced_matches.items():
                        for resource_id, confidence in resource_matches:
                            cur.execute("""
                                INSERT INTO expert_resource_links 
                                    (expert_id, resource_id, confidence_score)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (expert_id, resource_id) 
                                DO UPDATE SET confidence_score = %s
                            """, (expert_id, resource_id, confidence, confidence))
                    
                    # Count total links
                    total_links = sum(len(resource_matches) for resource_matches in enhanced_matches.values())
                    self.logger.info(f"Successfully linked {len(enhanced_matches)} experts to {total_links} resources using enhanced matching")
                
                else:
                    # Original method
                    cur.execute("""
                        SELECT id, first_name, last_name 
                        FROM experts_expert
                        WHERE is_active = TRUE
                    """)
                    experts = cur.fetchall()

                    # Get all resources
                    cur.execute("""
                        SELECT id, authors 
                        FROM resources_resource 
                        WHERE authors IS NOT NULL
                    """)
                    resources = cur.fetchall()

                    # Get matches using tuple data directly
                    matches = self.match_experts_to_resources(experts, resources)

                    # Store matches in database
                    for expert_id, resource_ids in matches.items():
                        for resource_id in resource_ids:
                            cur.execute("""
                                INSERT INTO expert_resource_links 
                                    (expert_id, resource_id, confidence_score)
                                VALUES (%s, %s, 1.0)
                                ON CONFLICT (expert_id, resource_id) DO NOTHING
                            """, (expert_id, resource_id))

                    self.logger.info(f"Successfully linked {len(matches)} experts to resources using original matching")
                
                conn.commit()

        except Exception as e:
            self.logger.error(f"Error linking experts to resources: {e}")
            raise

# Main function to demonstrate usage
def link_experts_to_resources(use_enhanced: bool = True):
    """
    Link experts to resources using the appropriate matcher
    
    Args:
        use_enhanced: Whether to use enhanced matching algorithm
    """
    matcher = Matcher()
    matcher.link_matched_experts_to_db(use_enhanced=use_enhanced)
    return "Expert-resource linking completed successfully"