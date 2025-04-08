import time
import logging
from typing import List, Tuple, Dict, Any

# Configure logger
logger = logging.getLogger(__name__)

# Enhanced version of PrefixTrie from paste-5.txt
class PrefixTrie:
    """Efficient prefix trie for fast query suggestions with improved filtering"""
    
    def __init__(self):
        self.root = {}
        self.end_symbol = "*"
        self.frequency_key = "freq"
        self.timestamp_key = "ts"
        self.source_key = "src"  # Track where this suggestion came from
    
    def insert(self, word: str, frequency: int = 1, timestamp: float = None, source: str = None) -> None:
        """
        Insert a word into the trie with its frequency, timestamp and source
        
        Args:
            word: The word to insert
            frequency: Occurrence frequency (higher = more important)
            timestamp: Optional timestamp of when this was added
            source: Optional source of this suggestion (e.g. 'history', 'trending')
        """
        if not word:
            return
            
        word = word.lower()
        node = self.root
        
        # Insert each character
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        
        # Mark the end of the word and update frequency
        if self.end_symbol not in node:
            node[self.end_symbol] = {}
            node[self.end_symbol][self.frequency_key] = frequency
            node[self.end_symbol][self.timestamp_key] = timestamp or time.time()
            if source:
                node[self.end_symbol][self.source_key] = source
        else:
            # Increment frequency for existing words
            node[self.end_symbol][self.frequency_key] += frequency
            # Update timestamp if newer
            if timestamp and (self.timestamp_key not in node[self.end_symbol] or 
                             timestamp > node[self.end_symbol][self.timestamp_key]):
                node[self.end_symbol][self.timestamp_key] = timestamp
            # Update source if provided
            if source:
                node[self.end_symbol][self.source_key] = source
    
    def search(self, prefix: str, limit: int = 10) -> List[Tuple[str, int, float, str]]:
        """
        Search for words that start with the given prefix
        
        Args:
            prefix: String prefix to search for
            limit: Maximum number of results to return
            
        Returns:
            List of (word, frequency, timestamp, source) tuples
        """
        if prefix is None:
            prefix = ""
            
        prefix = prefix.lower()
        node = self.root
        
        # For empty prefix, return top frequent items
        if not prefix:
            results = []
            self._collect_top_frequent_words(self.root, "", results, limit)
            return results
        
        # Navigate to the prefix node
        for char in prefix:
            if char not in node:
                return []  # Prefix not found
            node = node[char]
        
        # Find all words with this prefix
        results = []
        self._collect_words(node, prefix, results)
        
        # Sort by frequency (descending) and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _collect_top_frequent_words(self, node, current_prefix, results, limit):
        """Collect most frequent words from the trie"""
        if self.end_symbol in node:
            freq = node[self.end_symbol].get(self.frequency_key, 1)
            ts = node[self.end_symbol].get(self.timestamp_key, 0)
            src = node[self.end_symbol].get(self.source_key, "unknown")
            results.append((current_prefix, freq, ts, src))
        
        # Recursively explore all child nodes
        for char, child_node in node.items():
            if char != self.end_symbol:
                self._collect_top_frequent_words(child_node, current_prefix + char, results, limit)
                
                # Early return if we've collected enough results
                if len(results) >= limit * 2:  # Collect more than needed for better sorting
                    return
    
    def _collect_words(self, node: Dict, prefix: str, results: List[Tuple[str, int, float, str]]) -> None:
        """
        Recursively collect all words from the current node
        
        Args:
            node: Current trie node
            prefix: Current word prefix
            results: List to collect results in
        """
        if self.end_symbol in node:
            freq = node[self.end_symbol].get(self.frequency_key, 1)
            ts = node[self.end_symbol].get(self.timestamp_key, 0)
            src = node[self.end_symbol].get(self.source_key, "unknown")
            results.append((prefix, freq, ts, src))
        
        # Recursively explore all child nodes
        for char, child_node in node.items():
            if char != self.end_symbol:
                self._collect_words(child_node, prefix + char, results)

    def get_size(self) -> int:
        """Get the number of words in the trie"""
        count = [0]
        self._count_words(self.root, count)
        return count[0]
    
    def _count_words(self, node: Dict, count: List[int]) -> None:
        """Helper method to count words in trie"""
        if self.end_symbol in node:
            count[0] += 1
        
        for char, child_node in node.items():
            if char != self.end_symbol:
                self._count_words(child_node, count)
    
    def bulk_insert(self, words: List[Tuple[str, int, float]]) -> None:
        """
        Insert multiple words at once
        
        Args:
            words: List of (word, frequency, timestamp) tuples
        """
        for word, freq, ts in words:
            self.insert(word, freq, ts)

    def fuzzy_search(self, prefix: str, max_distance: int = 1, limit: int = 10) -> List[Tuple[str, int, float]]:
        """
        Search for words with fuzzy matching (allowing for typos/edits)
        
        Args:
            prefix: String prefix to search for
            max_distance: Maximum edit distance allowed
            limit: Maximum number of results to return
            
        Returns:
            List of (word, frequency, timestamp) tuples
        """
        if not prefix:
            return []
            
        prefix = prefix.lower()
        results = []
        
        # Helper function for recursive fuzzy matching
        def _fuzzy_match(node, current_prefix, remaining_prefix, distance, results):
            # If we've consumed the entire prefix
            if not remaining_prefix:
                # Check if this is a word ending
                if self.end_symbol in node:
                    freq = node[self.end_symbol].get(self.frequency_key, 1)
                    ts = node[self.end_symbol].get(self.timestamp_key, 0)
                    results.append((current_prefix, freq, ts))
                
                # Continue searching deeper nodes for longer completions
                for char, child_node in node.items():
                    if char != self.end_symbol:
                        _fuzzy_match(child_node, current_prefix + char, "", distance, results)
                return
            
            # Current character in the remaining prefix
            char = remaining_prefix[0]
            
            # Exact match (no distance used)
            if char in node:
                _fuzzy_match(node[char], current_prefix + char, remaining_prefix[1:], distance, results)
            
            # If we have distance remaining, try fuzzy matches
            if distance > 0:
                # Substitution
                for next_char, child_node in node.items():
                    if next_char != self.end_symbol and next_char != char:
                        _fuzzy_match(child_node, current_prefix + next_char, remaining_prefix[1:], distance - 1, results)
                
                # Deletion (skip this character in the prefix)
                _fuzzy_match(node, current_prefix, remaining_prefix[1:], distance - 1, results)
                
                # Insertion (use a character from the node but don't consume the prefix)
                for next_char, child_node in node.items():
                    if next_char != self.end_symbol:
                        _fuzzy_match(child_node, current_prefix + next_char, remaining_prefix, distance - 1, results)
        
        # Start the recursive fuzzy search
        _fuzzy_match(self.root, "", prefix, max_distance, results)
        
        # Sort by frequency and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    # Other methods remain the same...
   