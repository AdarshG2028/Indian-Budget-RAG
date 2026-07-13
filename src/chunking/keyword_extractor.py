from typing import List
import re
from collections import Counter

class KeywordExtractor:
    # Common stopwords and non-informative words
    STOPWORDS = {
        'since', 'this', 'that', 'these', 'those', 'keeping', 'being', 'having',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'been', 'were', 'was', 'are', 'is', 'am', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'can', 'cannot', 'need', 'needs', 'needed',
        'also', 'and', 'or', 'but', 'if', 'then', 'than', 'so', 'such',
        'because', 'while', 'when', 'where', 'which', 'who', 'whom', 'whose',
        'what', 'how', 'why', 'there', 'here', 'now', 'then', 'again',
        'further', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'also', 'now', 'well', 'way', 'our', 'their',
        'its', 'his', 'her', 'my', 'your', 'our', 'their', 'mine', 'yours',
        'hers', 'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself',
        'itself', 'ourselves', 'yourselves', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
    }
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract meaningful keywords from text using frequency-based approach.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords sorted by relevance
        """
        # Remove markdown formatting
        clean_text = re.sub(r'[*_`#]', ' ', text)
        clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)
        
        # Extract words (3+ characters, letters only)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_text)
        
        # Filter out stopwords and convert to lowercase
        filtered_words = [
            word.lower() for word in words 
            if word.lower() not in KeywordExtractor.STOPWORDS
        ]
        
        # Count frequency
        word_freq = Counter(filtered_words)
        
        # Filter by minimum frequency (at least 2 occurrences)
        frequent_words = {
            word: freq for word, freq in word_freq.items() 
            if freq >= 2
        }
        
        # Prioritize capitalized words (likely proper nouns/important terms)
        # by giving them a boost in score
        scored_words = {}
        original_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        original_lower = [w.lower() for w in original_words]
        
        for word, freq in frequent_words.items():
            # Check if word appears capitalized in original text
            capitalized_count = sum(1 for i, w in enumerate(original_lower) if w == word and original_words[i][0].isupper())
            score = freq + (capitalized_count * 0.5)  # Boost for capitalized occurrences
            scored_words[word] = score
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(scored_words.items(), key=lambda x: x[1], reverse=True)
        
        # Return original casing of words
        result = []
        for word, score in sorted_keywords[:max_keywords]:
            # Find the most common casing in original text
            word_variants = Counter([w for w in original_words if w.lower() == word])
            if word_variants:
                result.append(word_variants.most_common(1)[0][0])
            else:
                result.append(word.capitalize())
        
        return result
