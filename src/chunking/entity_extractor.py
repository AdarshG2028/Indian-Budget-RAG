import re
from typing import Dict, List
from .models import EntityExtraction

class EntityExtractor:
    @staticmethod
    def extract_entities(text: str) -> EntityExtraction:
        """Extract entities using regex where possible."""
        entities = EntityExtraction()
        
        # Extract amounts (₹ symbol followed by numbers and optionally crore/lakh/billion)
        amount_pattern = r'₹\s*[\d,]+(?:\.\d+)?\s*(?:crore|lakh|billion)?'
        entities.amounts = list(set(re.findall(amount_pattern, text, re.IGNORECASE)))
        
        # Extract percentages
        pct_pattern = r'\b\d+(?:\.\d+)?\s*%'
        entities.percentages = list(set(re.findall(pct_pattern, text)))
        
        # Extract Acts
        act_pattern = r'\b[A-Z][a-zA-Z\s]+Act\b'
        entities.acts = list(set(re.findall(act_pattern, text)))
        
        # Extract Schemes
        scheme_pattern = r'\b[A-Z][a-zA-Z\s]+(?:Scheme|Mission|Yojana|Fund|Programme)\b'
        entities.schemes = list(set(re.findall(scheme_pattern, text)))
        
        # Dates (basic extraction for years or specific months)
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b|\b\d{4}-\d{2}\b'
        entities.dates = list(set(re.findall(date_pattern, text)))
        
        # Leave others empty for now unless we implement proper NER
        
        return entities
