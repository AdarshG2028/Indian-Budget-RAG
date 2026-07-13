from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class EntityExtraction:
    amounts: List[str] = field(default_factory=list)
    percentages: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    acts: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    ministries: List[str] = field(default_factory=list)
    schemes: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)

@dataclass
class Chunk:
    chunk_id: str
    document: str
    year: int
    section: str
    subsection: str
    paragraph_start: int
    paragraph_end: int
    page_start: int
    page_end: int
    token_count: int
    char_count: int
    text: str
    keywords: List[str] = field(default_factory=list)
    entities: EntityExtraction = field(default_factory=EntityExtraction)
