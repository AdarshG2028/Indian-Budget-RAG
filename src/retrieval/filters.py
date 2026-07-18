"""
Extensible metadata filter system with support for various operators.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class FilterOperator(str, Enum):
    """Supported filter operators."""
    EQ = "eq"  # Equality
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    IN = "in"  # In list
    NIN = "nin"  # Not in list
    CONTAINS = "contains"  # String contains
    STARTS_WITH = "starts_with"  # String starts with
    ENDS_WITH = "ends_with"  # String ends with


@dataclass
class FilterCondition:
    """
    A single filter condition with field, operator, and value.
    
    Examples:
        FilterCondition(field="year", operator=FilterOperator.GTE, value=2020)
        FilterCondition(field="document", operator=FilterOperator.IN, value=["budget_speech", "finance_bill"])
        FilterCondition(field="section", operator=FilterOperator.CONTAINS, value="agriculture")
    """
    field: str
    operator: FilterOperator
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value
        }


class FilterBuilder:
    """
    Builder for constructing complex filter expressions.
    
    Supports:
    - Simple equality filters via dictionary
    - Complex conditions with operators
    - Multiple conditions combined with AND logic
    """
    
    @staticmethod
    def from_dict(filters: Dict[str, Any]) -> List[FilterCondition]:
        """
        Convert a simple dictionary to filter conditions.
        
        For simple dictionaries, assumes equality operator.
        For complex structures, expects operator specification.
        
        Examples:
            {"year": 2026} -> [FilterCondition(field="year", operator=EQ, value=2026)]
            {"year": {"gte": 2020, "lte": 2030}} -> [FilterCondition(field="year", operator=GTE, value=2020), ...]
        """
        conditions = []
        
        for field, value in filters.items():
            if isinstance(value, dict):
                # Handle operator specification
                for op_str, op_value in value.items():
                    try:
                        operator = FilterOperator(op_str.lower())
                        conditions.append(FilterCondition(field=field, operator=operator, value=op_value))
                    except ValueError:
                        raise ValueError(f"Unknown filter operator: {op_str}")
            else:
                # Default to equality
                conditions.append(FilterCondition(field=field, operator=FilterOperator.EQ, value=value))
        
        return conditions
    
    @staticmethod
    def build(conditions: List[FilterCondition]) -> List[FilterCondition]:
        """
        Build a filter expression from conditions.
        
        Currently supports AND logic (all conditions must match).
        Future extension: support OR logic and nested conditions.
        """
        return conditions


class FilterConverter:
    """
    Converts filter conditions to backend-specific filter expressions.
    
    This abstraction allows the retriever to work with different
    vector databases without changing the filter logic.
    """
    
    @staticmethod
    def to_qdrant_filter(conditions: List[FilterCondition]) -> Optional[Any]:
        """
        Convert filter conditions to Qdrant filter expression.
        
        Args:
            conditions: List of FilterCondition objects
            
        Returns:
            Qdrant Filter object or None
        """
        if not conditions:
            return None
        
        try:
            from qdrant_client.http import models as rest
        except ImportError:
            raise ImportError("qdrant-client is required")
        
        qdrant_conditions = []
        must_not_conditions = []
        
        for condition in conditions:
            field = condition.field
            operator = condition.operator
            value = condition.value
            
            if operator == FilterOperator.EQ:
                qdrant_conditions.append(
                    rest.FieldCondition(key=field, match=rest.MatchValue(value=value))
                )
            elif operator == FilterOperator.NE:
                must_not_conditions.append(
                    rest.FieldCondition(key=field, match=rest.MatchValue(value=value))
                )
            elif operator == FilterOperator.GT:
                qdrant_conditions.append(
                    rest.FieldCondition(key=field, range=rest.Range(gt=value))
                )
            elif operator == FilterOperator.GTE:
                qdrant_conditions.append(
                    rest.FieldCondition(key=field, range=rest.Range(gte=value))
                )
            elif operator == FilterOperator.LT:
                qdrant_conditions.append(
                    rest.FieldCondition(key=field, range=rest.Range(lt=value))
                )
            elif operator == FilterOperator.LTE:
                qdrant_conditions.append(
                    rest.FieldCondition(key=field, range=rest.Range(lte=value))
                )
            elif operator == FilterOperator.IN:
                qdrant_conditions.append(
                    rest.FieldCondition(key=field, match=rest.MatchAny(any=value))
                )
            elif operator == FilterOperator.NIN:
                must_not_conditions.append(
                    rest.FieldCondition(key=field, match=rest.MatchAny(any=value))
                )
            elif operator == FilterOperator.CONTAINS:
                qdrant_conditions.append(
                    rest.FieldCondition(key=field, match=rest.MatchText(text=value))
                )
            elif operator == FilterOperator.STARTS_WITH:
                if isinstance(value, str):
                    qdrant_conditions.append(
                        rest.FieldCondition(key=field, match=rest.MatchText(text=value))
                    )
            elif operator == FilterOperator.ENDS_WITH:
                if isinstance(value, str):
                    qdrant_conditions.append(
                        rest.FieldCondition(key=field, match=rest.MatchText(text=value))
                    )
        
        if qdrant_conditions or must_not_conditions:
            return rest.Filter(
                must=qdrant_conditions if qdrant_conditions else None,
                must_not=must_not_conditions if must_not_conditions else None
            )
        return None
