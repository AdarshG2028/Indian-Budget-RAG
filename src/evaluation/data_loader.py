"""
Dataset loader for evaluation framework.
Supports JSON and CSV formats with validation.
"""
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import EvaluationQuery, GroundTruthChunk

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


class DataLoader:
    """
    Loads and validates evaluation datasets from JSON or CSV files.
    
    Supported formats:
    - JSON: Array of query objects with ground truth
    - CSV: Rows with query and ground truth information
    """
    
    @staticmethod
    def load_json(file_path: str) -> List[EvaluationQuery]:
        """
        Load evaluation dataset from JSON file.
        
        Expected format:
        [
            {
                "query_id": "q1",
                "query_text": "What is the budget?",
                "ground_truth": [
                    {"chunk_id": "chunk_001", "relevance": 3},
                    {"chunk_id": "chunk_005", "relevance": 2}
                ],
                "metadata": {"category": "budget"}
            }
        ]
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of EvaluationQuery objects
            
        Raises:
            DatasetValidationError: If dataset is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise DatasetValidationError(f"File not found: {file_path}")
        
        logger.info(f"Loading dataset from JSON: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise DatasetValidationError("JSON must contain an array of queries")
        
        queries = []
        chunk_ids_seen = set()
        
        for idx, item in enumerate(data):
            try:
                query = DataLoader._parse_json_query(item, idx, chunk_ids_seen)
                queries.append(query)
            except Exception as e:
                raise DatasetValidationError(f"Error parsing query at index {idx}: {e}")
        
        logger.info(f"Loaded {len(queries)} queries from JSON")
        DataLoader._validate_dataset(queries, chunk_ids_seen)
        
        return queries
    
    @staticmethod
    def _parse_json_query(
        item: Dict[str, Any],
        idx: int,
        chunk_ids_seen: set[str]
    ) -> EvaluationQuery:
        """Parse a single query from JSON."""
        if "query_id" not in item:
            raise ValueError(f"Missing 'query_id' field")
        if "query_text" not in item:
            raise ValueError(f"Missing 'query_text' field")
        if "ground_truth" not in item:
            raise ValueError(f"Missing 'ground_truth' field")
        
        query_id = item["query_id"]
        query_text = item["query_text"]
        ground_truth_data = item["ground_truth"]
        
        if not isinstance(ground_truth_data, list):
            raise ValueError(f"'ground_truth' must be a list")
        
        ground_truth = []
        for gt_item in ground_truth_data:
            if "chunk_id" not in gt_item:
                raise ValueError(f"Missing 'chunk_id' in ground truth")
            if "relevance" not in gt_item:
                raise ValueError(f"Missing 'relevance' in ground truth")
            
            chunk_id = gt_item["chunk_id"]
            relevance = gt_item["relevance"]
            
            # Check for duplicate chunk IDs within same query
            if chunk_id in {gt.chunk_id for gt in ground_truth}:
                raise ValueError(f"Duplicate chunk_id '{chunk_id}' in ground truth")
            
            ground_truth.append(GroundTruthChunk(chunk_id=chunk_id, relevance=relevance))
            chunk_ids_seen.add(chunk_id)
        
        metadata = item.get("metadata")
        
        return EvaluationQuery(
            query_id=query_id,
            query_text=query_text,
            ground_truth=ground_truth,
            metadata=metadata
        )
    
    @staticmethod
    def load_csv(file_path: str) -> List[EvaluationQuery]:
        """
        Load evaluation dataset from CSV file.
        
        Expected format:
        query_id,query_text,chunk_id,relevance,metadata
        q1,What is the budget?,chunk_001,3,category:budget
        q1,What is the budget?,chunk_005,2,category:budget
        q2,Tax changes,chunk_010,3,category:tax
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of EvaluationQuery objects
            
        Raises:
            DatasetValidationError: If dataset is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise DatasetValidationError(f"File not found: {file_path}")
        
        logger.info(f"Loading dataset from CSV: {file_path}")
        
        queries_dict: Dict[str, Dict[str, Any]] = {}
        chunk_ids_seen = set()
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            required_fields = ["query_id", "query_text", "chunk_id", "relevance"]
            for field in required_fields:
                if field not in reader.fieldnames:
                    raise DatasetValidationError(f"Missing required field: {field}")
            
            for row_idx, row in enumerate(reader):
                try:
                    query_id = row["query_id"]
                    query_text = row["query_text"]
                    chunk_id = row["chunk_id"]
                    relevance = int(row["relevance"])
                    metadata = row.get("metadata")
                    
                    # Initialize query if not exists
                    if query_id not in queries_dict:
                        queries_dict[query_id] = {
                            "query_text": query_text,
                            "ground_truth": [],
                            "metadata": {}
                        }
                    
                    # Validate consistency
                    if queries_dict[query_id]["query_text"] != query_text:
                        raise ValueError(
                            f"Inconsistent query_text for query_id '{query_id}' "
                            f"at row {row_idx + 1}"
                        )
                    
                    # Check for duplicate chunk IDs within same query
                    existing_chunk_ids = {gt.chunk_id for gt in queries_dict[query_id]["ground_truth"]}
                    if chunk_id in existing_chunk_ids:
                        raise ValueError(
                            f"Duplicate chunk_id '{chunk_id}' for query '{query_id}' "
                            f"at row {row_idx + 1}"
                        )
                    
                    # Add ground truth chunk
                    queries_dict[query_id]["ground_truth"].append(
                        GroundTruthChunk(chunk_id=chunk_id, relevance=relevance)
                    )
                    chunk_ids_seen.add(chunk_id)
                    
                    # Parse metadata if provided
                    if metadata:
                        try:
                            if isinstance(metadata, str):
                                # Parse key:value format
                                for item in metadata.split(','):
                                    key, value = item.split(':', 1)
                                    queries_dict[query_id]["metadata"][key.strip()] = value.strip()
                        except Exception as e:
                            logger.warning(f"Failed to parse metadata at row {row_idx + 1}: {e}")
                
                except Exception as e:
                    raise DatasetValidationError(f"Error parsing row {row_idx + 1}: {e}")
        
        # Convert to list of EvaluationQuery
        queries = []
        for query_id, data in queries_dict.items():
            queries.append(
                EvaluationQuery(
                    query_id=query_id,
                    query_text=data["query_text"],
                    ground_truth=data["ground_truth"],
                    metadata=data["metadata"] if data["metadata"] else None
                )
            )
        
        logger.info(f"Loaded {len(queries)} queries from CSV")
        DataLoader._validate_dataset(queries, chunk_ids_seen)
        
        return queries
    
    @staticmethod
    def _validate_dataset(queries: List[EvaluationQuery], chunk_ids_seen: set[str]) -> None:
        """
        Validate the loaded dataset.
        
        Args:
            queries: List of evaluation queries
            chunk_ids_seen: Set of all chunk IDs seen
            
        Raises:
            DatasetValidationError: If validation fails
        """
        if not queries:
            raise DatasetValidationError("Dataset is empty")
        
        # Check for duplicate query IDs
        query_ids = [q.query_id for q in queries]
        if len(query_ids) != len(set(query_ids)):
            raise DatasetValidationError("Duplicate query IDs found")
        
        # Check for queries with no ground truth
        empty_queries = [q for q in queries if not q.ground_truth]
        if empty_queries:
            raise DatasetValidationError(
                f"{len(empty_queries)} queries have no ground truth"
            )
        
        # Validate relevance scores
        for query in queries:
            for gt in query.ground_truth:
                if not 0 <= gt.relevance <= 3:
                    raise DatasetValidationError(
                        f"Invalid relevance score {gt.relevance} for chunk '{gt.chunk_id}' "
                        f"in query '{query.query_id}'. Must be between 0 and 3."
                    )
        
        logger.info("Dataset validation passed")
    
    @staticmethod
    def load(file_path: str) -> List[EvaluationQuery]:
        """
        Auto-detect format and load dataset.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            List of EvaluationQuery objects
            
        Raises:
            DatasetValidationError: If loading fails
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".json":
            return DataLoader.load_json(file_path)
        elif extension == ".csv":
            return DataLoader.load_csv(file_path)
        else:
            raise DatasetValidationError(
                f"Unsupported file format: {extension}. "
                "Supported formats: .json, .csv"
            )
