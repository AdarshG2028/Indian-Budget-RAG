"""
Experiment tracking and management for evaluation framework.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import EvaluationReport, ExperimentComparison

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Tracks and manages evaluation experiments.
    
    Responsibilities:
    - Save experiment reports to disk
    - Load experiment reports from disk
    - Compare experiments
    - List available experiments
    """
    
    def __init__(self, storage_dir: str = "evaluation/experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            storage_dir: Directory to store experiment reports
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ExperimentTracker initialized with storage: {self.storage_dir}")
    
    def save_experiment(self, report: EvaluationReport) -> str:
        """
        Save experiment report to disk.
        
        Args:
            report: Evaluation report to save
            
        Returns:
            Path to saved file
        """
        filename = f"{report.metadata.experiment_id}.json"
        filepath = self.storage_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved experiment to: {filepath}")
        return str(filepath)
    
    def load_experiment(self, experiment_id: str) -> EvaluationReport:
        """
        Load experiment report from disk.
        
        Args:
            experiment_id: Experiment ID to load
            
        Returns:
            EvaluationReport object
            
        Raises:
            FileNotFoundError: If experiment not found
        """
        filepath = self.storage_dir / f"{experiment_id}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct EvaluationReport from dict
        from .models import (
            ExperimentMetadata, QueryResult, EvaluationQuery, 
            GroundTruthChunk, RetrievedChunk
        )
        
        # Reconstruct metadata
        metadata = ExperimentMetadata(
            experiment_id=data["metadata"]["experiment_id"],
            timestamp=datetime.fromisoformat(data["metadata"]["timestamp"]),
            embedding_model=data["metadata"]["embedding_model"],
            retriever=data["metadata"]["retriever"],
            reranker=data["metadata"]["reranker"],
            vector_database=data["metadata"]["vector_database"],
            chunk_size=data["metadata"]["chunk_size"],
            chunk_overlap=data["metadata"]["chunk_overlap"],
            collection=data["metadata"]["collection"],
            dataset=data["metadata"]["dataset"],
            retrieval_config=data["metadata"]["retrieval_config"],
            additional_metadata=data["metadata"].get("additional_metadata")
        )
        
        # Reconstruct query results (simplified for now)
        # In a full implementation, we'd reconstruct all objects properly
        query_results = []  # Simplified
        
        report = EvaluationReport(
            metadata=metadata,
            query_results=query_results,
            aggregate_metrics=data["aggregate_metrics"],
            failure_analysis=data["failure_analysis"],
            latency_metrics=data["latency_metrics"]
        )
        
        logger.info(f"Loaded experiment: {experiment_id}")
        return report
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all available experiments.
        
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for filepath in self.storage_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                experiments.append({
                    "experiment_id": data["metadata"]["experiment_id"],
                    "timestamp": data["metadata"]["timestamp"],
                    "dataset": data["metadata"]["dataset"],
                    "retriever": data["metadata"]["retriever"],
                    "embedding_model": data["metadata"]["embedding_model"],
                    "aggregate_metrics": data["aggregate_metrics"]
                })
            except Exception as e:
                logger.warning(f"Failed to load experiment {filepath}: {e}")
        
        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return experiments
    
    def compare_experiments(
        self,
        experiment_id_a: str,
        experiment_id_b: str
    ) -> ExperimentComparison:
        """
        Compare two experiments.
        
        Args:
            experiment_id_a: First experiment ID
            experiment_id_b: Second experiment ID
            
        Returns:
            ExperimentComparison object
        """
        report_a = self.load_experiment(experiment_id_a)
        report_b = self.load_experiment(experiment_id_b)
        
        # Compute metric differences
        metric_differences = {}
        improvement_summary = {
            "improved": [],
            "degraded": [],
            "unchanged": []
        }
        
        for metric_name in report_a.aggregate_metrics:
            if metric_name in report_b.aggregate_metrics:
                value_a = report_a.aggregate_metrics[metric_name]
                value_b = report_b.aggregate_metrics[metric_name]
                difference = value_b - value_a
                percent_change = (difference / value_a * 100) if value_a != 0 else 0
                
                metric_differences[metric_name] = {
                    "value_a": value_a,
                    "value_b": value_b,
                    "difference": difference,
                    "percent_change": percent_change
                }
                
                # Categorize improvement
                if abs(percent_change) < 0.01:  # Less than 1% change
                    improvement_summary["unchanged"].append(metric_name)
                elif percent_change > 0:
                    improvement_summary["improved"].append({
                        "metric": metric_name,
                        "percent_change": percent_change
                    })
                else:
                    improvement_summary["degraded"].append({
                        "metric": metric_name,
                        "percent_change": percent_change
                    })
        
        return ExperimentComparison(
            experiment_a=report_a,
            experiment_b=report_b,
            metric_differences=metric_differences,
            improvement_summary=improvement_summary
        )
    
    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete an experiment.
        
        Args:
            experiment_id: Experiment ID to delete
        """
        filepath = self.storage_dir / f"{experiment_id}.json"
        
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted experiment: {experiment_id}")
        else:
            logger.warning(f"Experiment not found: {experiment_id}")
