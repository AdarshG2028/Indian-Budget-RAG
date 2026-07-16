"""
Evaluation service for business logic.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from evaluation import EvaluationRunner, ExperimentTracker, ReportGenerator
from retrieval import RetrievalConfig

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Background job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationJob:
    """Background evaluation job."""
    
    def __init__(
        self,
        job_id: str,
        dataset_path: str,
        config: Dict[str, Any],
        save_experiment: bool,
        output_formats: list[str]
    ):
        """
        Initialize evaluation job.
        
        Args:
            job_id: Unique job identifier
            dataset_path: Path to evaluation dataset
            config: Evaluation configuration
            save_experiment: Whether to save experiment
            output_formats: Output formats
        """
        self.job_id = job_id
        self.dataset_path = dataset_path
        self.config = config
        self.save_experiment = save_experiment
        self.output_formats = output_formats
        self.status = JobStatus.PENDING
        self.message = None
        self.result = None
        self.created_at = datetime.utcnow()
        self.completed_at = None


class EvaluationService:
    """
    Service for evaluation operations.
    
    Handles business logic for evaluation experiments.
    """
    
    def __init__(
        self,
        evaluation_runner: EvaluationRunner,
        output_dir: str = "evaluation/reports",
        experiment_dir: str = "evaluation/experiments"
    ):
        """
        Initialize evaluation service.
        
        Args:
            evaluation_runner: Evaluation runner instance
            output_dir: Output directory for reports
            experiment_dir: Directory for experiment storage
        """
        self.evaluation_runner = evaluation_runner
        self.output_dir = output_dir
        self.experiment_dir = experiment_dir
        self.jobs: Dict[str, EvaluationJob] = {}
    
    def create_job(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        save_experiment: bool = True,
        output_formats: list[str] = None
    ) -> str:
        """
        Create an evaluation job.
        
        Args:
            dataset_path: Path to evaluation dataset
            config: Evaluation configuration
            save_experiment: Whether to save experiment
            output_formats: Output formats
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        if output_formats is None:
            output_formats = ["json", "markdown"]
        
        job = EvaluationJob(
            job_id=job_id,
            dataset_path=dataset_path,
            config=config,
            save_experiment=save_experiment,
            output_formats=output_formats
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created evaluation job: {job_id}")
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[EvaluationJob]:
        """
        Get evaluation job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Evaluation job or None
        """
        return self.jobs.get(job_id)
    
    def run_job(self, job_id: str) -> None:
        """
        Run evaluation job (synchronous for now).
        
        Args:
            job_id: Job identifier
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        job.status = JobStatus.RUNNING
        job.message = "Running evaluation"
        
        try:
            logger.info(f"Running evaluation job: {job_id}")
            
            # Get configuration
            top_k = job.config.get("top_k", 10)
            score_threshold = job.config.get("score_threshold", None)
            
            # Create retrieval config
            retrieval_config = RetrievalConfig(
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            # Run evaluation
            report = self.evaluation_runner.run(
                dataset_path=job.dataset_path,
                retrieval_config=retrieval_config,
                top_k=top_k
            )
            
            # Generate reports
            from pathlib import Path
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if "json" in job.output_formats:
                json_path = output_dir / f"{report.metadata.experiment_id}.json"
                ReportGenerator.generate_json(report, str(json_path))
            
            if "csv" in job.output_formats:
                csv_dir = output_dir / f"{report.metadata.experiment_id}_csv"
                ReportGenerator.generate_csv(report, str(csv_dir))
            
            if "markdown" in job.output_formats:
                md_path = output_dir / f"{report.metadata.experiment_id}.md"
                ReportGenerator.generate_markdown(report, str(md_path))
            
            # Save experiment if requested
            if job.save_experiment:
                tracker = ExperimentTracker(storage_dir=self.experiment_dir)
                tracker.save_experiment(report)
            
            # Update job
            job.status = JobStatus.COMPLETED
            job.message = "Evaluation completed successfully"
            job.result = {
                "experiment_id": report.metadata.experiment_id,
                "aggregate_metrics": report.aggregate_metrics,
                "latency_metrics": report.latency_metrics,
                "failure_analysis": report.failure_analysis
            }
            job.completed_at = datetime.utcnow()
            
            logger.info(f"Evaluation job completed: {job_id}")
            
        except Exception as e:
            logger.error(f"Evaluation job failed: {job_id}, error: {e}")
            job.status = JobStatus.FAILED
            job.message = f"Evaluation failed: {str(e)}"
            job.completed_at = datetime.utcnow()
    
    def list_jobs(self) -> list[Dict[str, Any]]:
        """
        List all evaluation jobs.
        
        Returns:
            List of job summaries
        """
        return [
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "message": job.message,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            for job in self.jobs.values()
        ]
