"""
Evaluation router for evaluation endpoints.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List

from ..models import EvaluationRequest, EvaluationJobResponse, JobStatus
from ..services import EvaluationService
from ..dependencies import get_evaluation_runner, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluationJobResponse)
async def evaluate(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    evaluation_service: EvaluationService = Depends(lambda r: EvaluationService(r, get_settings().evaluation_output_dir, get_settings().evaluation_experiment_dir)),
    runner = Depends(get_evaluation_runner)
) -> EvaluationJobResponse:
    """
    Run evaluation experiment.
    
    Creates a background job to run evaluation.
    
    Args:
        request: Evaluation request
        background_tasks: FastAPI background tasks
        evaluation_service: Evaluation service
        runner: Evaluation runner
        
    Returns:
        Evaluation job response with job ID
    """
    try:
        logger.info("Creating evaluation job")
        
        # Create service instance
        service = EvaluationService(
            runner,
            output_dir=get_settings().evaluation_output_dir,
            experiment_dir=get_settings().evaluation_experiment_dir
        )
        
        # Create job
        job_id = service.create_job(
            dataset_path=request.dataset,
            config=request.config or {},
            save_experiment=request.save_experiment,
            output_formats=request.output_formats
        )
        
        # Run job in background
        background_tasks.add_task(service.run_job, job_id)
        
        job = service.get_job(job_id)
        
        return EvaluationJobResponse(
            request_id="evaluation",  # Will be set by middleware
            job_id=job_id,
            status=JobStatus(job.status),
            message=job.message,
            created_at=job.created_at
        )
        
    except Exception as e:
        logger.error(f"Evaluation job creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=List[EvaluationJobResponse])
async def list_jobs(
    evaluation_service: EvaluationService = Depends(lambda r: EvaluationService(r, get_settings().evaluation_output_dir, get_settings().evaluation_experiment_dir)),
    runner = Depends(get_evaluation_runner)
) -> List[EvaluationJobResponse]:
    """
    List all evaluation jobs.
    
    Args:
        evaluation_service: Evaluation service
        runner: Evaluation runner
        
    Returns:
        List of evaluation jobs
    """
    try:
        # Create service instance
        service = EvaluationService(
            runner,
            output_dir=get_settings().evaluation_output_dir,
            experiment_dir=get_settings().evaluation_experiment_dir
        )
        
        jobs = service.list_jobs()
        
        return [
            EvaluationJobResponse(
                request_id="evaluation",
                job_id=job["job_id"],
                status=JobStatus(job["status"]),
                message=job.get("message"),
                created_at=job["created_at"],
                completed_at=job.get("completed_at")
            )
            for job in jobs
        ]
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=EvaluationJobResponse)
async def get_job(
    job_id: str,
    evaluation_service: EvaluationService = Depends(lambda r: EvaluationService(r, get_settings().evaluation_output_dir, get_settings().evaluation_experiment_dir)),
    runner = Depends(get_evaluation_runner)
) -> EvaluationJobResponse:
    """
    Get evaluation job by ID.
    
    Args:
        job_id: Job identifier
        evaluation_service: Evaluation service
        runner: Evaluation runner
        
    Returns:
        Evaluation job details
    """
    try:
        # Create service instance
        service = EvaluationService(
            runner,
            output_dir=get_settings().evaluation_output_dir,
            experiment_dir=get_settings().evaluation_experiment_dir
        )
        
        job = service.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return EvaluationJobResponse(
            request_id="evaluation",
            job_id=job.job_id,
            status=JobStatus(job.status),
            message=job.message,
            result=job.result,
            created_at=job.created_at,
            completed_at=job.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job: {e}")
        raise HTTPException(status_code=500, detail=str(e))
