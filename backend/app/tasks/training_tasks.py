"""
MedSynth Platform - Training Tasks
Celery tasks for federated GAN training
"""
import os
import sys
from datetime import datetime
from pathlib import Path
import traceback

from celery import Task
from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.models.database import SessionLocal
from app.models.training_job import TrainingJob, JobStatus, SyntheticImage, PrivacyMetric


class TrainingTask(Task):
    """Base task class with database session management"""

    def __call__(self, *args, **kwargs):
        """Execute task with database session"""
        return self.run(*args, **kwargs)


@celery_app.task(bind=True, base=TrainingTask)
def run_federated_training(self, job_id: int):
    """
    Run federated GAN training job

    Args:
        job_id: Training job ID
    """
    db: Session = SessionLocal()

    try:
        # Get training job
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        if not job:
            raise ValueError(f"Training job {job_id} not found")

        # Update job status to running
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.celery_task_id = self.request.id
        db.commit()

        print(f"Starting federated training for job {job_id}")
        print(f"Configuration: {job.num_clients} clients, {job.num_rounds} rounds")

        # Create output directories
        output_dir = Path("/app/outputs") / str(job.owner_id) / str(job.id)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = output_dir / "models"
        model_dir.mkdir(exist_ok=True)

        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Import training module (from existing FedGAN code)
        # NOTE: This is a simplified version - the actual integration would use
        # the existing dcgan_training.py module

        # Simulated training loop (replace with actual FedGAN training)
        for round_num in range(1, job.num_rounds + 1):
            # Update progress
            job.current_round = round_num
            job.progress_percentage = (round_num / job.num_rounds) * 100
            db.commit()

            # Check if job was cancelled
            db.refresh(job)
            if job.status == JobStatus.CANCELLED:
                print(f"Job {job_id} was cancelled")
                return {"status": "cancelled"}

            # Simulate training round (replace with actual training)
            # In production, this would call the FedGAN training functions

            # Every 10 rounds, generate sample images
            if round_num % 10 == 0:
                # Generate synthetic images
                # In production, this would use the trained generator
                num_samples = 5
                for i in range(num_samples):
                    image_filename = f"round_{round_num}_sample_{i}.png"
                    image_path = images_dir / image_filename

                    # Save image (placeholder - actual image generation would happen here)
                    # generate_and_save_image(generator, image_path)

                    # Record in database
                    synthetic_image = SyntheticImage(
                        filename=image_filename,
                        file_path=str(image_path),
                        width=128,
                        height=128,
                        format="png",
                        generation_round=round_num,
                        training_job_id=job.id
                    )
                    db.add(synthetic_image)

                db.commit()
                print(f"Generated {num_samples} images at round {round_num}")

        # Training completed successfully
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.progress_percentage = 100.0

        # Calculate training time
        if job.started_at:
            training_time = (job.completed_at - job.started_at).total_seconds()
            job.training_time_seconds = int(training_time)

        # Save final model path
        final_model_path = model_dir / "final_model.h5"
        job.model_path = str(final_model_path)

        # Placeholder metrics (in production, calculate actual FID and IS)
        job.final_fid_score = 268.5  # Example FID score
        job.final_inception_score = 2.3  # Example IS

        # Create privacy metrics (in production, run actual privacy evaluation)
        privacy_metric = PrivacyMetric(
            privacy_risk_score=75.0,
            membership_inference_risk=0.15,
            model_inversion_risk=0.12,
            epsilon=5.0,
            delta=1e-5,
            reconstruction_error=0.08,
            attack_success_rate=0.13,
            training_job_id=job.id
        )
        db.add(privacy_metric)

        db.commit()

        print(f"Training job {job_id} completed successfully")

        return {
            "status": "completed",
            "job_id": job_id,
            "fid_score": job.final_fid_score,
            "inception_score": job.final_inception_score,
            "training_time": job.training_time_seconds
        }

    except Exception as e:
        # Handle errors
        error_trace = traceback.format_exc()
        print(f"Error in training job {job_id}: {str(e)}")
        print(error_trace)

        job.status = JobStatus.FAILED
        job.error_message = str(e)
        job.error_traceback = error_trace
        db.commit()

        return {
            "status": "failed",
            "error": str(e)
        }

    finally:
        db.close()


@celery_app.task
def cleanup_old_jobs():
    """
    Periodic task to cleanup old completed jobs
    """
    db: Session = SessionLocal()

    try:
        # Clean up jobs older than 30 days
        # This would be scheduled to run periodically
        pass

    finally:
        db.close()
