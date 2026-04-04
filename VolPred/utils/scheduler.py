"""APScheduler setup for background tasks."""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime

from utils.config import (
    DAILY_PREDICTION_HOUR,
    DAILY_PREDICTION_MINUTE,
    RETRAIN_INTERVAL_DAYS,
    RETRAIN_HOUR,
)
from utils.logger import setup_logger
from utils.tasks import daily_predict_task, retrain_task_10d

logger = setup_logger("scheduler")


class BackgroundTaskScheduler:
    """Manages APScheduler for background tasks."""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        
    def start(self):
        """Start the scheduler and register all jobs."""
        try:
            # Daily prediction task - every day at specified UTC hour
            self.scheduler.add_job(
                daily_predict_task,
                CronTrigger(hour=DAILY_PREDICTION_HOUR, minute=DAILY_PREDICTION_MINUTE),
                id="daily_prediction",
                name="Daily Volatility Prediction",
                replace_existing=True,
                misfire_grace_time=900,  # 15 minutes grace period if missed
            )
            logger.info(
                f"Registered daily prediction task at "
                f"{DAILY_PREDICTION_HOUR:02d}:{DAILY_PREDICTION_MINUTE:02d} UTC"
            )
            
            # 10-day retraining task - every 10 days
            # Note: IntervalTrigger runs every N days from first execution
            self.scheduler.add_job(
                retrain_task_10d,
                IntervalTrigger(days=RETRAIN_INTERVAL_DAYS),
                id="retrain_10d",
                name="10-Day Model Retraining",
                replace_existing=True,
                misfire_grace_time=900,
            )
            logger.info(
                f"Registered 10-day retraining task "
                f"(interval: {RETRAIN_INTERVAL_DAYS} days)"
            )
            
            # Start scheduler
            self.scheduler.start()
            logger.info("Background task scheduler started")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}", exc_info=True)
            raise
    
    def stop(self):
        """Stop the scheduler gracefully."""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)
                logger.info("Background task scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {str(e)}", exc_info=True)
    
    def get_jobs(self):
        """Get list of registered jobs."""
        return self.scheduler.get_jobs()
    
    def get_job_status(self, job_id: str):
        """Get status of a specific job."""
        job = self.scheduler.get_job(job_id)
        if job:
            return {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            }
        return None


# Global scheduler instance
_scheduler = None


def get_scheduler() -> BackgroundTaskScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundTaskScheduler()
    return _scheduler


def init_scheduler():
    """Initialize and start the scheduler."""
    scheduler = get_scheduler()
    scheduler.start()
    return scheduler


def shutdown_scheduler():
    """Shutdown the scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None
