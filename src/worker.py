"""Celery worker for background task processing."""

from celery import Celery
from kombu import Queue

from .core.config import settings
from .core import get_logger

logger = get_logger(__name__)

# Create Celery app
app = Celery('on_call_agent')

# Create alias for Celery command line
celery = app

# Configure Celery
app.conf.update(
    broker_url=settings.redis_url,
    result_backend=settings.redis_url,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Queue configuration
    task_routes={
        'on_call_agent.tasks.ml_training.*': {'queue': 'ml_training'},
        'on_call_agent.tasks.incident_processing.*': {'queue': 'incident_processing'},
        'on_call_agent.tasks.action_execution.*': {'queue': 'action_execution'},
    },
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
)

# Define queues
app.conf.task_routes = {
    'ml_training': Queue('ml_training'),
    'incident_processing': Queue('incident_processing'), 
    'action_execution': Queue('action_execution'),
    'default': Queue('default'),
}

# Auto-discover tasks
app.autodiscover_tasks([
    'src.tasks.ml_training',
    'src.tasks.incident_processing', 
    'src.tasks.action_execution',
])

@app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery worker."""
    logger.info(f"Debug task executed: {self.request!r}")
    return f"Debug task completed at {self.request.id}"

if __name__ == '__main__':
    app.start()
