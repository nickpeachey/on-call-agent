"""
Celery worker configuration for AI On-Call Agent.
"""
import os
from celery import Celery

# Get Redis URL from environment
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
app = Celery("oncall_agent", broker=redis_url, backend=redis_url)

# Configure Celery
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "src.tasks.*": {"queue": "oncall_tasks"},
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
)

# Auto-discover tasks
app.autodiscover_tasks(["src.tasks"])

if __name__ == "__main__":
    app.start()
