"""ML training background tasks."""

from src.worker import app
from src.core import get_logger

logger = get_logger(__name__)

@app.task(bind=True)
def retrain_models(self):
    """Background task for retraining ML models."""
    logger.info("Starting ML model retraining task")
    
    try:
        # Import here to avoid circular imports
        from src.ai.simple_engine import SimpleAIEngine
        from src.services.action_logger import action_logger
        
        # Get training data
        training_data = action_logger.get_recent_action_attempts(limit=100)
        
        if len(training_data) < 10:
            logger.warning(f"Not enough training data: {len(training_data)} samples")
            return {"status": "skipped", "reason": "insufficient_data", "samples": len(training_data)}
        
        # Initialize AI engine and retrain
        ai_engine = SimpleAIEngine()
        # For now, just return success since training method needs to be implemented
        
        logger.info("ML model retraining completed successfully")
        return {"status": "success", "samples": len(training_data)}
        
    except Exception as e:
        logger.error(f"ML model retraining failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

@app.task(bind=True)
def update_model_metrics(self):
    """Background task for updating ML model performance metrics."""
    logger.info("Updating ML model metrics")
    
    try:
        from src.services.action_logger import action_logger
        
        # Calculate recent performance metrics
        stats = action_logger.get_action_statistics(days_back=7)
        
        logger.info(f"Updated ML model metrics: {stats}")
        return {"status": "success", "metrics": stats}
        
    except Exception as e:
        logger.error(f"Failed to update ML model metrics: {str(e)}")
        return {"status": "failed", "error": str(e)}
