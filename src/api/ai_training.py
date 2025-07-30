"""API endpoints for AI model training and management."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from datetime import datetime

from ..core import get_logger
from ..models.schemas import IncidentCreate
from ..ai import AIDecisionEngine


logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/ai", tags=["AI Training"])

# Global AI engine instance (will be injected)
_ai_engine: Optional[AIDecisionEngine] = None


def get_ai_engine() -> AIDecisionEngine:
    """Get the AI engine instance."""
    if _ai_engine is None:
        raise HTTPException(status_code=503, detail="AI engine not initialized")
    return _ai_engine


def set_ai_engine(ai_engine: AIDecisionEngine):
    """Set the AI engine instance."""
    global _ai_engine
    _ai_engine = ai_engine


@router.post("/retrain")
async def retrain_models(
    min_samples: int = 50,
    ai_engine: AIDecisionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    Retrain AI models with latest data from database.
    
    Args:
        min_samples: Minimum number of training samples required
        
    Returns:
        Training results including success status, sample count, and evaluation metrics
    """
    logger.info("🔄 Manual retrain requested", min_samples=min_samples)
    
    try:
        results = await ai_engine.retrain_models(min_samples=min_samples)
        
        if results["success"]:
            logger.info("✅ Manual retrain completed successfully", 
                       samples=results["training_samples"],
                       accuracy=results.get("evaluation", {}).get("classification_accuracy", 0))
        else:
            logger.warning("⚠️ Manual retrain failed", error=results.get("error"))
        
        return {
            "status": "success" if results["success"] else "failed",
            "message": "Models retrained successfully" if results["success"] else results.get("error"),
            "training_samples": results.get("training_samples", 0),
            "evaluation": results.get("evaluation", {}),
            "metadata": results.get("metadata", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("❌ Error during manual retrain", error=str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/model-status")
async def get_model_status(
    ai_engine: AIDecisionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    Get current AI model status and metadata.
    
    Returns:
        Model status including training state, accuracy, and metadata
    """
    try:
        return {
            "models_trained": {
                "incident_classifier": ai_engine.incident_classifier is not None,
                "confidence_model": ai_engine.confidence_model is not None,
                "pattern_clustering": ai_engine.pattern_clustering is not None
            },
            "training_data_samples": len(ai_engine.training_data.get("incidents", [])),
            "metadata": ai_engine.model_metadata,
            "is_running": ai_engine.is_running,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("❌ Error getting model status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.post("/add-training-data")
async def add_training_data(
    incident: IncidentCreate,
    outcome: str,
    resolution_time: float,
    success: bool,
    confidence_score: Optional[float] = None,
    actions_executed: Optional[list] = None,
    ai_engine: AIDecisionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    Manually add training data for an incident resolution.
    
    Args:
        incident: The incident data
        outcome: Resolution outcome ("resolved", "escalated", etc.)
        resolution_time: Time taken to resolve in seconds
        success: Whether resolution was successful
        confidence_score: AI confidence at time of resolution
        actions_executed: List of actions that were executed
        
    Returns:
        Success status and updated sample count
    """
    try:
        await ai_engine.add_training_data_async(
            incident=incident,
            outcome=outcome,
            resolution_time=resolution_time,
            success=success,
            confidence_score=confidence_score,
            actions_executed=actions_executed or []
        )
        
        return {
            "status": "success",
            "message": "Training data added successfully",
            "total_samples": len(ai_engine.training_data.get("incidents", [])),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("❌ Error adding training data", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to add training data: {str(e)}")


@router.get("/training-stats")
async def get_training_statistics(
    ai_engine: AIDecisionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    Get training data statistics and trends.
    
    Returns:
        Statistics about training data and model performance
    """
    try:
        training_data = ai_engine.training_data.get("incidents", [])
        
        if not training_data:
            return {
                "total_samples": 0,
                "success_rate": 0.0,
                "average_resolution_time": 0,
                "recent_trends": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Calculate statistics
        successful = [sample for sample in training_data if sample.get("success", False)]
        success_rate = len(successful) / len(training_data) if training_data else 0
        
        resolution_times = [sample.get("resolution_time", 0) for sample in training_data]
        avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        
        # Recent trends (last 10 vs all)
        recent_data = training_data[-10:] if len(training_data) >= 10 else training_data
        recent_successful = [sample for sample in recent_data if sample.get("success", False)]
        recent_success_rate = len(recent_successful) / len(recent_data) if recent_data else 0
        
        # Outcome distribution
        outcomes = {}
        for sample in training_data:
            outcome = sample.get("outcome", "unknown")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        return {
            "total_samples": len(training_data),
            "success_rate": success_rate,
            "recent_success_rate": recent_success_rate,
            "average_resolution_time": avg_resolution_time,
            "outcome_distribution": outcomes,
            "models_available": {
                "incident_classifier": ai_engine.incident_classifier is not None,
                "confidence_model": ai_engine.confidence_model is not None,
                "pattern_clustering": ai_engine.pattern_clustering is not None
            },
            "metadata": ai_engine.model_metadata,
            "recent_trends": {
                "improving": recent_success_rate > success_rate,
                "trend_direction": "up" if recent_success_rate > success_rate else "down"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("❌ Error getting training statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get training statistics: {str(e)}")
