"""Simple API endpoints for AI model training and status."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime

from ..core import get_logger
from ..ai.simple_engine import SimpleAIEngine

logger = get_logger(__name__)
router = APIRouter(prefix="/ai", tags=["AI Training"])

# Global AI engine instance (will be injected)
_ai_engine: SimpleAIEngine | None = None


def set_ai_engine(ai_engine: SimpleAIEngine):
    """Set the AI engine instance."""
    global _ai_engine
    _ai_engine = ai_engine


def get_ai_engine() -> SimpleAIEngine:
    """Get the AI engine instance."""
    if _ai_engine is None:
        raise HTTPException(status_code=503, detail="AI engine not initialized")
    return _ai_engine


@router.get("/model-status")
async def get_model_status() -> Dict[str, Any]:
    """Get current AI model status."""
    try:
        ai_engine = get_ai_engine()
        status = ai_engine.get_model_status()
        
        return {
            "status": "initialized" if status["model_loaded"] else "not_initialized",
            "model_loaded": status["model_loaded"],
            "is_running": status["is_running"],
            "metadata": status["metadata"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.post("/retrain")
async def retrain_model(min_samples: int = 1000) -> Dict[str, Any]:
    """Retrain AI model with latest database data."""
    try:
        ai_engine = get_ai_engine()
        
        logger.info("🔄 Manual retrain requested", min_samples=min_samples)
        
        results = await ai_engine.retrain_from_database(min_samples=min_samples)
        
        if results["success"]:
            logger.info("✅ Manual retrain completed successfully",
                       samples=results.get("training_samples", 0),
                       accuracy=results.get("accuracy", 0))
        else:
            logger.warning("⚠️ Manual retrain failed", error=results.get("error"))
        
        return {
            "status": "success" if results["success"] else "failed",
            "message": "Model retrained successfully" if results["success"] else results.get("error"),
            "training_samples": results.get("training_samples", 0),
            "accuracy": results.get("accuracy", 0),
            "retrained_at": results.get("retrained_at"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error during manual retrain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/training-stats")
async def get_training_stats() -> Dict[str, Any]:
    """Get basic training statistics."""
    try:
        ai_engine = get_ai_engine()
        status = ai_engine.get_model_status()
        
        return {
            "model_available": status["model_loaded"],
            "metadata": status["metadata"],
            "last_trained": status["metadata"].get("trained_at"),
            "training_samples": status["metadata"].get("training_samples", 0),
            "accuracy": status["metadata"].get("accuracy", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting training stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get training stats: {str(e)}")
