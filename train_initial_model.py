#!/usr/bin/env python3
"""
Initial AI Model Training Script
Loads sample training data and trains the AI models for the first time.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate, IncidentSeverity, IncidentCategory
from src.core import get_logger

logger = get_logger(__name__)


async def load_sample_training_data(file_path: str = "data/sample_training.json"):
    """Load training data from JSON file."""
    logger.info(f"📚 Loading sample training data from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"❌ Training data file not found: {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"✅ Loaded {len(data)} training samples")
    return data


def convert_to_incident_create(sample: dict) -> IncidentCreate:
    """Convert sample data to IncidentCreate schema."""
    context = sample.get("context", {})
    
    # Map severity
    severity_map = {
        "high": IncidentSeverity.HIGH,
        "medium": IncidentSeverity.MEDIUM,
        "low": IncidentSeverity.LOW,
        "critical": IncidentSeverity.CRITICAL
    }
    severity = severity_map.get(context.get("severity", "medium"), IncidentSeverity.MEDIUM)
    
    # Map category based on incident type
    category_map = {
        "airflow_task_failure": IncidentCategory.WORKFLOW,
        "spark_application_failure": IncidentCategory.PERFORMANCE,
        "database_connection_failure": IncidentCategory.DATABASE,
        "memory_leak": IncidentCategory.PERFORMANCE,
        "service_unavailable": IncidentCategory.AVAILABILITY,
        "network_timeout": IncidentCategory.NETWORK,
        "authentication_failure": IncidentCategory.SECURITY,
        "disk_space_full": IncidentCategory.RESOURCE,
        "api_rate_limit": IncidentCategory.PERFORMANCE,
        "configuration_error": IncidentCategory.CONFIGURATION
    }
    
    incident_type = sample.get("incident_type", "")
    category = category_map.get(incident_type, IncidentCategory.OTHER)
    
    return IncidentCreate(
        title=sample.get("description", f"Sample incident: {incident_type}"),
        description=sample.get("description", ""),
        service=context.get("service", "unknown"),
        severity=severity,
        category=category,
        source="training_data",
        metadata={
            "incident_type": incident_type,
            "original_context": context,
            "sample_id": sample.get("id", "unknown")
        }
    )


async def train_ai_models():
    """Train AI models using sample data."""
    logger.info("🤖 Starting AI model training process")
    
    # Initialize AI engine
    ai_engine = AIDecisionEngine()
    await ai_engine.start()
    
    try:
        # Load sample training data
        training_samples = await load_sample_training_data()
        
        if not training_samples:
            logger.error("❌ No training data available")
            return False
        
        # Add training data to AI engine
        logger.info(f"📊 Adding {len(training_samples)} training samples")
        
        for sample in training_samples:
            try:
                incident = convert_to_incident_create(sample)
                outcome = sample.get("outcome", "success")
                resolution_time = sample.get("feedback", {}).get("time_to_resolution", 300)
                success = outcome == "success"
                confidence_score = sample.get("feedback", {}).get("effectiveness", 0.8)
                
                # Extract actions executed
                resolution_action = sample.get("resolution_action", {})
                actions_executed = []
                if resolution_action:
                    actions_executed = [{
                        "action_type": resolution_action.get("action_type", "unknown"),
                        "parameters": resolution_action.get("parameters", {}),
                        "success": success
                    }]
                
                await ai_engine.add_training_data_async(
                    incident=incident,
                    outcome=outcome,
                    resolution_time=resolution_time,
                    success=success,
                    confidence_score=confidence_score,
                    actions_executed=actions_executed
                )
                
                logger.debug(f"✅ Added training sample: {incident.title[:50]}...")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to add training sample: {str(e)}")
                continue
        
        # Train the models
        logger.info("🔄 Training AI models with loaded data...")
        training_results = await ai_engine.retrain_models(min_samples=5)  # Lower threshold for initial training
        
        if training_results["success"]:
            logger.info("✅ AI model training completed successfully!")
            logger.info(f"📊 Training samples: {training_results.get('training_samples', 0)}")
            
            evaluation = training_results.get("evaluation", {})
            if evaluation:
                logger.info(f"🎯 Classification accuracy: {evaluation.get('classification_accuracy', 0):.2f}")
                logger.info(f"📈 Confidence model R²: {evaluation.get('confidence_r2', 0):.2f}")
                logger.info(f"🎭 Clustering silhouette: {evaluation.get('clustering_silhouette', 0):.2f}")
            
            # Save models to file
            model_dir = Path("data/models")
            model_dir.mkdir(exist_ok=True)
            model_file = model_dir / "ai_model.pkl"
            
            if ai_engine.save_model(str(model_file)):
                logger.info(f"💾 Models saved to {model_file}")
            else:
                logger.warning("⚠️ Failed to save models to file")
            
            return True
        else:
            logger.error(f"❌ AI model training failed: {training_results.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}")
        return False
    
    finally:
        if ai_engine.is_running:
            await ai_engine.stop()


async def main():
    """Main training function."""
    logger.info("🚀 Starting initial AI model training")
    
    success = await train_ai_models()
    
    if success:
        logger.info("🎉 Initial AI model training completed successfully!")
        logger.info("🔄 You can now restart the application to use the trained models")
        return 0
    else:
        logger.error("💥 Initial AI model training failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        sys.exit(1)
