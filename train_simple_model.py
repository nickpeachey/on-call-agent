#!/usr/bin/env python3
"""
Simplified AI Model Training Script
Uses sample_training.json for initial training, then database entries for retraining.
"""

import asyncio
import json
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import get_logger

logger = get_logger(__name__)


def load_sample_training_data() -> List[Dict[str, Any]]:
    """Load training data from sample_training.json."""
    training_file = Path("data/sample_training.json")
    
    if not training_file.exists():
        logger.error(f"❌ Training file not found: {training_file}")
        return []
    
    with open(training_file, 'r') as f:
        data = json.load(f)
    
    logger.info(f"📚 Loaded {len(data)} training samples from {training_file}")
    return data


def prepare_training_features(samples: List[Dict[str, Any]]) -> tuple:
    """Extract features and labels from training samples."""
    features = []
    labels = []
    
    for sample in samples:
        # Extract features from incident context
        context = sample.get("context", {})
        
        # Create feature vector
        feature = {
            "incident_type": sample.get("incident_type", "unknown"),
            "service": context.get("service", "unknown"),
            "severity": context.get("severity", "medium"),
            "has_dag_info": bool(context.get("dag_id")),
            "has_task_info": bool(context.get("task_id")),
            "error_length": len(context.get("error_message", "")),
            "resolution_action": sample.get("resolution_action", {}).get("action_type", "unknown")
        }
        
        # Label is success/failure
        label = 1 if sample.get("outcome") == "success" else 0
        
        features.append(feature)
        labels.append(label)
    
    return features, labels


def train_simple_model(features: List[Dict], labels: List[int]) -> Dict[str, Any]:
    """Train a simple model for incident resolution prediction."""
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # Convert features to numerical format
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(features)
    y = labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"🎯 Model training completed with accuracy: {accuracy:.2f}")
    
    # Return trained components
    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "feature_names": vectorizer.get_feature_names_out(),
        "training_samples": len(features),
        "trained_at": datetime.utcnow().isoformat()
    }


def save_trained_model(model_data: Dict[str, Any], model_path: str) -> bool:
    """Save the trained model to file."""
    try:
        # Ensure models directory exists
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save model: {str(e)}")
        return False


def load_trained_model(model_path: str) -> Dict[str, Any] | None:
    """Load a trained model from file."""
    try:
        if not Path(model_path).exists():
            return None
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info(f"📥 Model loaded from {model_path}")
        return model_data
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return None


async def main():
    """Main training function."""
    logger.info("🚀 Starting simplified AI model training")
    
    # Check if model already exists
    model_path = "data/models/ai_model.pkl"
    existing_model = load_trained_model(model_path)
    
    if existing_model:
        logger.info(f"✅ Model already exists (accuracy: {existing_model.get('accuracy', 0):.2f})")
        logger.info("🔄 Use the retrain API endpoint to update with new data")
        return 0
    
    # Load training data
    samples = load_sample_training_data()
    if not samples:
        logger.error("❌ No training data available")
        return 1
    
    # Prepare features and labels
    features, labels = prepare_training_features(samples)
    
    if len(features) < 5:
        logger.error("❌ Insufficient training data (need at least 5 samples)")
        return 1
    
    # Train model
    logger.info(f"🧠 Training model with {len(features)} samples")
    model_data = train_simple_model(features, labels)
    
    # Save model
    if save_trained_model(model_data, model_path):
        logger.info("🎉 Initial model training completed successfully!")
        logger.info(f"📊 Model accuracy: {model_data['accuracy']:.2f}")
        logger.info(f"📈 Training samples: {model_data['training_samples']}")
        logger.info("🔄 Restart the application to use the trained model")
        return 0
    else:
        logger.error("💥 Failed to save trained model")
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
