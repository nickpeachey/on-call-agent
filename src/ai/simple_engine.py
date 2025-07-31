"""
Simplified AI Decision Engine
Uses pre-trained models from pickle files and database entries for retraining.
"""

import asyncio
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core import get_logger
from ..models.schemas import IncidentCreate, Severity

logger = get_logger(__name__)


class SimpleAIEngine:
    """Simplified AI Decision Engine."""
    
    def __init__(self):
        self.is_running = False
        self.model = None
        self.vectorizer = None
        self.model_metadata = {}
        self.incident_queue = asyncio.Queue()
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self, model_path: str = "data/models/ai_model.pkl") -> bool:
        """Load trained model from file."""
        try:
            if not Path(model_path).exists():
                logger.warning(f"📭 No trained model found at {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get("model")
            self.vectorizer = model_data.get("vectorizer")
            self.model_metadata = {
                "accuracy": model_data.get("accuracy", 0),
                "training_samples": model_data.get("training_samples", 0),
                "trained_at": model_data.get("trained_at"),
                "feature_names": model_data.get("feature_names", [])
            }
            
            logger.info("🎯 AI model loaded successfully", 
                       accuracy=self.model_metadata["accuracy"],
                       samples=self.model_metadata["training_samples"])
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    async def start(self):
        """Start the AI engine."""
        if not self.is_running:
            self.is_running = True
            logger.info("🤖 Simple AI Engine started")
            
            if not self.model:
                logger.warning("⚠️ No AI model loaded - train a model first!")
    
    async def stop(self):
        """Stop the AI engine."""
        if self.is_running:
            self.is_running = False
            logger.info("🛑 Simple AI Engine stopped")
    
    def extract_features(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Extract features from an incident for prediction."""
        return {
            "incident_type": "unknown",  # Could be inferred from title/description
            "service": incident.service,
            "severity": incident.severity.value,
            "has_dag_info": "dag" in incident.title.lower() or "dag" in incident.description.lower(),
            "has_task_info": "task" in incident.title.lower() or "task" in incident.description.lower(),
            "error_length": len(incident.description),
            "resolution_action": "unknown"  # This would be predicted
        }
    
    def predict_resolution_confidence(self, incident: IncidentCreate) -> float:
        """Predict confidence for automated resolution."""
        if not self.model or not self.vectorizer:
            logger.warning("⚠️ No model available for prediction")
            return 0.3  # Default low confidence
        
        try:
            # Extract features
            features = self.extract_features(incident)
            
            # Transform features using list of dictionaries
            X = self.vectorizer.transform([features])
            
            # Get prediction probability
            probabilities = self.model.predict_proba(X)
            
            # Return confidence for successful resolution (class 1)
            confidence = probabilities[0][1] if len(probabilities[0]) > 1 else 0.3
            
            logger.debug(f"🎯 Predicted resolution confidence: {confidence:.2f}")
            return confidence
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            return 0.3
    
    def should_automate_resolution(self, incident: IncidentCreate, threshold: float = 0.6) -> bool:
        """Determine if incident should be automatically resolved."""
        confidence = self.predict_resolution_confidence(incident)
        should_automate = confidence >= threshold
        
        logger.info(f"🤔 Automation decision", 
                   confidence=f"{confidence:.2f}",
                   threshold=threshold,
                   should_automate=should_automate,
                   incident_title=incident.title[:50] + "...")
        
        return should_automate
    
    async def retrain_from_database(self, min_samples: int = 20) -> Dict[str, Any]:
        """Retrain model using incidents and resolutions from database."""
        try:
            from ..database import get_database
            from sqlalchemy import text
            
            # Get database session
            db = await get_database()
            
            # Query successful incident resolutions
            query = text("""
                SELECT i.title, i.description, i.service, i.severity,
                       r.success, r.resolution_time, r.actions_executed
                FROM incidents i
                JOIN incident_resolutions r ON i.id = r.incident_id
                WHERE r.created_at > datetime('now', '-30 days')
                ORDER BY r.created_at DESC
                LIMIT 1000
            """)
            
            result = await db.execute(query)
            rows = result.fetchall()
            
            if len(rows) < min_samples:
                return {
                    "success": False,
                    "error": f"Insufficient data: {len(rows)} samples (need {min_samples})"
                }
            
            # Prepare training data
            features = []
            labels = []
            
            for row in rows:
                feature = {
                    "incident_type": "database_incident",
                    "service": row.service,
                    "severity": row.severity,
                    "has_dag_info": "dag" in row.title.lower(),
                    "has_task_info": "task" in row.title.lower(),
                    "error_length": len(row.description),
                    "resolution_action": "automated" if row.actions_executed else "manual"
                }
                
                label = 1 if row.success else 0
                
                features.append(feature)
                labels.append(label)
            
            # Retrain model
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
            
            vectorizer = DictVectorizer()
            X = vectorizer.fit_transform(features)
            y = labels
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Update model
            self.model = model
            self.vectorizer = vectorizer
            self.model_metadata.update({
                "accuracy": accuracy,
                "training_samples": len(features),
                "trained_at": datetime.utcnow().isoformat(),
                "retrained_from_db": True
            })
            
            # Save updated model
            model_data = {
                "model": self.model,
                "vectorizer": self.vectorizer,
                "accuracy": accuracy,
                "training_samples": len(features),
                "trained_at": datetime.utcnow().isoformat()
            }
            
            model_path = "data/models/ai_model.pkl"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("✅ Model retrained successfully", 
                       samples=len(features),
                       accuracy=f"{accuracy:.2f}")
            
            return {
                "success": True,
                "training_samples": len(features),
                "accuracy": accuracy,
                "retrained_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Retraining failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            "model_loaded": self.model is not None,
            "vectorizer_loaded": self.vectorizer is not None,
            "is_running": self.is_running,
            "metadata": self.model_metadata
        }
