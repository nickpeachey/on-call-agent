# Model Integration Guide

## Overview

This guide explains how machine learning models are trained, saved, and used consistently across the AI On-Call Agent system. The system ensures that models trained in development notebooks are the exact same models used in production.

## Model Lifecycle

### 1. Development and Training

#### Notebook Development (`ml_training_fixed.ipynb`)

The training notebook contains the complete ML pipeline:

```python
class NotebookMLService:
    """ML service specifically for notebook training and experimentation."""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.incident_classifier = None
        self.action_recommender = None  
        self.text_vectorizer = None
        
    def load_or_train_models(self):
        """Load existing models or train new ones if needed."""
        if self._models_exist():
            self.load_models()
        else:
            self.train_models()
            self.save_models()
```

#### Training Process

1. **Data Generation**: Create synthetic incident data
2. **Feature Engineering**: Extract text and numerical features
3. **Model Training**: Train classification and recommendation models
4. **Validation**: Test model accuracy and performance
5. **Persistence**: Save models to disk using joblib

#### Model Files Created

```
models/
├── incident_classifier.joblib      # Incident severity classification
├── action_recommender.joblib       # Action recommendation
└── text_vectorizer.joblib         # Text feature extraction
```

### 2. Production Deployment

#### ML Service (`src/services/ml_service.py`)

The production ML service automatically loads saved models:

```python
class MLService:
    """Production ML service that loads models from disk."""
    
    async def initialize(self):
        """Initialize service and load existing models."""
        await self.load_models()
        
    async def load_models(self):
        """Load models from disk if they exist."""
        
        # Load incident classifier
        classifier_path = self.model_path / "incident_classifier.joblib"
        if classifier_path.exists():
            self.incident_classifier = joblib.load(classifier_path)
            self.model_metadata["incident_classifier"]["loaded"] = True
            
        # Load action recommender  
        recommender_path = self.model_path / "action_recommender.joblib"
        if recommender_path.exists():
            self.action_recommender = joblib.load(recommender_path)
            self.model_metadata["action_recommender"]["loaded"] = True
            
        # Load text vectorizer
        vectorizer_path = self.model_path / "text_vectorizer.joblib"
        if vectorizer_path.exists():
            self.text_vectorizer = joblib.load(vectorizer_path)
```

### 3. AI Decision Engine Integration

#### Unified Model Usage

The AI Decision Engine delegates all ML operations to the ML Service:

```python
class AIDecisionEngine:
    def __init__(self):
        # ML Service integration - single source of truth
        self.ml_service = MLService()
        
    async def start(self):
        # Initialize ML service to load models from disk
        await self.ml_service.initialize()
        
    async def predict_incident_category(self, incident):
        """Predict incident category using ML Service models."""
        # Delegate to ML Service (uses models loaded from disk)
        incident_text = f"{incident.title} {incident.description} service:{incident.service}"
        severity, confidence = await self.ml_service.predict_incident_severity(incident_text)
        return category, confidence
        
    async def _recommend_action_types(self, incident):
        """Recommend actions using ML Service models."""  
        # Delegate to ML Service (uses models loaded from disk)
        incident_text = f"{incident.title} {incident.description} service:{incident.service}"
        action, confidence = await self.ml_service.recommend_action(incident_text)
        return [action]
```

## Model Consistency Verification

### Development Testing

Verify models work in development:

```python
# In notebook
service = NotebookMLService()
service.load_or_train_models()

# Test prediction
test_incident = "Database connection timeout in api-service"
severity, confidence = service.predict_incident_severity(test_incident)
print(f"Predicted: {severity} (confidence: {confidence:.2f})")
```

### Production Testing

Verify same models work in production:

```python
# In production
ml_service = MLService()
await ml_service.initialize()

# Test same prediction
test_incident = "Database connection timeout in api-service"  
severity, confidence = await ml_service.predict_incident_severity(test_incident)
print(f"Production predicted: {severity} (confidence: {confidence:.2f})")
```

### Integration Testing

Test complete AI Decision Engine:

```python
ai_engine = AIDecisionEngine()
await ai_engine.start()

test_incident = IncidentCreate(
    title="Database Connection Timeout",
    description="Application experiencing timeouts when connecting to PostgreSQL", 
    service="api-service",
    severity="high",
    tags=["database", "timeout"]
)

# This uses ML Service models loaded from disk
category, confidence = await ai_engine.predict_incident_category(test_incident)
print(f"AI Engine predicted: {category} (confidence: {confidence:.2f})")
```

## Model Update Process

### 1. Retraining Models

When new training data is available:

```python
# In notebook - retrain with new data
service = NotebookMLService()
service.train_models()  # Train with latest data
service.save_models()   # Overwrite existing models
```

### 2. Production Deployment

Models are automatically picked up on next service restart:

```python
# Production service will load updated models
ml_service = MLService() 
await ml_service.initialize()  # Loads latest models from disk
```

### 3. Zero-Downtime Updates

For production environments, implement graceful model updates:

```python
class MLService:
    async def reload_models(self):
        """Reload models from disk without service restart."""
        await self.load_models()
        logger.info("Models reloaded from disk")
```

## Troubleshooting

### Model Loading Issues

Check if models exist and are loadable:

```python
# Verify model files exist
from pathlib import Path
import joblib

models_dir = Path("models")
required_files = [
    "incident_classifier.joblib",
    "action_recommender.joblib", 
    "text_vectorizer.joblib"
]

for file in required_files:
    path = models_dir / file
    if path.exists():
        try:
            model = joblib.load(path)
            print(f"✅ {file} loaded successfully")
        except Exception as e:
            print(f"❌ {file} failed to load: {e}")
    else:
        print(f"⚠️  {file} not found")
```

### Prediction Failures

Test individual model components:

```python
# Test ML Service initialization
ml_service = MLService()
await ml_service.initialize()

# Check model loading status
for model_name, metadata in ml_service.model_metadata.items():
    print(f"{model_name}: loaded={metadata['loaded']}")

# Test individual predictions
if ml_service.incident_classifier:
    # Test prediction
    pass
```

### Version Compatibility

Ensure scikit-learn versions match between training and production:

```python
# Check scikit-learn version
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

# Models trained with different versions may show warnings
# But should still work for basic operations
```

## Best Practices

### Model Development

1. **Consistent Environment**: Use same Python/scikit-learn versions
2. **Reproducible Training**: Set random seeds for consistent results
3. **Validation**: Always test models before saving
4. **Documentation**: Document model versions and training data

### Production Deployment

1. **Graceful Fallbacks**: Handle model loading failures gracefully
2. **Health Checks**: Verify models loaded correctly on startup
3. **Monitoring**: Track model prediction performance
4. **Backup Models**: Keep previous model versions as backup

### Model Updates

1. **Testing**: Validate new models in staging environment
2. **Gradual Rollout**: Deploy to subset of traffic first
3. **Rollback Plan**: Keep ability to revert to previous models
4. **Performance Monitoring**: Watch for performance regressions

## Configuration

### Model Paths

Configure model storage location:

```python
# In settings
ML_MODEL_PATH = "models"

# In ML Service
self.model_path = Path(settings.ml_model_path)
```

### Model Metadata

Track model information:

```python
model_metadata = {
    "incident_classifier": {
        "loaded": True,
        "accuracy": 0.85,
        "trained_at": "2025-07-31T10:00:00Z"
    },
    "action_recommender": {
        "loaded": True, 
        "accuracy": 0.78,
        "trained_at": "2025-07-31T10:00:00Z"
    }
}
```

This integration ensures that the AI On-Call Agent uses consistent, well-trained models across all environments and provides a reliable foundation for automated incident resolution.
