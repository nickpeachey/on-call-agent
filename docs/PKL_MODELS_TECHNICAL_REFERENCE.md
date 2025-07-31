# PKL Models Implementation Reference

## File Locations Using PKL Models

This document provides a technical reference for PKL model usage across the AI On-Call Agent codebase.

### Core Implementation Files

| File | Purpose | PKL Operations | Model Usage |
|------|---------|----------------|-------------|
| `src/ai/__init__.py` | Main AI Engine | `save_model()`, `load_model()` | Incident classification, confidence prediction, anomaly detection |
| `cli.py` | Command Line Interface | Model loading for evaluation | Training, testing, model inspection |
| `scripts/train_models.py` | Training Script | Model saving with metadata | Automated training pipeline |
| `scripts/evaluate_model.py` | Evaluation Script | Model loading for testing | Performance evaluation, benchmarking |
| `scripts/test_continuous_learning.py` | Learning Tests | AI engine initialization | Testing model adaptation |

### Model Loading Points

#### 1. AI Engine Initialization
```python
# File: src/ai/__init__.py, Line: 34
def __init__(self, model_path: Optional[str] = None):
    # ...model component initialization...
    if model_path:
        self.load_model(model_path)  # Line: 68
```

#### 2. Runtime Analysis
```python
# File: src/ai/__init__.py, Line: 226
if self.incident_classifier and self.confidence_model:
    # ML-based analysis using loaded models
    root_cause_category, category_confidence = self.predict_incident_category(incident)
    resolution_confidence = self.predict_resolution_confidence(incident)
    anomaly_info = self.detect_anomalies(incident)
```

#### 3. CLI Model Operations
```python
# File: cli.py, Line: 568
if not ai_engine.load_model(model_file):
    console.print(f"❌ Failed to load model: {model_file}", style="red")
```

### PKL File Structure in Codebase

```
models/
├── first_model.pkl      # Initial trained model (316KB)
├── latest.pkl          # Symlink to first_model.pkl
└── trained/            # Future versioned models
    └── ai_model_*.pkl
```

### Model Components Serialization

#### Save Operation (src/ai/__init__.py:1418)
```python
def save_model(self, file_path: str) -> bool:
    """Save trained models to file."""
    model_data = {
        "incident_classifier": self.incident_classifier,      # RandomForestClassifier
        "confidence_model": self.confidence_model,            # CalibratedClassifierCV  
        "pattern_clustering": self.pattern_clustering,        # dict with PCA + KMeans
        "feature_vectorizer": self.feature_vectorizer,        # TfidfVectorizer
        "label_encoders": self.label_encoders,               # dict of LabelEncoders
        "model_metadata": self.model_metadata,               # training info
        "training_data": self.training_data                  # optional historical data
    }
    
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)  # Line: 1436
```

#### Load Operation (src/ai/__init__.py:1445)
```python
def load_model(self, file_path: str) -> bool:
    """Load trained models from file."""
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)  # Line: 1453
    
    # Load model components
    self.incident_classifier = model_data.get("incident_classifier")
    self.confidence_model = model_data.get("confidence_model")
    self.pattern_clustering = model_data.get("pattern_clustering")
    self.feature_vectorizer = model_data.get("feature_vectorizer")
    self.label_encoders = model_data.get("label_encoders", {})
    self.model_metadata = model_data.get("model_metadata", {})
```

### Prediction Methods Using Loaded Models

#### Incident Classification (src/ai/__init__.py:1336)
```python
def predict_incident_category(self, incident: IncidentCreate) -> Tuple[str, float]:
    """Predict incident category using trained model."""
    if not self.incident_classifier or not self.feature_vectorizer:
        return self._determine_root_cause_category(incident), 0.5
    
    # Extract features and vectorize
    features = self._extract_ml_features(incident)
    X = self._vectorize_features([features])
    
    # Use loaded classifier
    y_pred = self.incident_classifier.predict(X)[0]  # Line: 1348
    y_proba = self.incident_classifier.predict_proba(X)[0]
    
    # Decode using loaded label encoder
    category = self.label_encoders["category"].inverse_transform([y_pred])[0]
    confidence = float(np.max(y_proba))
    
    return category, confidence
```

#### Confidence Prediction (src/ai/__init__.py:1361)
```python
def predict_resolution_confidence(self, incident: IncidentCreate) -> float:
    """Predict confidence for automated resolution."""
    if not self.confidence_model or not self.feature_vectorizer:
        return 0.5
    
    features = self._extract_ml_features(incident)
    X = self._vectorize_features([features])
    
    # Use loaded confidence model
    confidence_proba = self.confidence_model.predict_proba(X)[0]
    confidence = float(confidence_proba[1]) if len(confidence_proba) > 1 else 0.5
    
    return confidence
```

### Training Pipeline PKL Operations

#### Training Script (scripts/train_models.py)
```python
# Line: 166
if ai_engine.save_model(output_path):
    console.print(f"✅ Model saved: {output_path}")
    
    # Create latest.pkl symlink
    latest_path = Path(output_path).parent / "latest.pkl"  # Line: 170
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(Path(output_path).name)
```

#### CLI Training Command (cli.py)
```python
# Line: 499
if ai_engine.save_model(output_file):
    console.print(f"✅ Model saved to {output_file}")
    
    # Create latest.pkl symlink
    latest_path = Path(output_file).parent / "latest.pkl"  # Line: 503
```

### Model State Detection

The AI engine checks for loaded models before making predictions:

```python
# Runtime model availability check
if self.incident_classifier and self.confidence_model:
    # Use ML models for analysis
    ml_analysis = {
        "analysis_method": "ml_models",
        # ... ML predictions
    }
else:
    # Fallback to rule-based analysis
    analysis = {
        "analysis_method": "rule_based",
        # ... heuristic analysis
    }
```

### Current Production Status

#### Main Application (src/main.py:35)
```python
# Currently NO models are loaded in production
ai_engine = AIDecisionEngine()  # model_path=None
```

This means the system currently operates in **fallback mode** using rule-based analysis.

#### To Enable Models in Production
```python
# Modify src/main.py to auto-load models
ai_engine = AIDecisionEngine(model_path="models/latest.pkl")
```

### Error Handling and Fallbacks

All prediction methods include fallback behavior:

```python
def predict_incident_category(self, incident: IncidentCreate) -> Tuple[str, float]:
    if not self.incident_classifier or not self.feature_vectorizer:
        # Fallback to rule-based classification
        return self._determine_root_cause_category(incident), 0.5
    
    try:
        # ML prediction logic
        return category, confidence
    except Exception as e:
        logger.error("Error predicting incident category", error=str(e))
        # Fallback with lower confidence
        return self._determine_root_cause_category(incident), 0.3
```

### CLI Model Commands

| Command | File | Purpose |
|---------|------|---------|
| `ml train` | cli.py:480 | Train new models and save to PKL |
| `ml evaluate` | cli.py:560 | Load PKL and evaluate performance |
| `ml model-info` | cli.py:720 | Load PKL and display metadata |
| `ml test-decision` | cli.py:780 | Load PKL and test predictions |

### Import Dependencies

Required imports for PKL operations:

```python
# src/ai/__init__.py:4
import pickle

# Dependencies for model components
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
```

### Model Lifecycle

1. **Training**: `scripts/train_models.py` → Save to PKL
2. **Deployment**: Copy PKL to production location
3. **Loading**: AI engine loads on initialization
4. **Prediction**: Models used for real-time analysis
5. **Evaluation**: `scripts/evaluate_model.py` loads for testing
6. **Retraining**: New data → retrain → save new PKL

### Performance Implications

- **Model Loading**: ~1-2 seconds for 316KB PKL file
- **Prediction Time**: <50ms per incident analysis
- **Memory Usage**: ~50-100MB for loaded models
- **Fallback Latency**: Rule-based analysis ~5-10ms

### Security Considerations

- PKL files can execute arbitrary code during unpickling
- Only load PKL files from trusted sources
- Consider using `joblib` for safer serialization
- Validate model checksums in production

### Monitoring and Logging

Model operations are logged with structured logging:

```python
logger.info("Model loaded successfully", 
           file_path=file_path,
           version=self.model_metadata.get("version", "unknown"),
           samples=self.model_metadata.get("training_samples", 0))
```

Track model performance metrics:
- Prediction accuracy over time
- Model loading success/failure rates
- Fallback usage frequency
- Response time metrics
