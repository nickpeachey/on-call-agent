# PKL Models Training and Usage Guide

## Overview

This guide documents the training, storage, and usage of machine learning models (PKL files) in the AI On-Call Agent system. The PKL files contain trained scikit-learn models that enable intelligent incident classification, resolution confidence prediction, and anomaly detection.

## 📁 Model Architecture

### Model Components Stored in PKL Files

Each PKL file contains a serialized dictionary with the following components:

```python
model_data = {
    "incident_classifier": RandomForestClassifier,      # Incident categorization
    "confidence_model": CalibratedClassifierCV,        # Resolution success prediction
    "pattern_clustering": dict,                        # Anomaly detection (KMeans + PCA)
    "feature_vectorizer": TfidfVectorizer,            # Text feature extraction
    "label_encoders": dict,                           # Category label encoding/decoding
    "model_metadata": dict,                           # Training info, accuracy, timestamps
    "training_data": dict                             # Optional: historical training data
}
```

### Model Purposes

1. **Incident Classifier**: Categorizes incidents into types (database_connectivity, memory_issues, workflow_failure, etc.)
2. **Confidence Model**: Predicts the likelihood of successful automated resolution
3. **Pattern Clustering**: Detects anomalous incidents that deviate from normal patterns
4. **Feature Vectorizer**: Converts incident text into numerical features for ML models

## 🎯 Model Training Process

### 1. Data Collection

Training data is collected from resolved incidents:

```python
def add_training_data(self, incident: IncidentCreate, outcome: str, 
                     resolution_time: float, success: bool):
    """Add training data from resolved incidents."""
    features = self._extract_ml_features(incident)
    training_sample = {
        "incident": incident.dict(),
        "features": features,
        "outcome": outcome,
        "resolution_time": resolution_time,
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }
    self.training_data["incidents"].append(training_sample)
```

### 2. Feature Extraction

Features are extracted from incidents for ML training:

```python
def _extract_ml_features(self, incident: IncidentCreate) -> Dict[str, Any]:
    """Extract ML features from incident for training."""
    features = {
        # Text features
        "title_length": len(incident.title),
        "description_length": len(incident.description),
        "text_content": f"{incident.title} {incident.description}",
        
        # Categorical features
        "service": incident.service,
        "severity": incident.severity,
        
        # Pattern-based features
        "has_timeout": "timeout" in incident.description.lower(),
        "has_memory_issue": "memory" in incident.description.lower(),
        "has_connection_issue": "connection" in incident.description.lower(),
        "has_database_issue": "database" in incident.description.lower(),
        "has_spark_issue": "spark" in incident.description.lower(),
        "has_airflow_issue": "airflow" in incident.description.lower(),
        
        # Error patterns
        "error_patterns": self._extract_error_patterns(incident),
        "num_error_patterns": len(self._extract_error_patterns(incident)),
        
        # Time-based features
        "hour_of_day": datetime.utcnow().hour,
        "day_of_week": datetime.utcnow().weekday(),
    }
    return features
```

### 3. Model Training

Three models are trained simultaneously:

#### A. Incident Classifier
```python
def _train_incident_classifier(self, X: np.ndarray, y_category: np.ndarray):
    """Train incident classification model."""
    # Encode categorical labels
    self.label_encoders["category"] = LabelEncoder()
    y_encoded = self.label_encoders["category"].fit_transform(y_category)
    
    # Train Random Forest classifier
    self.incident_classifier = Pipeline([
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    self.incident_classifier.fit(X, y_encoded)
```

#### B. Confidence Model
```python
def _train_confidence_model(self, X: np.ndarray, y_success: np.ndarray, y_confidence: np.ndarray):
    """Train confidence prediction model."""
    base_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    self.confidence_model = CalibratedClassifierCV(base_classifier, method='isotonic', cv=3)
    self.confidence_model.fit(X, y_success)
```

#### C. Pattern Clustering
```python
def _train_pattern_clustering(self, X: np.ndarray):
    """Train pattern clustering model for anomaly detection."""
    # PCA for dimensionality reduction
    n_components = min(50, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    # K-means clustering
    n_clusters = min(10, len(X) // 5, 3)
    self.pattern_clustering = {
        'pca': pca,
        'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
        'n_clusters': n_clusters
    }
    self.pattern_clustering['kmeans'].fit(X_reduced)
```

## 💾 Model Storage and Management

### File Structure

```
models/
├── first_model.pkl          # Initial trained model (316KB)
├── latest.pkl              # Symlink to current production model
└── trained/
    ├── ai_model_v1.pkl     # Versioned models
    ├── ai_model_v2.pkl
    └── backup/
        ├── current.pkl     # Production backup
        └── backup_20250730_123456.pkl
```

### Saving Models

```python
def save_model(self, file_path: str) -> bool:
    """Save trained models to file."""
    try:
        model_data = {
            "incident_classifier": self.incident_classifier,
            "confidence_model": self.confidence_model,
            "pattern_clustering": self.pattern_clustering,
            "feature_vectorizer": self.feature_vectorizer,
            "label_encoders": self.label_encoders,
            "model_metadata": self.model_metadata,
            "training_data": self.training_data
        }
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with pickle
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Create latest.pkl symlink
        latest_path = Path(file_path).parent / "latest.pkl"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(Path(file_path).name)
        
        return True
    except Exception as e:
        logger.error("Error saving model", error=str(e))
        return False
```

### Loading Models

```python
def load_model(self, file_path: str) -> bool:
    """Load trained models from file."""
    try:
        if not Path(file_path).exists():
            logger.warning("Model file not found", file_path=file_path)
            return False
        
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load model components
        self.incident_classifier = model_data.get("incident_classifier")
        self.confidence_model = model_data.get("confidence_model")
        self.pattern_clustering = model_data.get("pattern_clustering")
        self.feature_vectorizer = model_data.get("feature_vectorizer")
        self.label_encoders = model_data.get("label_encoders", {})
        self.model_metadata = model_data.get("model_metadata", {})
        
        logger.info("Model loaded successfully", 
                   version=self.model_metadata.get("version", "unknown"),
                   samples=self.model_metadata.get("training_samples", 0))
        return True
    except Exception as e:
        logger.error("Error loading model", error=str(e))
        return False
```

## 🚀 Model Usage in Production

### 1. Runtime Prediction

When an incident occurs, the loaded models are used for real-time analysis:

```python
async def _analyze_incident(self, incident: IncidentCreate) -> Dict[str, Any]:
    """Analyze incident using AI/ML to extract key information."""
    try:
        # Use ML models if available
        if self.incident_classifier and self.confidence_model:
            # ML-based analysis
            root_cause_category, category_confidence = self.predict_incident_category(incident)
            resolution_confidence = self.predict_resolution_confidence(incident)
            anomaly_info = self.detect_anomalies(incident)
            
            ml_analysis = {
                "root_cause_category": root_cause_category,
                "category_confidence": category_confidence,
                "confidence_score": resolution_confidence,
                "anomaly_detection": anomaly_info,
                "analysis_method": "ml_models"
            }
            return ml_analysis
    except Exception as e:
        logger.error("Error in ML analysis, falling back to rule-based", error=str(e))
    
    # Fallback to rule-based analysis if models unavailable
    return self._rule_based_analysis(incident)
```

### 2. Incident Classification

```python
def predict_incident_category(self, incident: IncidentCreate) -> Tuple[str, float]:
    """Predict incident category using trained model."""
    if not self.incident_classifier or not self.feature_vectorizer:
        return self._determine_root_cause_category(incident), 0.5
    
    try:
        # Extract features
        features = self._extract_ml_features(incident)
        X = self._vectorize_features([features])
        
        # Predict category
        y_pred = self.incident_classifier.predict(X)[0]
        y_proba = self.incident_classifier.predict_proba(X)[0]
        
        # Decode label
        category = self.label_encoders["category"].inverse_transform([y_pred])[0]
        confidence = float(np.max(y_proba))
        
        return category, confidence
    except Exception as e:
        logger.error("Error predicting incident category", error=str(e))
        return self._determine_root_cause_category(incident), 0.3
```

### 3. Resolution Confidence Prediction

```python
def predict_resolution_confidence(self, incident: IncidentCreate) -> float:
    """Predict confidence for automated resolution."""
    if not self.confidence_model or not self.feature_vectorizer:
        return 0.5  # Default confidence
    
    try:
        features = self._extract_ml_features(incident)
        X = self._vectorize_features([features])
        
        # Predict confidence (probability of successful resolution)
        confidence_proba = self.confidence_model.predict_proba(X)[0]
        confidence = float(confidence_proba[1]) if len(confidence_proba) > 1 else 0.5
        
        return confidence
    except Exception as e:
        logger.error("Error predicting resolution confidence", error=str(e))
        return 0.3
```

### 4. Anomaly Detection

```python
def detect_anomalies(self, incident: IncidentCreate) -> Dict[str, Any]:
    """Detect if incident is anomalous compared to training data."""
    if not self.pattern_clustering:
        return {"is_anomaly": False, "anomaly_score": 0.0}
    
    try:
        features = self._extract_ml_features(incident)
        X = self._vectorize_features([features])
        
        # Apply PCA if available
        pca = self.pattern_clustering.get('pca')
        if pca:
            X_reduced = pca.transform(X)
        else:
            X_reduced = X
        
        # Get cluster assignment and distance
        kmeans = self.pattern_clustering['kmeans']
        cluster_id = kmeans.predict(X_reduced)[0]
        distance = kmeans.transform(X_reduced)[0].min()
        
        # Anomaly threshold (tunable)
        anomaly_threshold = 2.0
        is_anomaly = distance > anomaly_threshold
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(distance),
            "cluster_id": int(cluster_id),
            "threshold": anomaly_threshold
        }
    except Exception as e:
        logger.error("Error detecting anomalies", error=str(e))
        return {"is_anomaly": False, "anomaly_score": 0.0, "error": str(e)}
```

## 🛠️ Training Workflows

### Command Line Training

```bash
# Train new model from data
python scripts/train_models.py \
  --data data/training.json \
  --output models/trained/ai_model_v2.pkl \
  --min-samples 100 \
  --export-summary models/exports/model_summary_v2.json

# Evaluate existing model
python scripts/evaluate_model.py \
  --model models/latest.pkl \
  --test-data data/test.json \
  --interactive

# CLI training interface
python cli.py ml train \
  --data-file data/incidents.json \
  --output-file models/new_model.pkl \
  --min-samples 50
```

### Programmatic Training

```python
# Initialize AI engine
ai_engine = AIDecisionEngine()

# Add training data from resolved incidents
for incident_data in training_dataset:
    incident = IncidentCreate(**incident_data["incident"])
    ai_engine.add_training_data(
        incident=incident,
        outcome=incident_data["outcome"],
        resolution_time=incident_data["resolution_time"],
        success=incident_data["success"]
    )

# Train models
training_results = ai_engine.train_models(min_samples=100)

# Save trained model
if training_results["success"]:
    ai_engine.save_model("models/trained/new_model.pkl")
```

## 🔄 Production Deployment

### Current Limitation

The main application (`src/main.py`) currently initializes the AI engine **without** loading models:

```python
# Current - no models loaded
ai_engine = AIDecisionEngine()  # Falls back to rule-based analysis
```

### Enabling Model Usage

To use trained models in production, modify the initialization:

```python
# Modified - load trained models
ai_engine = AIDecisionEngine(model_path="models/latest.pkl")
```

### Model Loading at Runtime

You can also load models after initialization:

```python
ai_engine = AIDecisionEngine()
if ai_engine.load_model("models/latest.pkl"):
    logger.info("ML models loaded successfully")
else:
    logger.warning("Failed to load models, using rule-based analysis")
```

### Environment Variable Configuration

Add model path configuration:

```bash
# .env
AI_MODEL_PATH=models/latest.pkl
```

```python
# src/main.py
from src.core.config import get_settings

settings = get_settings()
model_path = getattr(settings, 'ai_model_path', None)
ai_engine = AIDecisionEngine(model_path=model_path)
```

## 📊 Model Metadata and Versioning

### Metadata Structure

```python
model_metadata = {
    "version": "1.0.0",
    "trained_at": "2025-07-30T10:15:00Z",
    "training_samples": 500,
    "accuracy": 0.94,
    "feature_names": ["feature_0", "feature_1", ...],
    "label_classes": ["database_connectivity", "memory_issues", ...],
    "model_parameters": {
        "n_estimators": 100,
        "random_state": 42,
        "class_weight": "balanced"
    }
}
```

### Version Management

```bash
# Create versioned models
MODEL_VERSION="v2.1.0"
MODEL_FILE="models/trained/ai_model_${MODEL_VERSION}.pkl"

# Backup current production model
PROD_PATH="models/production"
if [ -f "${PROD_PATH}/current.pkl" ]; then
    cp "${PROD_PATH}/current.pkl" "${PROD_PATH}/backup_$(date +%Y%m%d_%H%M%S).pkl"
fi

# Deploy new model
cp "$MODEL_FILE" "${PROD_PATH}/current.pkl"
```

## 🧪 Testing and Validation

### Model Evaluation

```python
def _evaluate_models(self, X: np.ndarray, y_category: np.ndarray, y_success: np.ndarray):
    """Evaluate trained models."""
    # Split data for evaluation
    X_train, X_test, y_cat_train, y_cat_test, y_succ_train, y_succ_test = train_test_split(
        X, y_category, y_success, test_size=0.2, random_state=42
    )
    
    # Classification accuracy
    y_pred = self.incident_classifier.predict(X_test)
    classification_accuracy = accuracy_score(y_cat_test, y_pred)
    
    # Confidence model accuracy
    confidence_accuracy = accuracy_score(y_succ_test, self.confidence_model.predict(X_test))
    
    return {
        "classification_accuracy": classification_accuracy,
        "confidence_accuracy": confidence_accuracy,
        "test_samples": len(X_test)
    }
```

### Interactive Testing

```python
# CLI interactive mode
python scripts/evaluate_model.py --model models/latest.pkl --interactive

# Test specific incident
python cli.py ml test-decision \
  --model models/latest.pkl \
  --title "Database timeout error" \
  --description "Connection timeout to PostgreSQL database" \
  --service "postgresql" \
  --severity "high"
```

## 📈 Performance Monitoring

### Model Performance Metrics

Track model performance in production:

```python
class ModelPerformanceTracker:
    def track_prediction(self, incident, prediction, actual_outcome):
        """Track prediction accuracy for model monitoring."""
        accuracy = prediction == actual_outcome
        
        metrics = {
            "timestamp": datetime.utcnow(),
            "incident_id": incident.id,
            "predicted_category": prediction["category"],
            "actual_category": actual_outcome["category"],
            "confidence": prediction["confidence"],
            "accuracy": accuracy
        }
        
        # Store metrics for analysis
        self.store_metrics(metrics)
```

### Continuous Learning

Set up automated retraining:

```python
async def continuous_learning_pipeline():
    """Automated model retraining pipeline."""
    # Collect new training data
    new_data = await collect_recent_incidents()
    
    # Check if retraining is needed
    if len(new_data) >= 100:  # Retrain every 100 new incidents
        ai_engine = AIDecisionEngine()
        
        # Add new training data
        for incident_data in new_data:
            ai_engine.add_training_data(**incident_data)
        
        # Retrain models
        results = ai_engine.train_models()
        
        # Deploy if improvement
        if results["evaluation"]["classification_accuracy"] > current_model_accuracy:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/trained/retrained_{timestamp}.pkl"
            ai_engine.save_model(model_path)
```

## 🔧 Troubleshooting

### Common Issues

1. **Model not loading**: Check file path and permissions
2. **Low accuracy**: Need more training data or feature engineering
3. **Memory issues**: Large models may require more RAM
4. **Prediction errors**: Verify feature extraction consistency

### Debug Commands

```bash
# Check model info
python cli.py ml model-info --model models/latest.pkl

# Validate model file
python -c "import pickle; pickle.load(open('models/latest.pkl', 'rb'))"

# Test feature extraction
python cli.py ml test-features --incident-file test_incident.json
```

### Fallback Behavior

When models fail to load or predict:

```python
# System automatically falls back to rule-based analysis
if not self.incident_classifier:
    return self._determine_root_cause_category(incident), 0.5
```

## 📚 Best Practices

1. **Regular Retraining**: Retrain models monthly or after 500+ new incidents
2. **Version Control**: Keep model versions with metadata
3. **A/B Testing**: Test new models against production models
4. **Monitoring**: Track prediction accuracy and model drift
5. **Backup**: Always backup production models before deployment
6. **Validation**: Validate models on held-out test data
7. **Documentation**: Document model changes and performance

## 🚀 Future Enhancements

1. **Deep Learning**: Implement neural networks for complex pattern recognition
2. **Online Learning**: Update models in real-time with new data
3. **Model Ensemble**: Combine multiple models for better accuracy
4. **Feature Engineering**: Advanced feature extraction from logs and metrics
5. **Automated Pipeline**: Fully automated training and deployment pipeline
