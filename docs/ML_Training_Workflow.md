# AI Decision Engine - ML Training Workflow

## Overview

The AI Decision Engine implements a complete machine learning training workflow that automatically learns from incident resolutions to improve future predictions. This document explains exactly how the models are trained, saved, and used.

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Decision Engine                           │
├─────────────────────────────────────────────────────────────────┤
│  🧠 ML Models:                                                 │
│  • RandomForestClassifier (Incident Classification)            │
│  • CalibratedClassifierCV (Resolution Confidence)              │
│  • PCA + KMeans (Anomaly Detection)                           │
│  • TfidfVectorizer (Text Feature Extraction)                  │
├─────────────────────────────────────────────────────────────────┤
│  💾 Persistence:                                              │
│  • Database: TrainingData table (SQLite/PostgreSQL)           │
│  • File System: models/ai_decision_engine.pkl                 │
│  • In-Memory: self.training_data["incidents"]                 │
├─────────────────────────────────────────────────────────────────┤
│  🔄 Training Triggers:                                        │
│  • Startup: Auto-load/train if ≥10 samples                   │
│  • Incremental: Every 25 new samples                          │
│  • Manual: POST /api/v1/retrain endpoint                      │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 **Training Data Flow**

### 1. **Data Collection**
When an incident is resolved, training data is collected:

```python
# Called after incident resolution
await ai_engine.add_training_data_async(
    incident=incident,
    outcome="resolved",  # or "failed"
    resolution_time=180.5,  # seconds
    success=True,  # or False
    confidence_score=0.85,  # AI's confidence in the resolution
    actions_executed=["restart_service", "scale_resources"]
)
```

**What gets stored:**
- **Incident metadata**: title, service, severity, description
- **Extracted features**: text patterns, error types, time-based features
- **Resolution outcome**: success/failure, time taken, actions used
- **Confidence tracking**: AI's prediction accuracy

### 2. **Feature Extraction**
From each incident, the system extracts ML features:

```python
features = {
    # Text features
    "title_length": len(incident.title),
    "description_length": len(incident.description),
    "text_content": f"{incident.title} {incident.description}",
    
    # Pattern-based features  
    "has_timeout": "timeout" in description.lower(),
    "has_memory_issue": "memory" in description.lower(),
    "has_connection_issue": "connection" in description.lower(),
    "has_database_issue": "database" in description.lower(),
    "has_spark_issue": "spark" in description.lower(),
    "has_airflow_issue": "airflow" in description.lower(),
    
    # Categorical features
    "service": incident.service,
    "severity": incident.severity,
    
    # Time-based features
    "hour_of_day": datetime.utcnow().hour,
    "day_of_week": datetime.utcnow().weekday(),
    
    # Error patterns
    "error_patterns": ["timeout", "failed", "error"],
    "num_error_patterns": 2
}
```

### 3. **Model Training**
Three ML models are trained from the collected data:

#### **A. Incident Classifier (RandomForestClassifier)**
- **Purpose**: Categorize incidents into types
- **Categories**: 
  - `database_connectivity`: DB issues, timeouts
  - `memory_issues`: OOM errors, heap problems  
  - `workflow_failure`: Airflow DAG failures
  - `compute_failure`: Spark job failures
  - `data_availability`: Missing files, data issues
  - `unknown`: Unclassified incidents
- **Input**: TF-IDF text vectors + numerical features
- **Output**: Category + confidence score

#### **B. Confidence Model (CalibratedClassifierCV)** 
- **Purpose**: Predict probability of resolution success
- **Input**: Same feature set as classifier
- **Output**: Confidence score (0.0 - 1.0)
- **Usage**: Determines if automation should be attempted (threshold: 0.6)

#### **C. Pattern Clustering (PCA + KMeans)**
- **Purpose**: Detect anomalous incidents
- **Process**: PCA reduces dimensions → KMeans clusters patterns
- **Output**: Anomaly score + is_anomaly boolean
- **Usage**: Identifies unusual incidents that need manual review

## 🔄 **Training Lifecycle**

### **Startup Training**
```python
async def start(self):
    # 1. Try to load existing models
    model_loaded = self.load_model("models/ai_decision_engine.pkl")
    
    # 2. Load training data from database  
    await self._load_training_data_from_db()
    
    # 3. Decide if training is needed
    if not model_loaded and samples >= 10:
        # Train new models
        results = self.train_models(min_samples=10)
        if results["success"]:
            self.save_model("models/ai_decision_engine.pkl")
    elif model_loaded and new_samples >= 100:
        # Retrain if models are outdated
        results = self.train_models(min_samples=10) 
        self.save_model("models/ai_decision_engine.pkl")
```

### **Incremental Training**
```python
async def _check_incremental_training(self):
    total_samples = len(self.training_data["incidents"])
    last_training_samples = self.model_metadata.get("training_samples", 0)
    new_samples = total_samples - last_training_samples
    
    # Trigger retraining every 25 new samples
    if new_samples >= 25 and total_samples >= 50:
        results = self.train_models(min_samples=10)
        if results["success"]:
            self.save_model("models/ai_decision_engine.pkl")
```

### **Manual Training**
```bash
# Via API endpoint
curl -X POST http://localhost:8000/api/v1/retrain
```

## 💾 **Model Persistence**

### **File Storage**
Models are saved as pickle files containing:
```python
model_data = {
    "incident_classifier": RandomForestClassifier,
    "confidence_model": CalibratedClassifierCV, 
    "pattern_clustering": {"pca": PCA, "kmeans": KMeans},
    "feature_vectorizer": TfidfVectorizer,
    "label_encoders": {"category": LabelEncoder},
    "model_metadata": {
        "version": "1.0.0",
        "training_samples": 125,
        "accuracy": 0.923,
        "trained_at": "2025-07-30T14:18:23Z"
    },
    "training_data": {...}  # Optional historical data
}
```

### **Database Storage**
Training data is persisted in the `training_data` table:
```sql
CREATE TABLE training_data (
    id VARCHAR PRIMARY KEY,
    incident_id VARCHAR,
    incident_title VARCHAR NOT NULL,
    incident_service VARCHAR NOT NULL, 
    incident_severity VARCHAR NOT NULL,
    incident_description TEXT,
    features JSON NOT NULL,           -- Extracted ML features
    outcome VARCHAR NOT NULL,         -- "resolved" or "failed"
    resolution_time FLOAT NOT NULL,   -- Seconds to resolve
    success BOOLEAN NOT NULL,         -- Was resolution successful
    confidence_score FLOAT,           -- AI's confidence in resolution  
    actions_executed JSON,            -- Actions that were executed
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🎯 **Model Usage**

### **Incident Classification**
```python
# Predict incident category
category, confidence = ai_engine.predict_incident_category(incident)
# Returns: ("database_connectivity", 0.87)
```

### **Resolution Confidence**
```python
# Predict automation success probability
confidence = ai_engine.predict_resolution_confidence(incident)
# Returns: 0.73 (73% confidence)

# Decision logic
if confidence > 0.6:
    # Proceed with automation
    actions = generate_actions(incident)
    await execute_automated_actions(incident, actions)
else:
    # Escalate to manual intervention
    await create_manual_intervention_alert(incident)
```

### **Anomaly Detection**
```python
# Detect unusual incidents
anomaly_result = ai_engine.detect_anomalies(incident)
# Returns: {"is_anomaly": True, "anomaly_score": 2.4}

if anomaly_result["is_anomaly"]:
    # Flag for extra attention
    await escalate_to_senior_engineer(incident)
```

## 📈 **Performance Monitoring**

### **Training Metrics**
- **Classification Accuracy**: How well the model categorizes incidents
- **Confidence Calibration**: How accurate confidence predictions are
- **Anomaly Detection Rate**: Percentage of incidents flagged as anomalous

### **API Endpoints**
```bash
# Get model status
GET /api/v1/model-status

# Trigger retraining  
POST /api/v1/retrain

# Add training data
POST /api/v1/add-training-data

# Get training statistics
GET /api/v1/training-stats
```

## 🛠️ **Configuration**

### **Training Parameters**
```python
# Minimum samples needed for training
MIN_TRAINING_SAMPLES = 10

# Incremental retraining threshold  
INCREMENTAL_THRESHOLD = 25

# Confidence threshold for automation
AUTOMATION_CONFIDENCE_THRESHOLD = 0.6

# Model file location
MODEL_PATH = "models/ai_decision_engine.pkl"
```

### **Feature Engineering**
```python
# Text vectorization settings
TfidfVectorizer(
    max_features=1000,        # Maximum vocabulary size
    stop_words='english',     # Remove common English words
    ngram_range=(1, 2),      # Use unigrams and bigrams
    min_df=1,                # Don't ignore any terms
    max_df=0.95              # Ignore very common terms
)

# Classification settings
RandomForestClassifier(
    n_estimators=100,         # Number of trees
    random_state=42,          # Reproducible results
    class_weight='balanced'   # Handle class imbalance
)
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **"empty vocabulary" errors**
   - **Cause**: TF-IDF vectorizer has no valid text features
   - **Solution**: System automatically falls back to rule-based classification

2. **Model file not found**
   - **Cause**: First startup or deleted model file
   - **Solution**: System trains new models from database data

3. **Database loading errors**
   - **Cause**: Timestamp format issues or corrupted data
   - **Solution**: System initializes empty and starts fresh training

### **Fallback Behavior**
The system is designed to be resilient:
- **No models**: Falls back to rule-based analysis
- **Prediction errors**: Returns conservative confidence scores
- **Training failures**: Continues with existing models
- **Database issues**: Starts with empty training data

## 🎉 **Success Example**

Here's a complete example from training to prediction:

```python
# 1. Incident occurs
incident = IncidentCreate(
    title="Database connection timeout",
    service="payment-db", 
    severity="high",
    description="Connection timeout after 30 seconds. Pool exhausted."
)

# 2. AI analyzes and suggests actions
category, cat_confidence = ai_engine.predict_incident_category(incident)
res_confidence = ai_engine.predict_resolution_confidence(incident)

print(f"Category: {category} ({cat_confidence:.2f})")
print(f"Automation confidence: {res_confidence:.2f}")

# 3. Actions are executed (automatically or manually)
actions = ["restart_database_connection", "scale_resources"]
success = await execute_actions(actions)

# 4. Training data is collected
await ai_engine.add_training_data_async(
    incident=incident,
    outcome="resolved",
    resolution_time=180,
    success=success,
    confidence_score=res_confidence,
    actions_executed=actions
)

# 5. Models automatically retrain after 25 samples
# 6. Future similar incidents benefit from learned experience
```

This creates a continuous learning loop where the AI gets smarter with each incident resolution! 🧠✨
