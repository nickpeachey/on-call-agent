# 🤖 AI/ML Model Training Guide

Complete guide for training, evaluating, and deploying machine learning models in the AI On-Call Agent system.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Model Export & Deployment](#model-export--deployment)
7. [Using Trained Models](#using-trained-models)
8. [Advanced Training Techniques](#advanced-training-techniques)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

## 🎯 Overview

The AI On-Call Agent uses scikit-learn models for:

- **Incident Classification**: Automatically categorize incidents by type
- **Resolution Confidence**: Predict likelihood of successful automated resolution
- **Anomaly Detection**: Identify unusual incidents that may need special attention
- **Pattern Recognition**: Learn from historical data to improve decision-making

### Model Architecture

```
Training Data → Feature Extraction → ML Pipeline → Trained Models
                     ↓
         [Text Features + Numerical Features]
                     ↓
    [RandomForest Classifier + Calibrated Confidence + KMeans Clustering]
```

## 🔧 Prerequisites

### 1. Install Dependencies

```bash
# Core ML dependencies (already in requirements.txt)
pip install scikit-learn>=1.3.0 numpy>=1.24.0 pandas>=2.0.0

# Optional: For advanced analysis
pip install matplotlib seaborn jupyter
```

### 2. Set Up Environment

```bash
# Create models directory
mkdir -p models/trained
mkdir -p models/exports
mkdir -p data/training

# Verify scikit-learn installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
```

## 📊 Data Preparation

### 1. Collect Training Data

Training data comes from resolved incidents with outcomes:

```python
# Example training data structure
training_sample = {
    "incident": {
        "title": "Database Connection Timeout",
        "description": "PostgreSQL connection pool exhausted after 30 seconds",
        "service": "postgres",
        "severity": "high",
        "tags": ["database", "timeout", "connection"]
    },
    "outcome": "restart_database_connection",
    "resolution_time": 120,  # seconds
    "success": True
}
```

### 2. Add Training Data via API

```python
from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate

# Initialize AI engine
ai_engine = AIDecisionEngine()

# Add training samples
incident = IncidentCreate(
    title="Database Connection Timeout",
    description="PostgreSQL connection pool exhausted",
    service="postgres",
    severity="high",
    tags=["database", "timeout"]
)

ai_engine.add_training_data(
    incident=incident,
    outcome="restart_database_connection",
    resolution_time=120,
    success=True
)
```

### 3. Bulk Import Historical Data

```python
import json
import pandas as pd

# Load from CSV
df = pd.read_csv('data/historical_incidents.csv')

for _, row in df.iterrows():
    incident = IncidentCreate(
        title=row['title'],
        description=row['description'],
        service=row['service'],
        severity=row['severity'],
        tags=row['tags'].split(',')
    )
    
    ai_engine.add_training_data(
        incident=incident,
        outcome=row['outcome'],
        resolution_time=row['resolution_time'],
        success=row['success']
    )
```

### 4. Sample Training Data

Create `data/sample_training_data.json`:

```json
[
  {
    "incident": {
      "title": "Spark Job OutOfMemoryError",
      "description": "Java heap space exceeded in executor spark-executor-1",
      "service": "spark",
      "severity": "high",
      "tags": ["spark", "memory", "oom"]
    },
    "outcome": "restart_spark_job",
    "resolution_time": 180,
    "success": true
  },
  {
    "incident": {
      "title": "Airflow DAG Timeout",
      "description": "data_pipeline DAG stuck for 45 minutes",
      "service": "airflow",
      "severity": "medium",
      "tags": ["airflow", "dag", "timeout"]
    },
    "outcome": "restart_airflow_dag",
    "resolution_time": 90,
    "success": true
  },
  {
    "incident": {
      "title": "Database Connection Failed",
      "description": "Cannot connect to PostgreSQL database",
      "service": "postgres",
      "severity": "critical",
      "tags": ["database", "connection", "failure"]
    },
    "outcome": "restart_database_connection",
    "resolution_time": 240,
    "success": false
  }
]
```

## 🎓 Model Training

### 1. Basic Training

```python
from src.ai import AIDecisionEngine

# Initialize with training data
ai_engine = AIDecisionEngine()

# Load sample data (implement your data loading here)
# ... add training data as shown above ...

# Train models (minimum 50 samples recommended)
training_results = ai_engine.train_models(min_samples=50)

if training_results["success"]:
    print(f"Training completed!")
    print(f"Samples used: {training_results['training_samples']}")
    print(f"Classification accuracy: {training_results['evaluation']['classification_accuracy']:.3f}")
else:
    print(f"Training failed: {training_results['error']}")
```

### 2. Advanced Training with CLI

Create a training script `scripts/train_models.py`:

```python
#!/usr/bin/env python3
"""
AI Model Training Script
Usage: python scripts/train_models.py --data data/training.json --output models/trained/ai_model.pkl
"""

import argparse
import json
from pathlib import Path
from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate

def load_training_data(file_path: str):
    """Load training data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    ai_engine = AIDecisionEngine()
    
    for sample in data:
        incident = IncidentCreate(**sample["incident"])
        ai_engine.add_training_data(
            incident=incident,
            outcome=sample["outcome"],
            resolution_time=sample.get("resolution_time", 300),
            success=sample.get("success", True)
        )
    
    return ai_engine

def main():
    parser = argparse.ArgumentParser(description="Train AI models")
    parser.add_argument("--data", required=True, help="Training data JSON file")
    parser.add_argument("--output", required=True, help="Output model file")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum training samples")
    parser.add_argument("--export-summary", help="Export model summary JSON")
    
    args = parser.parse_args()
    
    print("🚀 Starting AI model training...")
    
    # Load training data
    print(f"📊 Loading training data from {args.data}")
    ai_engine = load_training_data(args.data)
    
    # Train models
    print(f"🎓 Training models (min samples: {args.min_samples})")
    results = ai_engine.train_models(min_samples=args.min_samples)
    
    if results["success"]:
        print("✅ Training completed successfully!")
        print(f"   📈 Samples used: {results['training_samples']}")
        print(f"   🎯 Accuracy: {results['evaluation'].get('classification_accuracy', 0):.3f}")
        
        # Save model
        print(f"💾 Saving model to {args.output}")
        if ai_engine.save_model(args.output):
            print("✅ Model saved successfully!")
        else:
            print("❌ Failed to save model")
            return 1
        
        # Export summary if requested
        if args.export_summary:
            print(f"📄 Exporting summary to {args.export_summary}")
            ai_engine.export_model_summary(args.export_summary)
            
    else:
        print(f"❌ Training failed: {results['error']}")
        return 1
    
    print("🎉 Training complete!")
    return 0

if __name__ == "__main__":
    exit(main())
```

### 3. Run Training

```bash
# Create sample training data
python -c "
import json
from pathlib import Path

sample_data = [
    {
        'incident': {
            'title': 'Database Connection Timeout',
            'description': 'PostgreSQL connection timeout after 30 seconds',
            'service': 'postgres',
            'severity': 'high',
            'tags': ['database', 'timeout']
        },
        'outcome': 'restart_database_connection',
        'resolution_time': 120,
        'success': True
    },
    # Add more samples...
]

Path('data').mkdir(exist_ok=True)
with open('data/sample_training.json', 'w') as f:
    json.dump(sample_data * 20, f, indent=2)  # Duplicate to reach min samples
print('Sample training data created')
"

# Train the model
python scripts/train_models.py \
  --data data/sample_training.json \
  --output models/trained/ai_model_v1.pkl \
  --export-summary models/exports/model_summary_v1.json \
  --min-samples 50
```

## 📈 Model Evaluation

### 1. Training Metrics

```python
# Get model information
model_info = ai_engine.get_model_info()
print(f"Model accuracy: {model_info['metadata']['accuracy']:.3f}")
print(f"Training samples: {model_info['training_samples']}")
print(f"Feature count: {model_info['feature_count']}")

# Get feature importance
feature_importance = ai_engine._get_feature_importance()
print("Top 10 features:")
for feat in feature_importance[:10]:
    print(f"  {feat['feature']}: {feat['importance']:.3f}")
```

### 2. Evaluation Script

Create `scripts/evaluate_model.py`:

```python
#!/usr/bin/env python3
"""
Model Evaluation Script
Usage: python scripts/evaluate_model.py --model models/trained/ai_model.pkl --test-data data/test.json
"""

import argparse
import json
from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate

def evaluate_model(model_path: str, test_data_path: str):
    """Evaluate trained model on test data."""
    
    # Load model
    ai_engine = AIDecisionEngine()
    if not ai_engine.load_model(model_path):
        print(f"❌ Failed to load model from {model_path}")
        return
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    correct_predictions = 0
    total_predictions = len(test_data)
    
    print(f"🧪 Evaluating model on {total_predictions} test samples...")
    
    for i, sample in enumerate(test_data):
        incident = IncidentCreate(**sample["incident"])
        expected_category = sample.get("expected_category", "unknown")
        
        # Predict
        predicted_category, confidence = ai_engine.predict_incident_category(incident)
        resolution_confidence = ai_engine.predict_resolution_confidence(incident)
        anomaly_info = ai_engine.detect_anomalies(incident)
        
        # Check accuracy
        is_correct = predicted_category.lower() == expected_category.lower()
        if is_correct:
            correct_predictions += 1
        
        print(f"Sample {i+1}:")
        print(f"  Title: {incident.title}")
        print(f"  Expected: {expected_category}")
        print(f"  Predicted: {predicted_category} (confidence: {confidence:.3f})")
        print(f"  Resolution confidence: {resolution_confidence:.3f}")
        print(f"  Is anomaly: {anomaly_info.get('is_anomaly', False)}")
        print(f"  Correct: {'✅' if is_correct else '❌'}")
        print()
    
    accuracy = correct_predictions / total_predictions
    print(f"📊 Overall Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AI model")
    parser.add_argument("--model", required=True, help="Trained model file (.pkl)")
    parser.add_argument("--test-data", required=True, help="Test data JSON file")
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.test_data)

if __name__ == "__main__":
    main()
```

### 3. Cross-Validation

```python
from sklearn.model_selection import cross_val_score
import numpy as np

# Perform cross-validation during training
def train_with_cv(ai_engine, cv_folds=5):
    """Train with cross-validation."""
    X, y_category, y_success, y_confidence = ai_engine._prepare_training_data()
    
    # Cross-validate classification
    classifier = ai_engine.incident_classifier.named_steps['classifier']
    cv_scores = cross_val_score(classifier, X, y_category, cv=cv_folds)
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return cv_scores
```

## 🚀 Model Export & Deployment

### 1. Save Trained Model

```python
# Save model with metadata
model_path = "models/trained/ai_model_v1_0_0.pkl"
success = ai_engine.save_model(model_path)

if success:
    print(f"✅ Model saved to {model_path}")
    
    # Export detailed summary
    summary_path = "models/exports/ai_model_summary_v1_0_0.json"
    ai_engine.export_model_summary(summary_path)
    print(f"📄 Summary exported to {summary_path}")
```

### 2. Model Versioning

```python
from datetime import datetime

def save_versioned_model(ai_engine, base_path="models/trained"):
    """Save model with version and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy = ai_engine.model_metadata.get("accuracy", 0)
    samples = ai_engine.model_metadata.get("training_samples", 0)
    
    # Generate filename with metadata
    filename = f"ai_model_v1_acc{accuracy:.3f}_samples{samples}_{timestamp}.pkl"
    model_path = f"{base_path}/{filename}"
    
    if ai_engine.save_model(model_path):
        print(f"✅ Versioned model saved: {filename}")
        
        # Create symlink to latest
        latest_path = f"{base_path}/latest.pkl"
        Path(latest_path).unlink(missing_ok=True)
        Path(model_path).symlink_to(Path(latest_path).resolve())
        
        return model_path
    
    return None
```

### 3. Production Deployment

```bash
# Production deployment script
#!/bin/bash

# Deploy trained model to production
MODEL_VERSION="v1.0.0"
MODEL_FILE="models/trained/ai_model_${MODEL_VERSION}.pkl"
PROD_PATH="/opt/on-call-agent/models/production/"

echo "🚀 Deploying AI model ${MODEL_VERSION} to production..."

# Validate model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model file not found: $MODEL_FILE"
    exit 1
fi

# Create backup of current production model
if [ -f "${PROD_PATH}/current.pkl" ]; then
    cp "${PROD_PATH}/current.pkl" "${PROD_PATH}/backup_$(date +%Y%m%d_%H%M%S).pkl"
    echo "📦 Created backup of current model"
fi

# Copy new model to production
cp "$MODEL_FILE" "${PROD_PATH}/current.pkl"
echo "✅ Model deployed to production"

# Update model metadata
cat > "${PROD_PATH}/model_info.json" << EOF
{
    "version": "${MODEL_VERSION}",
    "deployed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "model_file": "current.pkl",
    "deployment_method": "manual"
}
EOF

echo "📄 Updated production model metadata"
echo "🎉 Deployment complete!"

# Restart application (example)
# systemctl restart on-call-agent
```

## 🔧 Using Trained Models

### 1. Load Model in Application

```python
# In your application startup
from src.ai import AIDecisionEngine

# Load production model
ai_engine = AIDecisionEngine(model_path="models/production/current.pkl")

# Verify model loaded
model_info = ai_engine.get_model_info()
print(f"Loaded model version: {model_info['metadata'].get('version', 'unknown')}")
print(f"Training samples: {model_info['training_samples']}")
print(f"Accuracy: {model_info['metadata'].get('accuracy', 0):.3f}")
```

### 2. Real-time Predictions

```python
# Predict incident category
incident = IncidentCreate(
    title="Database Connection Error",
    description="Cannot connect to PostgreSQL server",
    service="postgres",
    severity="high",
    tags=["database", "connection"]
)

# Get predictions
category, category_confidence = ai_engine.predict_incident_category(incident)
resolution_confidence = ai_engine.predict_resolution_confidence(incident)
anomaly_info = ai_engine.detect_anomalies(incident)

print(f"Predicted category: {category} (confidence: {category_confidence:.3f})")
print(f"Resolution confidence: {resolution_confidence:.3f}")
print(f"Is anomaly: {anomaly_info['is_anomaly']}")

# Use in decision making
if resolution_confidence > 0.8 and not anomaly_info['is_anomaly']:
    print("✅ Recommend automated resolution")
else:
    print("⚠️ Recommend manual intervention")
```

### 3. Continuous Learning

```python
# Add new training data as incidents are resolved
async def handle_incident_resolution(incident_id: str, outcome: str, success: bool, resolution_time: int):
    """Handle incident resolution for continuous learning."""
    
    # Get incident details (from database)
    incident = get_incident_by_id(incident_id)
    
    # Add to training data
    ai_engine.add_training_data(
        incident=incident,
        outcome=outcome,
        resolution_time=resolution_time,
        success=success
    )
    
    # Retrain periodically
    if len(ai_engine.training_data["incidents"]) % 100 == 0:  # Every 100 new samples
        print("🔄 Retraining model with new data...")
        results = ai_engine.train_models(min_samples=50)
        
        if results["success"]:
            # Save updated model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/trained/retrained_{timestamp}.pkl"
            ai_engine.save_model(model_path)
            print(f"✅ Retrained model saved: {model_path}")
```

## 🔬 Advanced Training Techniques

### 1. Feature Engineering

```python
def extract_advanced_features(incident: IncidentCreate) -> Dict[str, Any]:
    """Extract advanced features for better predictions."""
    description = incident.description.lower()
    
    # Time-based features
    now = datetime.utcnow()
    features = {
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "is_weekend": now.weekday() >= 5,
        "is_business_hours": 9 <= now.hour <= 17,
    }
    
    # Text complexity features
    features.update({
        "word_count": len(description.split()),
        "sentence_count": len(description.split('.')),
        "avg_word_length": np.mean([len(word) for word in description.split()]),
        "uppercase_ratio": sum(1 for c in description if c.isupper()) / len(description),
    })
    
    # Domain-specific features
    features.update({
        "has_error_code": bool(re.search(r'\b(error|code|exception)\s*:?\s*\d+', description)),
        "has_timestamp": bool(re.search(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}', description)),
        "has_ip_address": bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', description)),
        "has_file_path": bool(re.search(r'[/\\][a-zA-Z0-9_.-]+', description)),
    })
    
    return features
```

### 2. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_hyperparameters(ai_engine):
    """Tune model hyperparameters."""
    X, y_category, _, _ = ai_engine._prepare_training_data()
    
    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # Grid search
    base_pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y_category)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.3f}")
    
    # Update model with best parameters
    ai_engine.incident_classifier = grid_search.best_estimator_
    
    return grid_search.best_estimator_
```

### 3. Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def create_ensemble_model(ai_engine):
    """Create ensemble model with multiple algorithms."""
    
    # Define base models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    svm_model = SVC(probability=True, random_state=42)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('lr', lr_model),
            ('svm', svm_model)
        ],
        voting='soft'  # Use probability predictions
    )
    
    # Update AI engine with ensemble
    ai_engine.incident_classifier = Pipeline([
        ('classifier', ensemble)
    ])
    
    return ensemble
```

## 🔍 Troubleshooting

### 1. Common Training Issues

**Issue: Not enough training data**
```bash
Error: Need at least 50 training samples, got 23
```
**Solution:**
```python
# Generate synthetic data or collect more samples
# Use data augmentation for text data
from sklearn.utils import resample

# Upsample minority classes
def augment_training_data(ai_engine):
    # Implementation for data augmentation
    pass
```

**Issue: Low model accuracy**
```bash
Classification accuracy: 0.423
```
**Solution:**
```python
# 1. Check feature quality
feature_importance = ai_engine._get_feature_importance()
print("Low importance features:", [f for f in feature_importance if f['importance'] < 0.01])

# 2. Add more features
# 3. Tune hyperparameters
# 4. Check data quality
```

**Issue: Model overfitting**
```bash
Training accuracy: 0.95, Validation accuracy: 0.60
```
**Solution:**
```python
# Reduce model complexity
rf_params = {
    'n_estimators': 50,  # Reduce from 100
    'max_depth': 10,     # Add depth limit
    'min_samples_split': 10,  # Increase split threshold
}
```

### 2. Memory Issues

```python
# For large datasets, use incremental learning
from sklearn.linear_model import SGDClassifier

def create_memory_efficient_model():
    """Create model for large datasets."""
    return SGDClassifier(
        loss='log',  # For probability estimates
        learning_rate='adaptive',
        random_state=42
    )
```

### 3. Model Debugging

```python
def debug_model_predictions(ai_engine, test_incident):
    """Debug why model made a specific prediction."""
    
    # Get prediction details
    category, confidence = ai_engine.predict_incident_category(test_incident)
    features = ai_engine._extract_ml_features(test_incident)
    
    print(f"Prediction: {category} (confidence: {confidence:.3f})")
    print("\nFeatures:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Get feature importance
    importance = ai_engine._get_feature_importance()
    print("\nTop contributing features:")
    for feat in importance[:10]:
        print(f"  {feat['feature']}: {feat['importance']:.3f}")
```

## 📚 API Reference

### AIDecisionEngine Methods

#### Training Methods
- `add_training_data(incident, outcome, resolution_time, success)` - Add training sample
- `train_models(min_samples=50)` - Train all ML models
- `_prepare_training_data()` - Prepare data for training
- `_extract_ml_features(incident)` - Extract features from incident

#### Prediction Methods
- `predict_incident_category(incident)` - Predict incident category
- `predict_resolution_confidence(incident)` - Predict resolution confidence
- `detect_anomalies(incident)` - Detect anomalous incidents

#### Model Management
- `save_model(file_path)` - Save trained model to file
- `load_model(file_path)` - Load model from file
- `get_model_info()` - Get model metadata and info
- `export_model_summary(file_path)` - Export detailed summary

#### Evaluation Methods
- `_evaluate_models(X, y_category, y_success)` - Evaluate model performance
- `_get_feature_importance()` - Get feature importance rankings

### Model File Formats

**Saved Model (.pkl)**
- Complete model with all components
- Training data and metadata
- Feature vectorizers and encoders

**Model Summary (.json)**
- Human-readable model information
- Training history and metrics
- Feature importance rankings

### Configuration Options

```python
# Model initialization options
ai_engine = AIDecisionEngine(
    model_path="path/to/model.pkl"  # Optional: load existing model
)

# Training parameters
train_results = ai_engine.train_models(
    min_samples=50  # Minimum training samples required
)
```

---

## 🎯 Quick Start Example

```bash
# 1. Create training data
cat > data/quick_training.json << 'EOF'
[
  {
    "incident": {
      "title": "Database Connection Timeout",
      "description": "PostgreSQL connection pool exhausted",
      "service": "postgres",
      "severity": "high",
      "tags": ["database", "timeout"]
    },
    "outcome": "restart_database_connection",
    "resolution_time": 120,
    "success": true
  }
]
EOF

# 2. Train model
python -c "
from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate
import json

# Load training data
with open('data/quick_training.json') as f:
    data = json.load(f) * 60  # Duplicate to reach minimum

ai_engine = AIDecisionEngine()
for sample in data:
    incident = IncidentCreate(**sample['incident'])
    ai_engine.add_training_data(
        incident=incident,
        outcome=sample['outcome'], 
        resolution_time=sample['resolution_time'],
        success=sample['success']
    )

# Train and save
results = ai_engine.train_models(min_samples=50)
if results['success']:
    ai_engine.save_model('models/quick_model.pkl')
    print('✅ Model trained and saved!')
else:
    print(f'❌ Training failed: {results[\"error\"]}')
"

# 3. Use trained model
python -c "
from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate

# Load model
ai_engine = AIDecisionEngine(model_path='models/quick_model.pkl')

# Test prediction
incident = IncidentCreate(
    title='DB Connection Error',
    description='Cannot connect to database server',
    service='postgres',
    severity='high',
    tags=['database', 'connection']
)

category, confidence = ai_engine.predict_incident_category(incident)
print(f'Predicted: {category} (confidence: {confidence:.3f})')
"
```

🎉 **Congratulations!** You've successfully trained and deployed an AI model for incident classification and automated resolution!
