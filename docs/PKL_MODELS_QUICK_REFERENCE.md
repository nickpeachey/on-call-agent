# PKL Models Quick Reference

## 🎯 Current Status
- **Models Available**: `models/first_model.pkl` (316KB), `models/latest.pkl` (symlink)
- **Production Status**: **NOT LOADED** - Main app uses rule-based fallback
- **Training Data**: Available for retraining

## 🔧 Enable Models in Production

### Option 1: Modify Main Application
```python
# File: src/main.py, Line 35
# Change from:
ai_engine = AIDecisionEngine()

# Change to:
ai_engine = AIDecisionEngine(model_path="models/latest.pkl")
```

### Option 2: Environment Variable
```bash
# Add to .env
AI_MODEL_PATH=models/latest.pkl
```

```python
# In src/main.py
model_path = os.getenv("AI_MODEL_PATH")
ai_engine = AIDecisionEngine(model_path=model_path)
```

## 📊 Model Components in PKL Files

| Component | Type | Purpose |
|-----------|------|---------|
| `incident_classifier` | RandomForestClassifier | Categorize incidents (database, memory, workflow, etc.) |
| `confidence_model` | CalibratedClassifierCV | Predict resolution success probability |
| `pattern_clustering` | dict (PCA + KMeans) | Detect anomalous incidents |
| `feature_vectorizer` | TfidfVectorizer | Convert text to numerical features |
| `label_encoders` | dict | Encode/decode categorical labels |
| `model_metadata` | dict | Training info, accuracy, timestamps |

## 🚀 Key Usage Methods

### Load Model
```python
ai_engine = AIDecisionEngine()
success = ai_engine.load_model("models/latest.pkl")
```

### Save Model
```python
ai_engine.save_model("models/new_model.pkl")
```

### Predict Category
```python
category, confidence = ai_engine.predict_incident_category(incident)
# Returns: ("database_connectivity", 0.85)
```

### Predict Resolution Confidence
```python
confidence = ai_engine.predict_resolution_confidence(incident)
# Returns: 0.73 (73% confidence in automated resolution)
```

### Train New Model
```python
# Add training data
ai_engine.add_training_data(incident, "resolved", 120.5, True)

# Train when enough data
results = ai_engine.train_models(min_samples=100)
if results["success"]:
    ai_engine.save_model("models/retrained.pkl")
```

## 🎯 Decision Thresholds

| Confidence | Action | Behavior |
|------------|---------|----------|
| > 0.8 | High Confidence | Execute automated actions immediately |
| 0.6 - 0.8 | Medium Confidence | Execute with monitoring |
| < 0.6 | Low Confidence | Manual intervention required |

## 📁 Files Using PKL Models

| File | Operations | Purpose |
|------|------------|---------|
| `src/ai/__init__.py` | save, load, predict | Core ML functionality |
| `cli.py` | load, evaluate | Command-line model management |
| `scripts/train_models.py` | save | Automated training |
| `scripts/evaluate_model.py` | load, test | Model evaluation |

## 🛠️ CLI Commands

### Train Model
```bash
python scripts/train_models.py --data data/training.json --output models/ai_model.pkl
```

### Evaluate Model
```bash
python scripts/evaluate_model.py --model models/latest.pkl --test-data data/test.json
```

### Model Info
```bash
python cli.py ml model-info --model models/latest.pkl
```

### Test Prediction
```bash
python cli.py ml test-decision --model models/latest.pkl \
  --title "Database timeout" --service "postgresql" --severity "high"
```

## 🔄 Fallback Behavior

When models not loaded or prediction fails:
- **Classification**: Rule-based category detection
- **Confidence**: Default 0.5 (rule-based) or 0.3 (error)
- **Analysis Method**: Marked as "rule_based" vs "ml_models"

## ⚡ Performance

- **Model Loading**: ~1-2 seconds
- **Prediction**: <50ms per incident
- **Memory Usage**: ~50-100MB
- **Fallback Speed**: ~5-10ms

## 🚨 To Use Models Now

1. **Check models exist**:
   ```bash
   ls -la models/
   ```

2. **Enable in production**:
   ```python
   # In src/main.py
   ai_engine = AIDecisionEngine(model_path="models/latest.pkl")
   ```

3. **Rebuild Docker image**:
   ```bash
   docker-compose build oncall-agent
   docker-compose up -d oncall-agent
   ```

4. **Verify loading**:
   ```bash
   docker-compose logs oncall-agent | grep "Model loaded"
   ```

## 📈 Model Categories

Trained to classify incidents into:
- `database_connectivity` - DB timeouts, connection issues
- `memory_issues` - OOM errors, heap problems
- `workflow_failure` - Airflow DAG failures
- `compute_failure` - Spark job failures  
- `data_availability` - Missing files, data issues
- `unknown` - Unclassified incidents
