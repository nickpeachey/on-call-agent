# AI On-Call Agent Documentation

Welcome to the AI On-Call Agent documentation! This system provides intelligent, automated incident response for ETL infrastructure using machine learning and AI-powered decision making.

## Quick Start

1. **[System Architecture](SYSTEM_ARCHITECTURE.md)** - Understanding how the system works
2. **[Model Integration](MODEL_INTEGRATION.md)** - How ML models are trained and used  
3. **[API Documentation](API_DOCUMENTATION.md)** - REST API reference
4. **[Development Guide](DEVELOPMENT.md)** - Setting up development environment

## What is the AI On-Call Agent?

The AI On-Call Agent is an intelligent automation system that:

- **Monitors** ETL infrastructure and data pipelines
- **Analyzes** incidents using machine learning models
- **Decides** whether automated resolution is appropriate
- **Executes** remediation actions automatically
- **Learns** from resolution outcomes to improve over time

## Key Features

### 🤖 AI-Powered Analysis
- Machine learning models classify incidents by type and severity
- Natural language processing extracts key information from logs
- Risk assessment determines automation safety
- Confidence scoring ensures reliable decision making

### ⚡ Automated Resolution
- Executes appropriate actions based on incident type
- Service restarts, resource scaling, cache clearing
- Database connection management
- ETL pipeline recovery (Airflow, Spark)

### 📚 Knowledge Base Integration
- Learns from historical incident resolutions
- Matches similar past incidents for context
- Tracks success rates of different resolution approaches
- Continuously improves recommendations

### 🔍 Comprehensive Monitoring
- Real-time incident processing
- Detailed action execution logging
- Performance metrics and success rates
- Health monitoring of all system components

## System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   AI Decision    │    │     Action      │
│    Systems      │───▶│     Engine       │───▶│   Execution     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   ML Service     │
                       │  (Saved Models)  │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Knowledge Base   │
                       │   & Training     │
                       └──────────────────┘
```

## Core Components

### AI Decision Engine
- Central orchestrator for incident processing
- Coordinates analysis, decision making, and action execution
- Maintains incident queue and processing loop
- Integrates with all other system components

### ML Service  
- Manages machine learning models and predictions
- Loads trained models from disk for consistency
- Provides incident classification and action recommendations
- Ensures production uses same models as development

### Action Execution
- Executes remediation actions safely and reliably
- Supports multiple action types (restarts, scaling, etc.)
- Provides detailed logging and monitoring
- Includes rollback capabilities for failed actions

### Knowledge Base
- Stores historical incident data and resolutions
- Enables similarity matching for context
- Tracks success rates and effectiveness
- Supports continuous learning and improvement

## Getting Started

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Redis (for caching)
- Docker (for containerized services)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/on-call-agent.git
cd on-call-agent

# Install dependencies
pip install -r requirements.txt

# Set up database
python scripts/setup_database.py

# Train initial models
jupyter notebook ml_training_fixed.ipynb

# Start the system
python -m src.main
```

### Basic Usage

#### Submit an Incident via API

```bash
curl -X POST http://localhost:8000/api/incidents \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Database Connection Timeout",
    "description": "Application experiencing timeouts",
    "service": "api-service",
    "severity": "high"
  }'
```

#### Check System Health

```bash
curl http://localhost:8000/api/health
```

#### View Incident Status

```bash
curl http://localhost:8000/api/incidents/incident-id
```

## Architecture Deep Dive

### Model Training and Usage

1. **Development**: Models trained in Jupyter notebooks
2. **Persistence**: Models saved to disk using joblib
3. **Production**: ML Service loads exact same models
4. **Consistency**: All predictions use centralized ML Service

### Incident Processing Pipeline

1. **Ingestion**: Incidents submitted via API or monitoring systems
2. **Queuing**: Added to processing queue for AI analysis
3. **Analysis**: ML models extract features and classify incident
4. **Decision**: Risk assessment and automation decision logic
5. **Execution**: Appropriate actions executed if confidence threshold met
6. **Monitoring**: Results tracked and fed back for learning

### Safety and Reliability

- **Confidence Thresholds**: Actions only executed above 60% confidence
- **Risk Assessment**: Critical incidents require manual intervention  
- **Fallback Mechanisms**: Rule-based analysis if ML models fail
- **Action Validation**: All actions validated before execution
- **Comprehensive Logging**: Complete audit trail of all decisions

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/oncall_db

# ML Models
ML_MODEL_PATH=models/

# Automation Settings  
CONFIDENCE_THRESHOLD=0.6
MAX_CONCURRENT_ACTIONS=5

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

### Model Configuration

```python
# ML Service settings
ML_CONFIG = {
    "model_path": "models/",
    "auto_retrain": True,
    "min_training_samples": 10,
    "confidence_threshold": 0.6
}
```

## Monitoring and Metrics

### Key Metrics Tracked

- **Incident Volume**: Number of incidents processed
- **Automation Rate**: Percentage of incidents automated
- **Success Rate**: Percentage of successful resolutions  
- **Response Time**: Time from incident to resolution
- **Model Accuracy**: ML model prediction accuracy
- **Action Effectiveness**: Success rate by action type

### Health Monitoring

- **System Components**: All services health checked
- **Model Status**: Verify models loaded and functional
- **Queue Health**: Monitor processing queue depth
- **Performance**: Track response times and throughput

## Security

### Authentication
- API key based authentication for all endpoints
- Role-based access control for different operations
- Audit logging of all API access

### Action Security
- Action validation before execution
- Approval workflows for high-risk actions  
- Rollback capabilities for failed actions
- Comprehensive logging for accountability

## Troubleshooting

### Common Issues

**Models Not Loading**
- Check model files exist in configured path
- Verify file permissions and accessibility
- Check scikit-learn version compatibility

**High False Positive Rate**
- Review confidence threshold settings
- Analyze training data quality
- Consider retraining with more data

**Action Execution Failures**  
- Check action service connectivity
- Verify action parameters and permissions
- Review action execution logs

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
python -m src.main
```

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch
3. Set up development environment
4. Run tests: `pytest tests/`
5. Submit pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Write comprehensive tests
- Update documentation for changes

## Support

- **Documentation**: See docs/ directory for detailed guides
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join community discussions
- **Contact**: reach out to the development team

## License

This project is licensed under the MIT License - see LICENSE file for details.
