# Development Setup Guide

## Prerequisites

- Python 3.8 or higher
- PostgreSQL 12+
- Redis 6+
- Docker and Docker Compose
- Git

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/on-call-agent.git
cd on-call-agent
```

### 2. Python Environment

Create and activate virtual environment:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n oncall-agent python=3.9
conda activate oncall-agent
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Database Setup

#### Using Docker (Recommended)

```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Run database migrations
python scripts/setup_database.py
```

#### Manual Setup

```bash
# PostgreSQL
createdb oncall_agent_dev
export DATABASE_URL=postgresql://user:pass@localhost/oncall_agent_dev

# Redis  
redis-server --daemonize yes
export REDIS_URL=redis://localhost:6379
```

### 5. Environment Configuration

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/oncall_agent_dev
REDIS_URL=redis://localhost:6379

# ML Models
ML_MODEL_PATH=models/

# API Settings
API_HOST=localhost
API_PORT=8000
DEBUG=true

# Logging
LOG_LEVEL=DEBUG

# Authentication  
API_SECRET_KEY=your-secret-key-here
```

## Development Workflow

### 1. Model Development

Train initial models using the notebook:

```bash
# Start Jupyter
jupyter notebook

# Open and run ml_training_fixed.ipynb
# This creates models in the models/ directory
```

### 2. Running the System

#### Development Mode

```bash
# Start all services
python -m src.main

# Or start individual components
python -m src.api.main  # API server only
python -m src.ai       # AI engine only
```

#### Using Docker

```bash
# Build and start all services
docker-compose up --build

# Start specific services
docker-compose up api ai-engine
```

### 3. Testing

#### Run All Tests

```bash
pytest tests/
```

#### Run Specific Test Categories

```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# API tests
pytest tests/api/

# ML model tests
pytest tests/ml/
```

#### Test Coverage

```bash
pytest --cov=src tests/
```

### 4. Code Quality

#### Linting

```bash
# Run flake8
flake8 src/

# Run pylint
pylint src/

# Run mypy for type checking
mypy src/
```

#### Code Formatting

```bash
# Format with black
black src/ tests/

# Sort imports
isort src/ tests/
```

#### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Project Structure

```
on-call-agent/
├── src/
│   ├── ai/                 # AI Decision Engine
│   │   ├── __init__.py
│   │   └── models/         # ML model definitions
│   ├── api/                # REST API
│   │   ├── main.py
│   │   ├── incidents.py
│   │   └── health.py
│   ├── services/           # Core services
│   │   ├── ml_service.py
│   │   ├── action_execution.py
│   │   ├── knowledge_base.py
│   │   └── notifications.py
│   ├── models/             # Data models/schemas
│   │   └── schemas.py
│   ├── core/               # Core utilities
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── database.py
│   └── main.py             # Application entry point
├── tests/                  # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                   # Documentation
├── models/                 # Saved ML models
├── scripts/                # Utility scripts
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Configuration

### Settings Management

Settings are managed through `src/core/config.py`:

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    ml_model_path: str = "models/"
    confidence_threshold: float = 0.6
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Environment-Specific Config

- **Development**: `.env`
- **Testing**: `.env.test`
- **Production**: Environment variables

## Debugging

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
python -m src.main
```

### Interactive Debugging

```python
# Add breakpoints in code
import pdb; pdb.set_trace()

# Or use ipdb for better interface
import ipdb; ipdb.set_trace()
```

### VS Code Debugging

`.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug AI Engine",
            "type": "python",
            "request": "launch",
            "module": "src.main",
            "env": {
                "LOG_LEVEL": "DEBUG",
                "DEBUG": "true"
            },
            "console": "integratedTerminal"
        }
    ]
}
```

## Common Development Tasks

### Adding New ML Models

1. Create model in notebook
2. Save to `models/` directory
3. Update `MLService` to load new model
4. Add prediction methods
5. Integrate with AI Decision Engine

### Adding New Action Types

1. Define action in `schemas.py`
2. Implement in `ActionExecutionService`
3. Add to AI Decision Engine recommendations
4. Write tests for new action

### Database Schema Changes

1. Create migration script in `scripts/migrations/`
2. Update model definitions
3. Run migration on all environments
4. Update tests

### API Endpoint Changes

1. Update OpenAPI schema
2. Implement endpoint in appropriate router
3. Add input validation
4. Write integration tests
5. Update API documentation

## Testing Strategy

### Unit Tests

Test individual components in isolation:

```python
# Example: tests/unit/test_ml_service.py
import pytest
from src.services.ml_service import MLService

@pytest.fixture
async def ml_service():
    service = MLService()
    await service.initialize()
    return service

async def test_incident_classification(ml_service):
    text = "Database connection timeout"
    severity, confidence = await ml_service.predict_incident_severity(text)
    assert severity in ["low", "medium", "high", "critical"]
    assert 0 <= confidence <= 1
```

### Integration Tests

Test component interactions:

```python
# Example: tests/integration/test_ai_pipeline.py
import pytest
from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate

@pytest.fixture
async def ai_engine():
    engine = AIDecisionEngine()
    await engine.start()
    yield engine
    await engine.stop()

async def test_incident_processing_pipeline(ai_engine):
    incident = IncidentCreate(
        title="Test Incident",
        description="Test database timeout",
        service="test-service",
        severity="high"
    )
    
    # Queue incident and wait for processing
    await ai_engine.queue_incident(incident)
    # Add assertions for expected behavior
```

### API Tests

Test HTTP endpoints:

```python
# Example: tests/api/test_incidents.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_create_incident():
    response = client.post("/api/incidents", json={
        "title": "Test Incident",
        "description": "Test description",
        "service": "test-service",
        "severity": "high"
    })
    assert response.status_code == 201
    assert "id" in response.json()
```

## Performance Monitoring

### Local Profiling

```python
# Profile AI engine performance
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run code to profile
await ai_engine.process_incident(incident)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

### Memory Monitoring

```python
# Monitor memory usage
import tracemalloc

tracemalloc.start()

# Run code
# ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Database Connection Issues**
```bash
# Check database is running
docker-compose ps postgres

# Test connection
python -c "from src.database import get_db_session; print('DB connected')"
```

**Model Loading Failures**
```bash
# Check model files exist
ls -la models/

# Test model loading
python -c "from src.services.ml_service import MLService; import asyncio; asyncio.run(MLService().initialize())"
```

**Port Conflicts**
```bash
# Check what's using port 8000
lsof -i :8000

# Use different port
export API_PORT=8001
```

### Getting Help

1. Check the troubleshooting section in documentation
2. Search existing GitHub issues
3. Create new issue with:
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce

## Contributing

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and returns
- Write docstrings for all public functions and classes
- Maintain test coverage above 90%

### Pull Request Process

1. Create feature branch from `main`
2. Implement changes with tests
3. Run full test suite
4. Update documentation if needed
5. Submit pull request with description

### Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release tag
4. Deploy to staging for testing
5. Deploy to production

This development guide provides everything needed to set up a local development environment and contribute to the AI On-Call Agent system.
