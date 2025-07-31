# On-Call Agent - Complete Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Database Configuration](#database-configuration)
4. [Module Documentation](#module-documentation)
5. [Service Layer](#service-layer)
6. [AI Engine](#ai-engine)
7. [API Layer](#api-layer)
8. [Monitoring System](#monitoring-system)
9. [Configuration Guide](#configuration-guide)
10. [Deployment Guide](#deployment-guide)

## Project Overview

The On-Call Agent is an intelligent incident management system that automatically detects, analyzes, and resolves operational issues using AI-powered decision making. The system monitors log streams, identifies patterns, executes automated remediation actions, and escalates issues to human operators when necessary.

### Key Features
- **Real-time Log Monitoring**: Monitors Elasticsearch and local log files for anomalies
- **AI-Powered Analysis**: Uses machine learning to analyze incidents and recommend actions
- **Automated Resolution**: Executes system commands to resolve common issues
- **Knowledge Base**: Learns from past incidents to improve future responses
- **Manual Escalation**: Sends notifications via email and Microsoft Teams when human intervention is needed
- **REST API**: Provides programmatic access to all system functionality

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Layer     │    │   AI Engine     │
│   (Optional)    │◄──►│   FastAPI       │◄──►│   Decision      │
└─────────────────┘    └─────────────────┘    │   Making        │
                                              └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   Log Sources   │    │   Monitoring    │◄──────────┘
│   • Elasticsearch │◄──►│   Service       │
│   • Local Files │    │                 │
└─────────────────┘    └─────────────────┘
                                              ┌─────────────────┐
┌─────────────────┐    ┌─────────────────┐   │   Notification  │
│   Database      │◄──►│   Service Layer │◄──┤   Services      │
│   • PostgreSQL  │    │   • Incidents   │   │   • Email/Teams │
│   • SQLite      │    │   • Actions     │   └─────────────────┘
└─────────────────┘    │   • Auth        │
                       └─────────────────┘
```

## Database Configuration

The system supports multiple database backends with easy configuration switching.

### Current Database: SQLite (Development)
**Location**: `./data/incidents.db`

**Changing Database Backend**:

1. **PostgreSQL Configuration**:
```python
# src/core/config.py
DATABASE_URL = "postgresql://username:password@localhost:5432/oncall_agent"
```

2. **MySQL Configuration**:
```python
# src/core/config.py  
DATABASE_URL = "mysql://username:password@localhost:3306/oncall_agent"
```

3. **Environment Variable**:
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/oncall_agent"
```

### Database Schema

The system uses SQLAlchemy ORM with the following main tables:

- **incidents**: Stores incident records with AI analysis
- **actions**: Tracks automated and manual actions taken
- **users**: User authentication and authorization
- **incident_logs**: Audit trail of all incident activities

## Module Documentation

### 1. Core Module (`src/core/`)

#### `config.py`
**Purpose**: Central configuration management using Pydantic settings
**Key Classes**:
- `Settings`: Main configuration class with database URL, logging, and feature flags

#### `database.py` 
**Purpose**: Database connection and session management
**Key Functions**:
- `get_db()`: Dependency injection for database sessions
- `init_db()`: Database initialization and table creation

#### `logger.py`
**Purpose**: Structured logging configuration using structlog
**Features**:
- JSON-formatted logs for production
- Contextual logging with incident IDs and metadata

### 2. Models Module (`src/models/`)

#### `database.py`
**Purpose**: SQLAlchemy ORM models for data persistence
**Key Models**:
- `Incident`: Core incident entity with AI analysis fields
- `Action`: Automated and manual actions tracking
- `User`: User authentication and session management

#### `schemas.py`
**Purpose**: Pydantic models for API validation and serialization
**Key Schemas**:
- `IncidentCreate`: Input validation for new incidents
- `IncidentResponse`: API response format for incidents
- `ActionCreate`: Action execution parameters
- `LogEntry`: Log monitoring data structure

### 3. AI Engine (`src/ai/__init__.py`)

**Purpose**: Core AI decision-making engine that analyzes incidents and orchestrates automated responses

#### Key Classes:

##### `AIEngine`
The main AI orchestration class that coordinates incident analysis and resolution.

**Key Methods**:
- `analyze_incident(incident: IncidentCreate) -> Dict[str, Any]`
  - Analyzes incident using multiple AI techniques
  - Returns confidence scores, root cause analysis, and recommended actions
  - Integrates with knowledge base for pattern matching

- `handle_incident(incident: IncidentCreate) -> bool`
  - Complete incident handling workflow
  - Determines if automation should be attempted
  - Executes actions or escalates to humans based on risk assessment

- `_execute_real_action(action: ActionCreate, incident: IncidentCreate) -> bool`
  - Executes automated remediation actions via ActionExecutionService
  - Provides real system command execution (no mocks)
  - Tracks action success/failure for learning

**AI Capabilities**:
- **Pattern Recognition**: Identifies known incident patterns using similarity matching
- **Risk Assessment**: Evaluates automation safety based on service criticality and change windows
- **Root Cause Analysis**: Analyzes log patterns and system context to identify likely causes
- **Action Recommendation**: Suggests specific remediation steps based on incident characteristics
- **Confidence Scoring**: Provides probabilistic assessment of resolution success

#### Integration Points:
- **Knowledge Base Service**: Searches for similar past incidents
- **Action Execution Service**: Executes real system commands
- **Notification Service**: Escalates to humans when needed

### 4. Service Layer (`src/services/`)

#### `notifications.py`
**Purpose**: Real notification system for manual escalation alerts

**Key Class**: `NotificationService`

**Methods**:
- `send_manual_escalation_alert(incident: Dict, analysis: Dict, reason: str) -> bool`
  - Sends notifications via email (SMTP) and Microsoft Teams webhooks
  - Includes incident details, AI analysis, and escalation reason
  - Configurable recipient lists via environment variables

**Configuration Environment Variables**:
```bash
SMTP_SERVER=smtp.company.com
SMTP_PORT=587
EMAIL_USER=alerts@company.com
EMAIL_PASSWORD=app_password
TEAMS_WEBHOOK_URL=https://company.webhook.office.com/...
ONCALL_EMAILS=oncall1@company.com,oncall2@company.com
ONCALL_PHONES=+1234567890,+0987654321
```

#### `knowledge_base.py` 
**Purpose**: Real knowledge base service with file-based storage and pattern matching

**Key Class**: `KnowledgeBaseService`

**Storage**: JSON file at `data/knowledge_base.json`

**Methods**:
- `search_similar_incidents(error_message: str, service: str, severity: str) -> List[KnowledgeBaseEntry]`
  - Real pattern matching against error messages and tags
  - Scoring algorithm based on pattern matches and service correlation
  - Returns ranked list of relevant past incidents

- `add_entry(entry: KnowledgeBaseEntry) -> bool`
  - Adds new incident patterns to knowledge base
  - Automatic learning from resolved incidents

**Default Knowledge Patterns**:
- Spark Out of Memory errors with executor memory adjustments
- Airflow task timeouts with configuration recommendations  
- Database connection pool exhaustion with restart procedures

#### `action_execution.py`
**Purpose**: Real action execution service that performs actual system operations

**Key Class**: `ActionExecutionService`

**Supported Actions**:
- `restart_service`: Uses systemctl/docker commands to restart services
- `health_check`: Performs HTTP health checks with aiohttp
- `cleanup_logs`: Executes find commands to remove old log files
- `adjust_memory_config`: Updates service configuration files
- `restart_connection_pool`: Database connection pool management
- `terminate_queries`: Database query termination

**Real Implementation Examples**:
```python
# Actual systemctl restart
await asyncio.create_subprocess_exec('sudo', 'systemctl', 'restart', service_name)

# Real HTTP health check
async with aiohttp.ClientSession() as session:
    async with session.get(f"http://{service}:8080/health") as response:
        return response.status == 200

# Actual log cleanup
await asyncio.create_subprocess_exec('find', '/var/log', '-name', '*.log', '-mtime', '+7', '-delete')
```

#### `auth.py`
**Purpose**: JWT-based authentication service

**Key Functions**:
- `create_access_token()`: Generates JWT tokens with user context
- `verify_token()`: Validates and decodes JWT tokens
- `get_current_user()`: FastAPI dependency for protected endpoints

**Token Structure**:
- Contains user ID, username, and expiration
- Configurable expiration time
- HMAC-SHA256 signing

#### `enhanced_incident_service.py`
**Purpose**: Enhanced incident management with AI integration

**Key Features**:
- AI-powered decision making for incident triage
- Automated action execution based on confidence thresholds
- Integration with knowledge base for pattern matching
- Comprehensive incident lifecycle management

### 5. Monitoring System (`src/monitoring/__init__.py`)

**Purpose**: Real log monitoring system with Elasticsearch and local file support

**Key Class**: `LogMonitorService`

**Capabilities**:
- **Elasticsearch Integration**: Queries ES for recent logs using real HTTP API
- **Local File Monitoring**: Fallback to reading local log files when ES unavailable
- **Pattern Matching**: Configurable regex patterns for anomaly detection
- **Real-time Processing**: Continuous monitoring with configurable intervals

**Methods**:
- `start_monitoring()`: Begins continuous log monitoring
- `_monitor_elasticsearch(url: str)`: Real Elasticsearch log retrieval
- `_monitor_local_logs()`: Local log file parsing as fallback
- `get_recent_logs()`: API for retrieving filtered log entries

**Log Sources**:
- Elasticsearch indices via HTTP API
- Local files: `/var/log/application.log`, `/var/log/error.log`, `logs/application.log`

**Alert Patterns**:
Configurable regex patterns for detecting specific error conditions:
- Database connection failures
- Out of memory errors  
- HTTP 5xx errors
- Authentication failures
- Service timeout errors

### 6. API Layer (`src/api/`)

#### `incidents.py`
**Purpose**: FastAPI routes for incident management

**Endpoints**:
- `POST /incidents/`: Create new incident
- `GET /incidents/{incident_id}`: Retrieve incident details
- `PUT /incidents/{incident_id}`: Update incident
- `GET /incidents/`: List incidents with filtering
- `POST /incidents/{incident_id}/resolve`: Manual resolution

#### `monitoring.py`
**Purpose**: FastAPI routes for log monitoring

**Endpoints**:
- `GET /logs/recent`: Get recent logs with filtering
- `GET /logs/statistics`: Log analysis and statistics
- `POST /logs/patterns`: Add custom alert patterns

#### `auth.py`
**Purpose**: Authentication endpoints

**Endpoints**:
- `POST /auth/login`: User authentication
- `POST /auth/refresh`: Token refresh
- `GET /auth/me`: Current user information

## Configuration Guide

### Environment Variables

```bash
# Database Configuration
DATABASE_URL=sqlite:///./data/incidents.db

# JWT Authentication  
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Notification Configuration
SMTP_SERVER=smtp.company.com
SMTP_PORT=587  
EMAIL_USER=alerts@company.com
EMAIL_PASSWORD=app_password
TEAMS_WEBHOOK_URL=https://company.webhook.office.com/...
ONCALL_EMAILS=oncall1@company.com,oncall2@company.com

# Monitoring Configuration
ELASTICSEARCH_URL=http://localhost:9200
LOG_LEVEL=INFO
MONITORING_INTERVAL=30

# AI Configuration
CONFIDENCE_THRESHOLD=0.7
MAX_AUTOMATION_ACTIONS=5
RISK_ASSESSMENT_ENABLED=true
```

### Feature Configuration

```python
# src/core/config.py
class Settings(BaseSettings):
    # Enable/disable AI automation
    enable_ai_automation: bool = True
    
    # Confidence threshold for automated actions
    automation_confidence_threshold: float = 0.7
    
    # Maximum actions per incident
    max_actions_per_incident: int = 5
    
    # Enable risk assessment
    enable_risk_assessment: bool = True
    
    # Monitoring interval in seconds
    monitoring_interval: int = 30
```

## Deployment Guide

### 1. Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from src.core.database import init_db; init_db()"

# Start development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Production Deployment

```bash
# Using Docker
docker build -t oncall-agent .
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/oncall \
  -e SECRET_KEY=production-secret \
  oncall-agent

# Using systemd service
sudo cp oncall-agent.service /etc/systemd/system/
sudo systemctl enable oncall-agent
sudo systemctl start oncall-agent
```

### 3. Database Migration

```python
# For production database changes
from alembic import command
from alembic.config import Config

config = Config("alembic.ini")
command.upgrade(config, "head")
```

## System Architecture Principles

### No Mock Implementations
The system has been completely refactored to eliminate all mock implementations:

- **Real Notification System**: SMTP and Teams webhook integration
- **Real Action Execution**: Actual system commands via subprocess
- **Real Knowledge Base**: File-based storage with pattern matching
- **Real Log Monitoring**: Elasticsearch API and file system integration
- **Real AI Analysis**: Complete incident analysis pipeline

### Scalability Considerations

- **Database**: Supports PostgreSQL for production scale
- **Monitoring**: Async processing with configurable intervals
- **Action Execution**: Queued execution to prevent system overload
- **Caching**: Knowledge base results cached for performance

### Security Features

- **JWT Authentication**: Secure API access
- **Input Validation**: Pydantic schema validation on all inputs
- **SQL Injection Protection**: SQLAlchemy ORM with parameterized queries
- **Command Injection Protection**: Validated parameters for system commands

### Observability

- **Structured Logging**: JSON logs with correlation IDs
- **Metrics Collection**: Action success rates and response times
- **Error Tracking**: Comprehensive exception handling and logging
- **Audit Trail**: Complete incident and action history

---

This documentation provides complete coverage of the system architecture, including database configuration options and detailed module-level documentation as requested. The system has been thoroughly cleaned of mock implementations and now provides real functionality across all components.
- Gunicorn WSGI server
- Security hardening

## 🚀 Starting the System

### Development Mode
```bash
# Option 1: Using startup script
bash scripts/start_dev.sh

# Option 2: Using Python script
python3 scripts/start.py --mode dev

# Option 3: Direct uvicorn
python3 -m uvicorn src.main:app --host localhost --port 8000 --reload
```

### Production Mode
```bash
# Option 1: Using startup script
bash scripts/start_prod.sh

# Option 2: Using Docker Compose
docker-compose up -d

# Option 3: Using Kubernetes
kubectl apply -f k8s/deployment.yaml

# Option 4: Direct gunicorn
python3 -m gunicorn --host 0.0.0.0 --port 8000 --workers 4 src.main:app
```

## 🧪 Testing

### Run All Tests
```bash
# System integration test
python3 system_test.py

# AI training and testing
python3 test_and_train.py

# Unit tests (dev mode only)
python3 -m pytest
```

### Training the AI Models
```bash
# Progressive training with reports
python3 test_and_train.py

# View training reports
ls reports/
```

## 📚 Documentation

- **User Guide**: `docs/AI_ON_CALL_AGENT_DUMMYS_GUIDE.md`
- **HTML Guide**: `docs/AI_ON_CALL_AGENT_DUMMYS_GUIDE.html`
- **API Documentation**: http://localhost:8000/docs (when running)
- **Training Reports**: `reports/` directory

## ⚙️ Configuration

### Environment Variables (`.env`)
```bash
# Application Settings
DEBUG=true
HOST=localhost
PORT=8000

# Database
DATABASE_URL=sqlite:///./data/oncall_agent.db

# AI Configuration
OPENAI_API_KEY=your-api-key-here

# Monitoring
POLL_INTERVAL=60
```

### Application Config (`config/config.json`)
```json
{
  "app": {
    "name": "AI On-Call Agent",
    "debug": true,
    "host": "localhost",
    "port": 8000
  },
  "ai": {
    "model": "gpt-3.5-turbo",
    "temperature": 0.3
  }
}
```

## 🏗️ Project Structure

```
on-call-agent/
├── 📁 src/                 # Source code
├── 📁 config/              # Configuration files
├── 📁 data/                # Training data and databases
├── 📁 docs/                # Documentation
├── 📁 logs/                # Application logs
├── 📁 reports/             # Training reports
├── 📁 scripts/             # Startup scripts
├── 📁 tests/               # Test files
├── 🐳 docker-compose.yml   # Docker setup
├── ☸️ k8s/                 # Kubernetes configs
├── ⚙️ setup.py             # Main setup script
├── 🚀 quick_setup.sh       # Quick setup script
└── 📋 requirements.txt     # Python dependencies
```

## 🔧 CLI Commands

```bash
# System status
python3 cli.py status

# Start services
python3 cli.py start-services

# Train models
python3 cli.py train

# View logs
python3 cli.py logs

# Help
python3 cli.py --help
```

## 🚀 API Endpoints

Once running, the system provides:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Dashboard**: http://localhost:8000/ (if UI enabled)

## 🐳 Docker Deployment

### Development
```bash
# Build and run
docker build -t oncall-agent .
docker run -p 8000:8000 oncall-agent
```

### Production
```bash
# Using Docker Compose
docker-compose up -d

# Scale services
docker-compose up -d --scale oncall-agent=3
```

## ☸️ Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods

# View logs
kubectl logs -f deployment/oncall-agent
```

## 📊 Monitoring

### Built-in Monitoring
- **Health checks**: `/health` endpoint
- **Metrics**: `/metrics` endpoint (Prometheus format)
- **Logs**: Structured logging to files and stdout

### External Monitoring
- **Prometheus**: http://localhost:9090 (with Docker Compose)
- **Grafana**: Configure dashboards for metrics visualization
- **ELK Stack**: For log aggregation and analysis

## 🔒 Security

### Development
- SQLite database (local file)
- Debug mode enabled
- No authentication required

### Production
- PostgreSQL with authentication
- JWT token authentication
- Rate limiting
- HTTPS recommended
- Environment variable secrets

## 🤝 Contributing

### Development Setup
```bash
# Setup development environment
./quick_setup.sh dev

# Install pre-commit hooks (auto-installed in dev mode)
pre-commit install

# Run tests before committing
python3 -m pytest
python3 system_test.py
```

### Code Quality
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Unit testing
- **Pre-commit**: Git hooks

## 📦 Dependencies

### Core Dependencies
- FastAPI - Web framework
- Uvicorn - ASGI server
- SQLAlchemy - Database ORM
- Redis - Caching
- Celery - Task queue

### AI/ML Dependencies
- Scikit-learn - Machine learning
- OpenAI - GPT integration
- Pandas/Numpy - Data processing
- Transformers - NLP models

### Development Dependencies
- Pytest - Testing framework
- Black - Code formatter
- Flake8 - Linter
- Jupyter - Notebooks

## 🆘 Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   # Reinstall dependencies
   pip3 install -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   # Change port in .env file
   PORT=8001
   ```

3. **Database Connection Issues**
   ```bash
   # Reset database (dev mode)
   rm data/oncall_agent.db
   python3 -c "from src.database import init_db; init_db()"
   ```

4. **Permission Issues**
   ```bash
   # Fix script permissions
   chmod +x scripts/*.sh
   chmod +x quick_setup.sh
   ```

### Getting Help

1. **Check the user guide**: `docs/AI_ON_CALL_AGENT_DUMMYS_GUIDE.md`
2. **Run system test**: `python3 system_test.py`
3. **Check logs**: `tail -f logs/oncall_agent.log`
4. **View API docs**: http://localhost:8000/docs

## 📈 Performance

### Development
- SQLite database
- Single process
- Debug overhead
- ~100 requests/second

### Production
- PostgreSQL database
- Multi-worker (4 workers default)
- Optimized settings
- ~1000+ requests/second

## 🎯 Next Steps

After setup:

1. **Configure AI**: Add your OpenAI API key to `.env`
2. **Train Models**: Run `python3 test_and_train.py`
3. **Setup Monitoring**: Configure your log sources
4. **Test Integration**: Add your infrastructure endpoints
5. **Deploy**: Move to production environment

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Scikit-learn for machine learning capabilities
- OpenAI for GPT integration
- The open-source community for amazing tools

---

**🎉 Ready to revolutionize your infrastructure monitoring with AI!**

For detailed usage instructions, see the comprehensive user guide in `docs/AI_ON_CALL_AGENT_DUMMYS_GUIDE.md`.
graph TD
    A[Log Sources] --> B[Log Poller]
    B --> C[Pattern Matching]
    C --> D{Error Detected?}
    D -->|Yes| E[Create Incident]
    D -->|No| F[Continue Monitoring]
    E --> G[Knowledge Base Lookup]
    G --> H[Auto-Resolution]
    H --> I[Execute Action]
    I --> J[Monitor Success]
```

### **🚀 Quick Demo**

```bash
# 1. Create test log files with errors
python cli.py create-test-logs

# 2. Test pattern detection
python cli.py test-log-polling

# 3. Start real-time monitoring
python -m src.main --dev
```

**Watch the magic happen:**
- 📊 **Log entries scanned**: File-based logs every 15 seconds
- 🎯 **Error patterns detected**: Memory, connection, disk, service issues
- ⚡ **Incidents created**: Automatically from detected problems
- 🤖 **Solutions applied**: Based on knowledge base matches
- 📈 **System learns**: Success rates improve resolution confidence

### **📝 Example: Automatic Problem Resolution**

**1. Log Entry Detected:**
```log
2024-07-28 10:32:05 [CRITICAL] redis: OutOfMemoryError: heap space exhausted
```

**2. AI Analysis:**
```json
{
  "pattern_matched": "out_of_memory",
  "severity": "high", 
  "service": "redis",
  "confidence": 0.92,
  "recommended_action": "restart_service"
}
```

**3. Automatic Resolution:**
- ✅ Incident created automatically
- 🔍 Knowledge base searched for similar issues
- 🎯 Solution found: "restart_redis_service"  
- ⚡ Action executed automatically
- 📊 Success monitored and logged
- 📈 Pattern confidence increased

### **🎯 Built-in Alert Patterns**

| Pattern | Example Log | Auto-Action |
|---------|-------------|-------------|
| **Memory Issues** | `OutOfMemoryError: heap space` | Restart app, increase memory |
| **Connection Failures** | `connection timeout after 30s` | Restart service, check network |
| **Service Unavailable** | `health check failed` | Restart service, scale resources |
| **Disk Full** | `no space left on device` | Clean logs, expand storage |
| **Database Errors** | `connection pool exhausted` | Restart connections, scale pool |

### **🛠️ Test Pattern Detection**

```bash
# Test your log messages against patterns
python cli.py test-log-pattern "OutOfMemoryError.*heap" "ERROR: OutOfMemoryError Java heap space"
# ✅ MATCH FOUND

python cli.py test-log-pattern "connection.*timeout" "Database connection timeout after 30 seconds"  
# ✅ MATCH FOUND

# Add custom patterns for your apps
python cli.py add-log-pattern \
    --name "payment_failure" \
    --pattern "payment.*failed|transaction.*error" \
    --severity "high"
```

## 🏗️ Architecture

### Core Components

1. **Log Monitor Service**: Real-time log ingestion and analysis
2. **AI Decision Engine**: Pattern matching and issue classification using OpenAI
3. **Knowledge Base**: Repository of known issues and automated fixes
4. **Action Engine**: Executes remediation actions safely
5. **API Service**: RESTful API for system integration
6. **CLI Tool**: Command-line interface for development and testing

### Technology Stack

- **Python 3.11+** with FastAPI for API services
- **SQLite/PostgreSQL** for persistent storage
- **Redis** for caching and session management
- **Docker** for containerization
- **OpenAI GPT** for intelligent log analysis
- **Airflow Integration** via REST API
- **Spark Monitoring** via Spark History Server API

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL database
- Redis cache
- OpenAI API key
- Airflow instance (for ETL monitoring)
- Spark cluster (optional)

### 1. Environment Setup

**⚠️ Important**: The system requires proper environment configuration before startup.

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# At minimum, configure these REQUIRED variables:
# - DATABASE_PASSWORD (secure database password)
# - OPENAI_API_KEY (for AI features)
# - SECRET_KEY & JWT_SECRET_KEY (32+ character security keys)
# - AIRFLOW_BASE_URL, AIRFLOW_USERNAME, AIRFLOW_PASSWORD
# - ALERT_EMAIL_FROM, ALERT_EMAIL_TO
```

📋 **See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for complete configuration guide**

### 2. Installation & Startup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy redis openai structlog rich typer "python-jose[cryptography]" "passlib[bcrypt]" aiosqlite greenlet

# Start the system (recommended - includes validation)
python start.py

# Alternative: Direct start
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 3. Access the System

- **Web UI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs  
- **Health Check**: http://localhost:8000/health
```bash
# Development mode
python -m src.main --dev

# Or use the CLI
python cli.py start-services
```

### Configuration

Key environment variables in `.env`:

```env
# Application
DEBUG=true
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here-must-be-at-least-32-characters-long
JWT_SECRET_KEY=your-jwt-secret-key-here-must-be-at-least-32-characters-long

# Database
DATABASE_URL=sqlite+aiosqlite:///./test.db

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# OpenAI (required for AI features)
OPENAI_API_KEY=your_openai_api_key_here

# Monitoring
ELASTICSEARCH_URLS=["http://localhost:9200"]

# Airflow Integration
AIRFLOW_BASE_URL=http://localhost:8080
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin

# Spark Integration
SPARK_HISTORY_SERVER_URL=http://localhost:18080
```

## 🛠️ Features

### Automated Issue Resolution

The system can automatically handle:

- **Airflow DAG failures**: Restart failed tasks, clear task instances
- **Spark job failures**: Restart applications, adjust resource allocation
- **Data pipeline issues**: Retry failed ingestion, clean up temporary files
- **Service health checks**: Restart unhealthy services, scale resources
- **Database connection issues**: Restart connection pools, switch to backup databases

### Intelligent Monitoring

- **Real-time log analysis** using AI pattern recognition
- **Anomaly detection** for unusual system behavior
- **Predictive alerts** before issues escalate
- **Performance optimization** recommendations
- **Cost optimization** suggestions

### Knowledge Management

- **Issue categorization** with automatic tagging
- **Solution effectiveness tracking** and optimization
- **Learning from new incidents** to expand automation
- **Team knowledge sharing** and collaboration

## 📊 API Documentation

Once running, visit:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root Endpoint**: http://localhost:8000

## 🧪 Development & Testing

### CLI Tool

The system includes a powerful CLI for development:

```bash
# Check system status
python cli.py status

# List incidents
python cli.py incidents

# View knowledge base
python cli.py knowledge

# List recent actions
python cli.py actions

# Simulate an incident
python cli.py simulate-incident --title "Database timeout" --service "postgres" --severity "high"

# Test an action
python cli.py test-action --action-type "restart_service" --service-name "web-api"

# Monitor logs
python cli.py monitor-logs --duration 60

# Start all services
python cli.py start-services
```

## 🎯 Hands-On Tutorial: Training Your AI Agent

### Step 1: Initial Setup and Baseline

```bash
# 1. Start with a fresh system
python cli.py reset-knowledge-base --confirm

# 2. Check initial AI capabilities
python cli.py ai-status
# Expected: Limited knowledge, low confidence scores

# 3. Simulate a baseline incident
python cli.py simulate-incident \
    --title "Redis Connection Failed" \
    --description "Redis server not responding to ping" \
    --service "redis" \
    --severity "high"

# Expected: No automatic resolution, creates incident for manual handling
```

### Step 2: Manual Resolution and Teaching

```bash
# 4. Manually resolve the incident (simulating human intervention)
python cli.py resolve-incident 1 \
    --solution "restart_redis_service" \
    --description "Restarted Redis service, connections restored" \
    --resolution-time 300 \
    --success true

# 5. Add this pattern to knowledge base
python cli.py add-knowledge \
    --title "Redis Connection Failure" \
    --pattern "redis.*not responding|redis.*connection.*failed" \
    --category "connectivity" \
    --solution "restart_redis_service" \
    --confidence 0.7 \
    --tags "redis,connection,restart"
```

### Step 3: Test Learning

```bash
# 6. Simulate the same type of incident again
python cli.py simulate-incident \
    --title "Redis Server Down" \
    --description "Cannot connect to Redis instance" \
    --service "redis" \
    --severity "high"

# Expected: AI should now recognize pattern and suggest/execute solution

# 7. Check if learning occurred
python cli.py incidents --status resolved --auto-resolved
python cli.py knowledge search --query "redis"
```

### Step 4: Advanced Pattern Training

```bash
# 8. Train with variations to improve pattern recognition
python cli.py simulate-incident \
    --title "Cache Service Unavailable" \
    --description "Redis cluster node failed health check" \
    --service "redis-cluster" \
    --severity "medium"

# 9. Add related knowledge
python cli.py add-knowledge \
    --title "Redis Cluster Node Failure" \
    --pattern "redis.*cluster.*node.*failed|cache.*unavailable" \
    --category "connectivity" \
    --solution "restart_redis_service" \
    --confidence 0.8 \
    --parent-pattern "redis_connection_failure"

# 10. Test pattern generalization
python cli.py test-pattern \
    --text "ERROR: Cache service redis-prod-01 health check failed" \
    --expected-category "connectivity"
```

### Step 5: Confidence Tuning

```bash
# 11. Monitor success rates
python cli.py action-stats --action-type "restart_redis_service"

# 12. Adjust confidence based on success rate
python cli.py update-confidence \
    --pattern "redis_connection_failure" \
    --new-confidence 0.85 \
    --reason "95% success rate over 20 incidents"

# 13. Set automatic learning
python cli.py enable-auto-learning \
    --pattern-types "connectivity,performance" \
    --confidence-threshold 0.8 \
    --require-multiple-successes 3
```

### Step 6: Bulk Training with Historical Data

```bash
# 14. Create sample historical data file
cat > historical_incidents.csv << EOF
timestamp,title,description,service,severity,resolution_time,solution,tags,success
2024-01-01T10:00:00Z,"DB Timeout","PostgreSQL query timeout","postgres","high",600,"restart_db","database,timeout",true
2024-01-02T14:30:00Z,"Disk Full","Root filesystem 98% full","server","critical",1200,"cleanup_logs","disk,storage",true
2024-01-03T09:15:00Z,"API Rate Limit","Too many requests to external API","api-gateway","medium",300,"enable_backoff","api,rate-limit",true
2024-01-03T16:45:00Z,"Memory Leak","Java heap space exhausted","spark","high",900,"restart_application","memory,spark",true
2024-01-04T11:20:00Z,"Network Timeout","Cannot reach downstream service","network","medium",450,"retry_with_circuit_breaker","network,timeout",true
EOF

# 15. Import and learn from historical data
python cli.py import-incidents --file historical_incidents.csv --auto-learn

# 16. Verify knowledge base growth
python cli.py knowledge stats
# Expected: 5+ new patterns, increased coverage
```

### Step 7: Testing and Validation

```bash
# 17. Run comprehensive test suite
python cli.py test-ai-learning \
    --simulate-incidents 10 \
    --pattern-types "database,memory,network" \
    --measure-accuracy

# Sample output:
# AI Learning Test Results:
# ========================
# Incidents Simulated: 10
# Correctly Identified: 8 (80%)
# Auto-Resolved: 6 (60%)
# False Positives: 1 (10%)
# Average Confidence: 0.78
# Average Resolution Time: 145 seconds

# 18. Validate specific scenarios
python cli.py validate-scenario \
    --scenario "database_outage" \
    --expected-actions "restart_db,check_connections,notify_team" \
    --confidence-threshold 0.75
```

### Step 8: Production Readiness

```bash
# 19. Export trained model for backup
python cli.py export-knowledge \
    --output trained_model_$(date +%Y%m%d).json \
    --include-metrics \
    --include-confidence-history

# 20. Set production thresholds
python cli.py configure-production \
    --auto-resolve-threshold 0.85 \
    --escalation-threshold 0.60 \
    --max-actions-per-hour 10 \
    --enable-human-approval-for-critical

# 21. Final validation
python cli.py production-readiness-check
# Expected: PASS on all critical checks
```

## 📊 Training Data Examples

### Sample Log Patterns for Training

```bash
# Database Issues
python cli.py add-pattern --pattern "connection.*timeout.*database|database.*connection.*failed" --category "database_connectivity"
python cli.py add-pattern --pattern "deadlock.*detected|lock.*timeout" --category "database_performance"
python cli.py add-pattern --pattern "disk.*full.*database|no.*space.*left" --category "database_storage"

# Memory Issues  
python cli.py add-pattern --pattern "OutOfMemoryError|heap.*space.*exhausted" --category "memory_exhaustion"
python cli.py add-pattern --pattern "memory.*leak.*detected|gc.*overhead.*limit" --category "memory_leak"

# Network Issues
python cli.py add-pattern --pattern "connection.*reset|network.*unreachable" --category "network_connectivity"
python cli.py add-pattern --pattern "timeout.*waiting.*response|read.*timeout" --category "network_timeout"

# Application Issues
python cli.py add-pattern --pattern "application.*failed.*start|service.*unavailable" --category "service_failure"
python cli.py add-pattern --pattern "configuration.*error|invalid.*config" --category "configuration_error"
```

### Sample Solutions for Common Issues

```bash
# Restart-based solutions
python cli.py add-solution \
    --id "restart_database" \
    --name "Restart Database Service" \
    --type "service_restart" \
    --command "systemctl restart postgresql" \
    --timeout 60 \
    --success-check "systemctl is-active postgresql"

# API-based solutions  
python cli.py add-solution \
    --id "scale_application" \
    --name "Scale Application Instances" \
    --type "api_call" \
    --endpoint "POST /api/applications/{app_id}/scale" \
    --payload '{"instances": 3, "memory": "2GB"}' \
    --success-check "GET /api/applications/{app_id}/health"

# Command-based solutions
python cli.py add-solution \
    --id "cleanup_disk_space" \
    --name "Clean Up Disk Space" \
    --type "shell_command" \
    --command "find /var/log -name '*.log' -mtime +7 -delete && docker system prune -f" \
    --success-check "df -h | awk '/\// {if($5+0 < 90) print \"OK\"}'"
```

### Project Structure

```
src/
├── api/                 # FastAPI routes and endpoints
├── core/               # Core business logic and configuration
├── models/             # Database models and Pydantic schemas
├── services/           # Business logic services
├── ai/                 # AI/ML components for decision making
├── monitoring/         # Log monitoring and analysis
├── actions/           # Automated action executors
└── database/          # Database initialization and management

tests/                 # Test suite (to be added)
docs/                  # Documentation
docker/               # Docker configurations
cli.py                # Command-line interface
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests (when implemented)
pytest tests/ -v --cov=src
```

## 🐳 Docker Deployment

For production deployment with full infrastructure:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f oncall-agent

# Stop services
docker-compose down
```

The Docker setup includes:
- PostgreSQL database
- Redis cache
- Elasticsearch for log storage
- Kibana for log visualization
- Prometheus for metrics
- Grafana for dashboards

## 🔒 Security Considerations

- All automated actions are **logged and auditable**
- **Rate limiting** prevents runaway automation
- **Circuit breakers** stop cascading failures
- **Manual override** capabilities for emergency situations
- **Secure credential management** for external services
- **Network isolation** for sensitive operations

## 📈 Monitoring and Alerting

The system monitors itself and provides:

- **Health dashboards** for all components
- **Performance metrics** and SLA tracking
- **Alert escalation** when automation fails
- **Audit logs** for all automated actions
- **Cost tracking** for cloud resources

## 🎯 Current Status

✅ **Core Infrastructure**: Complete and functional
✅ **Log Monitoring**: Pattern detection working
✅ **AI Decision Engine**: Basic framework implemented
✅ **Action Engine**: Automated action execution
✅ **Knowledge Base**: CRUD operations and search
✅ **REST API**: FastAPI with comprehensive endpoints
✅ **CLI Tool**: Full development and testing interface
✅ **Database**: SQLite for development, PostgreSQL for production
✅ **Authentication**: JWT-based security

## 🔄 Next Steps

### Immediate Priorities

1. **Integrate real log sources** (Elasticsearch, Splunk)
2. **Connect to actual Airflow and Spark APIs**
3. **Implement OpenAI integration** for log analysis
4. **Add comprehensive test suite**
5. **Create web dashboard UI**

### Future Enhancements

1. **Machine learning models** for predictive maintenance
2. **Advanced workflow orchestration**
3. **Multi-cloud support**
4. **Custom alert routing**
5. **Integration with incident management systems**

## 🧠 How the AI Learning System Works

### Learning Architecture

The AI system uses a multi-layered approach to learn from incidents and improve over time:

1. **Pattern Recognition Engine**: Uses OpenAI GPT to analyze log patterns and extract meaningful insights
2. **Knowledge Base Evolution**: Automatically updates solutions based on success/failure rates
3. **Confidence Scoring**: Tracks the effectiveness of automated actions
4. **Feedback Loop**: Incorporates human feedback to refine decision-making

### Learning Process Flow

```mermaid
graph TD
    A[Log Event Detected] --> B[AI Pattern Analysis]
    B --> C[Knowledge Base Search]
    C --> D{Match Found?}
    D -->|Yes| E[Execute Known Solution]
    D -->|No| F[Create New Incident]
    E --> G[Monitor Outcome]
    F --> H[Human Resolution]
    G --> I[Update Success Rate]
    H --> J[Learn New Pattern]
    I --> K[Adjust Confidence Score]
    J --> L[Add to Knowledge Base]
    K --> M[Improve Future Decisions]
    L --> M
```

### 📊 Training the AI with Historical Data

#### 1. Bulk Import Past Incidents

```bash
# Import historical incidents from CSV
python cli.py import-incidents --file historical_incidents.csv

# Example CSV format:
# timestamp,title,description,service,severity,resolution_time,solution,tags
# 2024-01-15T10:30:00Z,"DB Connection Failed","PostgreSQL timeout errors","postgres","high",1800,"restart_service","database,timeout"
```

#### 2. Feed Real-time Log Data

```python
# Using the API to feed log data
import requests

# Send log entries for analysis
log_data = {
    "timestamp": "2024-01-15T10:30:00Z",
    "service": "airflow-scheduler",
    "level": "ERROR", 
    "message": "Task instance failed: Out of memory",
    "metadata": {
        "dag_id": "data_pipeline",
        "task_id": "spark_processing",
        "memory_usage": "8GB"
    }
}

response = requests.post("http://localhost:8000/api/logs/analyze", json=log_data)
```

#### 3. Manual Pattern Teaching

```bash
# Teach the AI about new error patterns
python cli.py add-pattern \
    --pattern "OutOfMemoryError.*heap space" \
    --category "memory_issue" \
    --solution "increase_memory" \
    --confidence 0.8 \
    --description "Java heap space exhaustion"

# Add solution templates
python cli.py add-solution \
    --id "increase_memory" \
    --name "Increase Memory Allocation" \
    --type "api_call" \
    --endpoint "/spark/applications/{app_id}/resize" \
    --params '{"memory": "16GB", "cores": 4}' \
    --description "Increase Spark application memory"
```

## 📝 Detailed Example Workflows

### Example 1: Automatic Database Connection Recovery

**Scenario**: PostgreSQL connection pool exhaustion during peak traffic

```bash
# 1. Log Detection (automatic)
[2024-01-15 10:30:15] ERROR [postgres-pool] Connection timeout after 30s
[2024-01-15 10:30:16] ERROR [api-service] Database connection failed: FATAL: remaining connection slots are reserved
```

**AI Analysis Process**:
```python
# AI pattern matching (internal process)
{
    "pattern_match": "connection_timeout_database",
    "confidence": 0.92,
    "service": "postgres", 
    "severity": "high",
    "suggested_actions": [
        {"action": "restart_connection_pool", "confidence": 0.85},
        {"action": "scale_database", "confidence": 0.70}
    ]
}
```

**Automated Resolution**:
```bash
# 2. Knowledge base lookup
python cli.py knowledge search --query "postgres connection timeout"

# 3. Action execution (automatic)
POST /api/actions/execute
{
    "action_type": "restart_service",
    "service_name": "postgres-connection-pool",
    "parameters": {"graceful": true, "timeout": 30}
}

# 4. Verification and learning
python cli.py monitor-service --service postgres --duration 300
# Success rate: 95% -> Increase confidence score to 0.87
```

### Example 2: Spark Job Memory Optimization

**Scenario**: Recurring OutOfMemoryError in Spark data processing

```bash
# Initial failure pattern
[2024-01-15 11:00:00] ERROR [spark-executor] java.lang.OutOfMemoryError: Java heap space
[2024-01-15 11:00:01] ERROR [spark-driver] Application application_1705317600000_0042 failed
```

**Learning Evolution**:

**First Occurrence** (No knowledge):
```bash
# Creates new incident for human resolution
python cli.py simulate-incident \
    --title "Spark OOM Error" \
    --service "spark" \
    --severity "high" \
    --auto-learn true

# Human provides solution
python cli.py resolve-incident 12345 \
    --solution "increase_executor_memory" \
    --action-taken "Increased executor memory from 4GB to 8GB" \
    --success true
```

**Second Occurrence** (Pattern recognized):
```bash
# AI detects similar pattern with 0.75 confidence
# Automatically applies learned solution
POST /api/spark/applications/resize
{
    "executor_memory": "8GB",
    "executor_cores": 2,
    "reason": "OOM prevention based on pattern analysis"
}
```

**Third+ Occurrences** (Optimized):
```bash
# Confidence increased to 0.90 after successful resolutions
# Now includes predictive action before failure
python cli.py set-alert \
    --pattern "memory_usage > 85%" \
    --service "spark" \
    --action "preemptive_scale" \
    --confidence 0.90
```

### Example 3: Airflow DAG Failure Recovery

**Complete Learning Cycle**:

```bash
# 1. Train with historical DAG failures
python cli.py import-airflow-logs \
    --start-date 2024-01-01 \
    --end-date 2024-01-15 \
    --include-failures \
    --auto-categorize

# 2. Real-time failure detection
# Log: "Task 'data_validation' failed with exit code 1"

# 3. AI decision process
{
    "incident_type": "dag_task_failure",
    "failure_pattern": "data_validation_error",
    "historical_success_rate": {
        "retry_task": 0.85,
        "clear_downstream": 0.60,
        "restart_dag": 0.40
    },
    "recommended_action": "retry_task",
    "confidence": 0.85
}

# 4. Automated execution
POST /api/airflow/dags/data_pipeline/dagRuns/2024-01-15T10:00:00/taskInstances/data_validation/retry

# 5. Learning feedback
{
    "incident_id": "inc_67890",
    "action_taken": "retry_task", 
    "outcome": "success",
    "resolution_time": 120,
    "update_confidence": true  # Increases to 0.87
}
```

## 🎓 Advanced Learning Techniques

### 1. Contextual Learning

The AI considers multiple factors when learning:

```python
# Context-aware pattern matching
learning_context = {
    "time_of_day": "peak_hours",      # 9am-5pm = different solutions
    "system_load": "high",            # CPU/Memory usage affects decisions
    "recent_deployments": True,        # New code = different failure modes
    "historical_incidents": 5,         # Similar issues in past 24h
    "user_feedback": "positive"        # Human validation of actions
}
```

### 2. Confidence Calibration

```bash
# View AI confidence metrics
python cli.py ai-metrics

# Output:
# Pattern Recognition Accuracy: 94.2%
# Action Success Rate: 87.5%
# False Positive Rate: 2.1%
# Learning Velocity: 12 new patterns/week

# Adjust confidence thresholds
python cli.py set-confidence-threshold \
    --action-type "restart_service" \
    --min-confidence 0.80 \
    --require-approval false

python cli.py set-confidence-threshold \
    --action-type "scale_infrastructure" \
    --min-confidence 0.95 \
    --require-approval true
```

### 3. Feedback Integration

```bash
# Provide feedback on AI decisions
python cli.py feedback \
    --incident-id 12345 \
    --action-correct true \
    --speed-rating 5 \
    --effectiveness-rating 4 \
    --comments "Good solution but could be faster"

# Mark false positives for learning
python cli.py mark-false-positive \
    --incident-id 12346 \
    --reason "Not actually an error, expected behavior" \
    --update-patterns true
```

### 4. Transfer Learning

```bash
# Import patterns from similar environments
python cli.py import-knowledge \
    --source-environment "staging" \
    --pattern-types "database,network" \
    --confidence-adjustment -0.1  # Reduce confidence for cross-env patterns

# Export learned patterns to other instances
python cli.py export-knowledge \
    --destination "prod-backup-agent" \
    --include-confidence-scores \
    --format json
```

## 📈 Measuring Learning Effectiveness

### Key Metrics Dashboard

```bash
# View learning progress
python cli.py learning-report --days 30

# Sample output:
# Learning Metrics (Last 30 Days):
# ================================
# New Patterns Learned: 24
# Confidence Improvements: 156
# Successful Auto-Resolutions: 89.3%
# Mean Time to Resolution: 2.4 minutes
# False Positive Rate: 1.8%
# Knowledge Base Growth: +15.2%
#
# Top Learning Categories:
# 1. Database Issues (42 incidents, 94% success)
# 2. Memory Errors (28 incidents, 87% success) 
# 3. Network Timeouts (19 incidents, 91% success)
```

### Learning Quality Indicators

```python
# API endpoint to check learning health
GET /api/ai/learning-health

# Response:
{
    "learning_velocity": "optimal",           # new patterns per week
    "pattern_diversity": "good",              # variety of learned patterns  
    "confidence_calibration": "well_tuned",   # accuracy vs confidence correlation
    "human_feedback_ratio": 0.12,            # 12% of actions get human feedback
    "knowledge_base_coverage": 0.78,         # 78% of incidents have known patterns
    "recommendation": "Consider expanding training data for network issues"
}
```

## 🔍 Monitoring AI Performance

### Real-time Learning Dashboard

```bash
# Start the learning monitor
python cli.py start-learning-monitor --dashboard

# Access dashboard at: http://localhost:8000/ai-dashboard
```

The dashboard shows:
- **Pattern Recognition Accuracy**: Live accuracy metrics
- **Learning Velocity**: New patterns learned over time
- **Confidence Calibration**: How well confidence scores predict success
- **Action Success Rates**: Effectiveness of automated actions
- **Knowledge Base Growth**: Coverage and diversity metrics

### Performance Tracking Commands

```bash
# Get detailed AI performance report
python cli.py ai-performance-report --format detailed

# Sample output:
# AI Performance Report - Last 30 Days
# ====================================
# 
# PATTERN RECOGNITION:
# - Total Patterns: 127
# - New This Period: 23
# - Recognition Accuracy: 94.2%
# - False Positive Rate: 2.1%
# 
# AUTOMATED ACTIONS:
# - Total Actions: 456
# - Success Rate: 89.3%
# - Average Resolution Time: 2.4 minutes
# - Human Intervention Required: 8.7%
# 
# LEARNING METRICS:
# - Knowledge Base Growth: +18.1%
# - Confidence Improvements: 167
# - Pattern Generalizations: 12
# - Cross-service Learnings: 8
# 
# TOP PERFORMING CATEGORIES:
# 1. Database Issues (96% success, 89 incidents)
# 2. Memory Problems (91% success, 67 incidents) 
# 3. Network Timeouts (88% success, 45 incidents)
# 
# AREAS FOR IMPROVEMENT:
# - Configuration Errors (73% success) - Need more training data
# - Security Alerts (65% success) - Consider manual review threshold
```

### A/B Testing AI Decisions

```bash
# Set up A/B test for competing solutions
python cli.py setup-ab-test \
    --pattern "database_timeout" \
    --solution-a "restart_database" \
    --solution-b "increase_connection_pool" \
    --split-ratio 0.5 \
    --duration-days 7

# Monitor A/B test results
python cli.py ab-test-results --test-id "db_timeout_test_001"

# Sample results:
# A/B Test Results: Database Timeout Solutions
# ============================================
# Test Duration: 7 days
# Total Incidents: 24
# 
# Solution A (restart_database):
# - Incidents: 12
# - Success Rate: 91.7% (11/12)
# - Avg Resolution Time: 45 seconds
# - Side Effects: 1 (brief downtime)
# 
# Solution B (increase_connection_pool):
# - Incidents: 12  
# - Success Rate: 75.0% (9/12)
# - Avg Resolution Time: 15 seconds
# - Side Effects: 0
# 
# RECOMMENDATION: Continue with Solution A
# CONFIDENCE: 0.85 (based on success rate difference)
```

## 🎯 Best Practices for AI Training

### 1. Data Quality Guidelines

```bash
# Validate training data quality
python cli.py validate-training-data \
    --check-duplicates \
    --check-missing-fields \
    --check-pattern-conflicts \
    --report-path validation_report.json

# Clean and standardize data
python cli.py clean-training-data \
    --remove-duplicates \
    --standardize-timestamps \
    --normalize-service-names \
    --merge-similar-patterns
```

### 2. Incremental Learning Strategy

```bash
# Set up progressive learning stages
python cli.py configure-learning-stages \
    --stage-1 "basic_patterns,confidence_0.6" \
    --stage-2 "complex_patterns,confidence_0.75" \
    --stage-3 "cross_service_patterns,confidence_0.85" \
    --advancement-criteria "success_rate_90_percent"

# Monitor learning stage progression
python cli.py learning-stage-status
# Current Stage: 2 (Complex Patterns)
# Progress: 78% (156/200 required successful resolutions)
# ETA to Stage 3: 4.2 days (at current incident rate)
```

### 3. Continuous Improvement Loop

```bash
# Schedule automated learning reviews
python cli.py schedule-learning-review \
    --frequency weekly \
    --actions "update_confidence,prune_ineffective_patterns,suggest_new_patterns" \
    --notification-email "devops@company.com"

# Manual learning optimization
python cli.py optimize-patterns \
    --merge-similar \
    --remove-low-confidence \
    --suggest-generalizations \
    --dry-run false
```

### 4. Safety and Guardrails

```bash
# Set up learning safety limits
python cli.py configure-safety-limits \
    --max-new-patterns-per-day 5 \
    --max-confidence-increase-per-week 0.1 \
    --require-human-approval-for "critical_systems,financial_data" \
    --rollback-on-failure-rate 0.3

# Emergency learning rollback
python cli.py rollback-learning \
    --to-date "2024-01-15" \
    --preserve-successful-patterns \
    --reason "Performance degradation after recent learning"
```

## 🚨 Troubleshooting AI Learning Issues

### Common Issues and Solutions

```bash
# Issue: Low pattern recognition accuracy
python cli.py diagnose-pattern-accuracy \
    --min-accuracy 0.85 \
    --analyze-failures \
    --suggest-improvements

# Potential solutions:
# - More diverse training data needed
# - Pattern conflicts detected  
# - Feature engineering required

# Issue: Overconfident AI making wrong decisions
python cli.py calibrate-confidence \
    --target-accuracy 0.90 \
    --method "platt_scaling" \
    --validation-split 0.2

# Issue: Slow learning on new patterns
python cli.py optimize-learning-rate \
    --current-rate 0.01 \
    --test-rates "0.005,0.02,0.05" \
    --evaluation-metric "pattern_acquisition_speed"
```

### Learning Health Checks

```bash
# Comprehensive AI health check
python cli.py ai-health-check --comprehensive

# Sample output:
# AI System Health Check
# ======================
# 
# ✅ Pattern Recognition: HEALTHY
#    - Accuracy: 94.2% (Target: >90%)
#    - Response Time: 0.3s (Target: <1s)
# 
# ⚠️  Learning Velocity: ATTENTION NEEDED  
#    - New Patterns/Week: 3.2 (Target: 5-10)
#    - Recommendation: Increase training data diversity
# 
# ✅ Knowledge Base: HEALTHY
#    - Coverage: 87% (Target: >80%)
#    - Pattern Conflicts: 0 (Target: 0)
# 
# ✅ Action Execution: HEALTHY
#    - Success Rate: 91.3% (Target: >85%)
#    - Safety Violations: 0 (Target: 0)
# 
# 🔄 RECOMMENDATIONS:
# 1. Add more network-related training incidents
# 2. Review confidence thresholds for memory issues
# 3. Consider ensemble learning for complex patterns
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📞 Support

For issues and questions:
- Create GitHub issues for bugs and feature requests
- Check the API documentation at `/docs`
- Review logs via the CLI tool or dashboard

---

**Ready to eliminate manual on-call work? Start with the quick setup above!** 🚀
