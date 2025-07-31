# 🤖 AI On-Call Agent: The Complete Dummy's Guide

> **A comprehensive, easy-to-understand guide for using, training, and managing the AI On-Call Agent system**

---

## 📚 Table of Contents

1. [What is the AI On-Call Agent?](#what-is-the-ai-on-call-agent)
2. [How to Use the System](#how-to-use-the-system)
3. [How to Train the AI Models](#how-to-train-the-ai-models)
4. [Where Data Comes From](#where-data-comes-from)
5. [How Often Polling Works](#how-often-polling-works)
6. [Where the Logs Go](#where-the-logs-go)
7. [System Architecture Made Simple](#system-architecture-made-simple)
8. [Common Tasks & Examples](#common-tasks--examples)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Quick Reference](#quick-reference)

---

## 1. What is the AI On-Call Agent?

### Think of it as Your 24/7 IT Assistant 🤖

The AI On-Call Agent is like having a super-smart IT person who never sleeps and watches your systems 24/7. Here's what it does:

- **Monitors logs** from your applications, databases, and services
- **Detects problems** automatically using pattern matching
- **Learns from experience** to get better at fixing issues
- **Takes action** to fix problems without waking you up at 3 AM
- **Reports everything** so you know what happened

### Real-World Example
```
Your Airflow pipeline fails at 2 AM
↓
AI Agent detects the error in logs
↓
AI recognizes this as a known issue
↓
AI automatically restarts the failed task
↓
Pipeline resumes successfully
↓
You wake up to a "Fixed automatically" notification
```

---

## 2. How to Use the System

### Getting Started (Step-by-Step)

#### Step 1: Start the System
```bash
# Option 1: Use the startup script (recommended)
python start.py

# Option 2: Use the CLI tool
python cli.py start-services

# Option 3: Direct start
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

#### Step 2: Access the Dashboard
Open your web browser and go to:
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/health

#### Step 3: Check System Status
```bash
# Using the CLI
python cli.py status

# Using the web interface
curl http://localhost:8000/health
```

### Daily Operations

#### View Recent Incidents
```bash
# Command line
python cli.py incidents

# Web interface
curl http://localhost:8000/api/v1/incidents
```

#### Monitor System Activity
```bash
# Check what the AI is learning
python cli.py monitor-logs --duration 60

# View knowledge base
python cli.py knowledge
```

#### Manual Problem Simulation (for testing)
```bash
# Simulate a database timeout
python cli.py simulate-incident --title "Database timeout" --service "postgres" --severity "high"
```

---

## 3. How to Train the AI Models

### Understanding AI Training (Simple Explanation)

Think of training the AI like teaching a new employee:
1. **Show examples** of problems and how to fix them
2. **Give feedback** on whether the AI's actions worked
3. **Let it practice** on new similar problems
4. **Monitor improvement** over time

### Training Data Structure

The AI learns from examples stored in JSON format:

```json
{
  "incident_type": "airflow_task_failure",
  "description": "Airflow DAG 'data_pipeline' task 'extract_data' failed",
  "context": {
    "service": "airflow",
    "severity": "high",
    "dag_id": "data_pipeline",
    "task_id": "extract_data"
  },
  "resolution_action": {
    "action_type": "restart_airflow_task",
    "parameters": {
      "dag_id": "data_pipeline",
      "task_id": "extract_data",
      "clear_downstream": true
    }
  },
  "outcome": "success",
  "feedback": {
    "effectiveness": 0.95,
    "time_to_resolution": 120,
    "learned_patterns": ["airflow.*failed", "task.*extract_data"]
  }
}
```

### How to Add Training Examples

#### Method 1: Direct File Editing
1. Open `data/sample_training.json`
2. Add your incident example following the structure above
3. Restart the AI engine to load new data

#### Method 2: Through the API
```bash
curl -X POST "http://localhost:8000/api/v1/ai/training" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_type": "database_connection_error",
    "description": "PostgreSQL connection pool exhausted",
    "resolution_action": {
      "action_type": "restart_database_connection",
      "parameters": {"database": "postgres", "pool_size": 50}
    }
  }'
```

### Training Process Explained

#### 1. Pattern Recognition Training
- **Input**: Log messages and error patterns
- **Process**: AI learns to recognize similar patterns
- **Output**: Confidence scores for automatic detection

#### 2. Action Effectiveness Training
- **Input**: Action results (success/failure)
- **Process**: AI learns which actions work best for specific problems
- **Output**: Improved action selection and parameters

#### 3. Continuous Learning
- **Real-time feedback**: Every action result trains the model
- **Pattern evolution**: AI discovers new problem patterns
- **Confidence calibration**: AI gets better at knowing when it's right

### Monitoring Training Progress

```bash
# Check AI learning statistics
curl http://localhost:8000/api/v1/ai/learning-stats

# View confidence scores
python cli.py ai-status

# Monitor training accuracy
python cli.py start-learning-monitor --dashboard
```

---

## 4. Where Data Comes From

### Data Sources Explained

The AI On-Call Agent gets information from multiple sources:

#### 1. Log Files (File System)
```
Default paths monitored:
- /var/log/app/*.log
- /var/log/airflow/*.log  
- /var/log/spark/*.log
- Application logs in logs/ directory
```

**Configuration:**
```bash
# Set in .env file
LOG_PATHS=/var/log/app/*.log,/var/log/airflow/*.log,/var/log/spark/*.log
LOG_POLL_INTERVAL_SECONDS=5
```

#### 2. Elasticsearch (Log Aggregation)
```
Source: Centralized log storage
Default: http://localhost:9200
Indexes: Application logs, system logs, infrastructure logs
```

**Configuration:**
```bash
ELASTICSEARCH_URLS=["http://localhost:9200"]
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=changeme
```

#### 3. Splunk (Enterprise Logging)
```
Source: Enterprise log management platform  
Default: localhost:8089
Data: Structured and unstructured log data
```

**Configuration:**
```bash
SPLUNK_HOST=localhost
SPLUNK_PORT=8089
SPLUNK_USERNAME=admin
SPLUNK_PASSWORD=changeme
```

#### 4. Demo Data (For Testing)
```
Source: Built-in mock data generator
Purpose: Testing and demonstration
Data: Realistic sample log entries
```

### Data Flow Diagram

```
External Systems → Log Sources → AI Agent → Action Engine
     ↓               ↓             ↓           ↓
   Airflow        Log Files    Pattern      Docker
   Spark      →  Elasticsearch → Matching → Kubernetes  
   APIs           Splunk       Analysis     Databases
   Databases      Demo Logs    Learning     Cache Systems
```

### What Data is Collected

#### Log Entry Structure
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "ERROR",
  "service": "airflow",
  "message": "Task 'extract_data' failed with exit code 1",
  "metadata": {
    "host": "server-01",
    "pod": "airflow-worker-123",
    "trace_id": "abc-123-def"
  }
}
```

#### Incident Data
```json
{
  "id": "incident_123",
  "title": "Airflow Task Failure",
  "description": "DAG data_pipeline task extract_data failed",
  "severity": "high",
  "service": "airflow",
  "status": "resolved",
  "actions_taken": ["restart_airflow_task"],
  "resolution_time": 120
}
```

---

## 5. How Often Polling Works

### Polling Frequencies Explained

Different components check for problems at different intervals:

#### 1. Log Monitoring (Every 15-30 seconds)
```python
# Main log monitoring loop
async def _monitor_logs(self):
    while self.is_running:
        # Check all log sources
        for source in log_sources:
            await self._check_source(source)
        
        await asyncio.sleep(30)  # Wait 30 seconds
```

**Why this frequency?**
- Fast enough to catch problems quickly
- Not so fast it overwhelms the system
- Balances responsiveness with resource usage

#### 2. AI Decision Engine (Every 2 seconds)
```python
# AI decision loop
while self.is_running:
    await self._process_incident_queue()
    await asyncio.sleep(2)  # Check queue every 2 seconds
```

**Why this frequency?**
- Processes incidents as soon as they're detected
- Allows for rapid automated response
- Prevents queue backlog

#### 3. File-based Log Polling (Every 5 seconds)
```bash
# Configuration
LOG_POLL_INTERVAL_SECONDS=5
```

**Why this frequency?**
- Good balance for file-based monitoring
- Catches new log entries quickly
- Doesn't cause excessive disk I/O

#### 4. Health Checks (Every 30 seconds)
```yaml
# Docker health check
healthcheck:
  interval: 30s
  timeout: 10s
  retries: 3
```

### Customizing Polling Intervals

#### Environment Variables
```bash
# Log polling frequency
LOG_POLL_INTERVAL_SECONDS=5

# AI decision loop frequency  
AI_DECISION_LOOP_INTERVAL_SECONDS=2

# Log monitoring frequency (in code)
# Modify MONITOR_INTERVAL in monitoring service
```

#### Performance Considerations
- **Too frequent**: High CPU usage, resource consumption
- **Too infrequent**: Delayed problem detection, slower response
- **Recommended**: Use defaults unless you have specific requirements

### Monitoring Polling Performance

```bash
# Check system performance
python cli.py status --detailed

# Monitor resource usage
htop  # CPU and memory
iotop # Disk I/O

# Check logs for performance issues
tail -f logs/app.log | grep "performance\|slow\|timeout"
```

---

## 6. Where the Logs Go

### Log Storage Locations

The AI On-Call Agent creates several types of logs in different locations:

#### 1. Application Logs
```
Location: logs/app.log
Content: General application events, errors, status messages
Format: JSON structured logging

Example entry:
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO", 
  "logger": "src.main",
  "message": "✅ All services started successfully"
}
```

#### 2. AI Decision Logs
```
Location: logs/ai_decisions.log  
Content: AI decision making, pattern matching, confidence scores
Format: Structured JSON with AI-specific fields

Example entry:
{
  "timestamp": "2024-01-15T10:31:00Z",
  "incident_id": "inc_123",
  "confidence": 0.87,
  "action_recommended": "restart_airflow_task",
  "reasoning": "Pattern match: airflow task failure"
}
```

#### 3. Action Execution Logs
```
Location: logs/actions.log
Content: Automated actions taken, results, execution time
Format: Detailed action tracking

Example entry:
{
  "timestamp": "2024-01-15T10:32:00Z",
  "action_id": "action_456",
  "action_type": "restart_airflow_task",
  "status": "success",
  "execution_time": 15.3,
  "parameters": {"dag_id": "data_pipeline", "task_id": "extract_data"}
}
```

#### 4. Startup Logs
```
Location: logs/startup.log
Content: System initialization, configuration validation
Format: Human-readable startup sequence

Example entry:
2024-01-15 10:25:00 - INFO - 🚀 Starting AI On-Call Agent...
2024-01-15 10:25:01 - INFO - ✅ Configuration validated
2024-01-15 10:25:02 - INFO - 🌐 All services healthy
```

### Log Rotation and Management

#### Automatic Log Rotation
```python
# Built-in log rotation (configured in logging.py)
{
    "version": 1,
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5       # Keep 5 old files
        }
    }
}
```

#### Manual Log Management
```bash
# View recent logs
tail -f logs/app.log

# Search logs for specific events
grep "ERROR" logs/app.log

# Archive old logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/

# Clean up old logs
find logs/ -name "*.log.*" -mtime +30 -delete
```

### Log Levels and Verbosity

#### Debug Mode (Development)
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG

# Shows everything including:
# - Detailed trace information
# - Variable values
# - Step-by-step execution
```

#### Production Mode
```bash
export DEBUG=false  
export LOG_LEVEL=INFO

# Shows only important events:
# - System status changes
# - Incidents and resolutions
# - Error conditions
```

### Accessing Logs Remotely

#### API Endpoints
```bash
# Get recent system logs
curl http://localhost:8000/api/v1/monitoring/logs?limit=100

# Filter by service
curl http://localhost:8000/api/v1/monitoring/logs?service=airflow

# Filter by level
curl http://localhost:8000/api/v1/monitoring/logs?level=ERROR
```

#### Log Aggregation Tools
```bash
# If using ELK stack
# Logs are also sent to Elasticsearch at:
http://localhost:9200/logs-*

# If using Splunk
# Logs are forwarded to:
http://localhost:8089/en-US/app/search
```

---

## 7. System Architecture Made Simple

### The Big Picture

Think of the AI On-Call Agent as a restaurant with different stations:

```
🏪 Restaurant Analogy:
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   👀 Watchers   │   🧠 Kitchen    │   🤖 Workers    │   📊 Manager    │
│                 │                 │                 │                 │
│ Log Monitor     │ AI Engine       │ Action Engine   │ Web Dashboard   │
│ File Poller     │ Decision Maker  │ Task Executor   │ API Server      │
│ Alert Detector  │ Pattern Matcher │ Problem Solver  │ Health Monitor  │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Core Components Deep Dive

#### 1. Log Monitor Service (`src/monitoring/`)
**What it does:**
- Watches log files and external systems (Elasticsearch, Splunk)
- Detects alert patterns using regular expressions
- Creates incidents when problems are found

**Key Files:**
- `src/monitoring/__init__.py` - Main monitoring service
- Alert patterns defined in `_load_alert_patterns()`

**Example Alert Pattern:**
```python
{
    "name": "Airflow Task Failure",
    "pattern": r"(?i)(airflow.*failed|task.*failed)",
    "severity": "high",
    "service": "airflow",
    "tags": ["airflow", "task", "failure"]
}
```

#### 2. AI Decision Engine (`src/ai/`)
**What it does:**
- Receives incidents from the Log Monitor
- Analyzes problems using machine learning
- Decides what actions to take based on training data
- Learns from action results to improve over time

**Key Files:**
- `src/ai/__init__.py` - Main AI engine with ML models
- Uses scikit-learn RandomForest classifiers
- Confidence scoring and pattern recognition

**Decision Process:**
```python
async def process_incident(self, incident):
    # 1. Extract features from incident
    features = self._extract_features(incident)
    
    # 2. Predict best action using ML model
    action, confidence = self._predict_action(features)
    
    # 3. If confidence is high enough, take action
    if confidence > self.confidence_threshold:
        await self._execute_action(action)
```

#### 3. Action Engine (`src/actions/`)
**What it does:**
- Executes automated fixes (restart services, clear caches, etc.)
- Manages concurrent action execution
- Reports success/failure back to the AI for learning

**Supported Actions:**
- **Docker**: Container restart, service scaling
- **Kubernetes**: Pod restart, deployment scaling
- **Airflow**: DAG restart, task clearing, triggering
- **Databases**: Connection pool restart, query optimization
- **Cache**: Redis, Memcached, filesystem cache clearing

#### 4. Log Poller Service (`src/services/log_poller.py`)
**What it does:**
- Specialized file-based log monitoring
- Configurable log sources (files, demo data)
- Integration with incident service for problem escalation

**Configuration:**
```python
log_sources = [
    {
        "name": "application_logs",
        "type": "file",
        "path": "/var/log/app/*.log",
        "polling_interval": 5
    },
    {
        "name": "demo_logs", 
        "type": "demo",
        "enabled": True
    }
]
```

### Data Flow Architecture

```
📊 Data Sources → 🔍 Detection → 🧠 Analysis → ⚡ Action → 📈 Learning

┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ Log Files   │   │ Pattern      │   │ AI Decision │   │ Action       │   │ Feedback    │
│ Elasticsearch │→│ Matching     │→│ Engine      │→│ Execution    │→│ Learning    │
│ Splunk      │   │ Alert Rules  │   │ ML Models   │   │ Infrastructure│   │ Model       │
│ APIs        │   │ Thresholds   │   │ Confidence  │   │ APIs         │   │ Updates     │
└─────────────┘   └──────────────┘   └─────────────┘   └──────────────┘   └─────────────┘
```

### Service Startup Sequence

```
1. 🗄️  Database initialization
2. 🤖 AI Decision Engine startup
3. 👀 Log Monitor startup  
4. ⚡ Action Engine startup
5. 📁 Log Poller startup
6. 🌐 Web API server startup
7. ✅ Health checks and validation
```

### Configuration Management

All configuration is centralized in `src/core/config.py`:

```python
@dataclass
class Settings:
    # Application settings
    app: AppConfig = field(default_factory=AppConfig)
    
    # Database configuration  
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # AI/ML settings
    ai: AIConfig = field(default_factory=AIConfig)
    
    # Monitoring configuration
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Action engine settings
    actions: ActionConfig = field(default_factory=ActionConfig)
```

---

## 8. Common Tasks & Examples

### Task 1: Add a New Alert Pattern

**Problem:** You want to detect when your custom application fails

**Solution:**
1. Find the log pattern for your application failure
2. Add it to the alert patterns

```python
# Edit src/monitoring/__init__.py
# Add to _load_alert_patterns() method:

{
    "name": "Custom App Failure",
    "pattern": r"(?i)(my_app.*error|custom_service.*failed)",
    "severity": "high",
    "service": "my_app",
    "tags": ["custom", "application", "failure"]
}
```

**Test it:**
```bash
# Simulate the error in logs
echo "ERROR: my_app service failed with exception" >> /var/log/app/test.log

# Check if incident was created
python cli.py incidents | grep "Custom App"
```

### Task 2: Train the AI on a New Problem Type

**Problem:** You have a new type of incident that needs automation

**Solution:**
1. Create training examples
2. Add them to the training data
3. Retrain the model

```json
// Add to data/sample_training.json
{
  "incident_type": "memory_leak_detection",
  "description": "High memory usage detected on server",
  "context": {
    "service": "web_api",
    "severity": "medium",
    "memory_usage": "95%",
    "server": "prod-web-01"
  },
  "resolution_action": {
    "action_type": "restart_service",
    "parameters": {
      "service_name": "web_api",
      "graceful": true,
      "wait_time": 30
    }
  },
  "outcome": "success",
  "feedback": {
    "effectiveness": 0.88,
    "time_to_resolution": 45,
    "learned_patterns": ["memory.*95%", "high memory usage"]
  }
}
```

**Retrain the model:**
```bash
# Restart AI engine to reload training data
python cli.py restart-ai

# Or use API
curl -X POST http://localhost:8000/api/v1/ai/retrain
```

### Task 3: Create a Custom Action

**Problem:** You need to automate a specific fix for your infrastructure

**Solution:**
1. Create a new action type
2. Implement the action logic
3. Register it with the action engine

```python
# Add to src/actions/__init__.py

async def execute_custom_restart(self, parameters: Dict[str, Any]) -> ActionResult:
    """Custom restart procedure for your application."""
    try:
        service_name = parameters.get("service_name")
        
        # Your custom restart logic here
        await self._stop_service(service_name)
        await asyncio.sleep(5)  # Wait for graceful shutdown
        await self._start_service(service_name)
        
        return ActionResult(
            success=True,
            message=f"Successfully restarted {service_name}",
            execution_time=10.5
        )
    except Exception as e:
        return ActionResult(
            success=False,
            message=f"Failed to restart {service_name}: {str(e)}",
            execution_time=0
        )

# Register the action
self.action_handlers["custom_restart"] = self.execute_custom_restart
```

### Task 4: Monitor System Performance

**Problem:** You want to track how well the AI is performing

**Solution:**
Use the built-in monitoring tools

```bash
# Check overall system status
python cli.py status

# View AI learning statistics  
curl http://localhost:8000/api/v1/ai/learning-stats

# Monitor real-time activity
python cli.py monitor-logs --duration 300  # 5 minutes

# Check action success rates
python cli.py actions --status success --limit 50
```

### Task 5: Debug a Problem

**Problem:** The AI isn't detecting or fixing a specific issue

**Solution:**
Follow the debugging process

```bash
# Step 1: Check if the pattern is detected
grep "your_error_pattern" logs/app.log

# Step 2: Verify alert patterns are loaded
python cli.py test-patterns --pattern "your_error_pattern"

# Step 3: Check if incidents are created
python cli.py incidents --service your_service

# Step 4: Verify AI confidence scores
python cli.py ai-status --verbose

# Step 5: Test actions manually
python cli.py test-action --action-type restart_service --service-name your_service
```

### Task 6: Scale for Production

**Problem:** You need to handle higher log volumes and more incidents

**Solution:**
Adjust configuration for production workloads

```bash
# Increase processing capacity
export MAX_CONCURRENT_ACTIONS=10
export AI_DECISION_LOOP_INTERVAL_SECONDS=1
export LOG_POLL_INTERVAL_SECONDS=3

# Use production database
export DATABASE_URL=postgresql://user:pass@prod-db:5432/oncall

# Enable horizontal scaling
docker-compose up --scale oncall-agent=3
```

---

## 9. Troubleshooting Guide

### Problem: System Won't Start

**Symptoms:**
- Application crashes on startup
- Connection errors
- Configuration validation fails

**Diagnostic Steps:**
```bash
# Check startup logs
cat logs/startup.log

# Validate configuration
python start.py --validate-only

# Test database connection
python cli.py test-db

# Check service dependencies
docker-compose ps
```

**Common Solutions:**
1. **Missing environment variables**: Copy `.env.example` to `.env` and fill in values
2. **Database connection failed**: Ensure PostgreSQL is running and accessible
3. **Port conflicts**: Change API_PORT in .env if 8000 is in use
4. **Permission issues**: Check file permissions on logs/ directory

### Problem: AI Not Learning

**Symptoms:**
- Confidence scores stay low
- Same problems aren't being auto-resolved
- Learning statistics show no improvement

**Diagnostic Steps:**
```bash
# Check training data
python cli.py ai-status --training-data

# Verify learning loop is running
grep "learning" logs/ai_decisions.log

# Check for training errors
grep "ERROR.*train" logs/app.log
```

**Common Solutions:**
1. **Insufficient training data**: Add more examples to `data/sample_training.json`
2. **Poor quality examples**: Ensure training data has clear patterns and outcomes
3. **Confidence threshold too high**: Lower `CONFIDENCE_THRESHOLD` in .env
4. **Model not retraining**: Restart AI engine to reload training data

### Problem: Actions Failing

**Symptoms:**
- Actions show "failed" status
- Infrastructure commands not executing
- Timeouts or connection errors

**Diagnostic Steps:**
```bash
# Check action logs
tail -f logs/actions.log

# Test action manually
python cli.py test-action --action-type restart_service --service-name test

# Verify credentials and permissions
python cli.py test-connections
```

**Common Solutions:**
1. **Missing credentials**: Update service credentials in .env
2. **Network connectivity**: Verify access to Airflow, K8s, databases
3. **Permission denied**: Ensure service accounts have required permissions
4. **Service unavailable**: Check that target services are running

### Problem: High Resource Usage

**Symptoms:**
- High CPU usage
- Memory leaks
- Slow response times

**Diagnostic Steps:**
```bash
# Monitor resource usage
htop
docker stats

# Check polling frequencies
grep "polling" logs/app.log

# Analyze slow queries
grep "slow" logs/app.log
```

**Common Solutions:**
1. **Polling too frequent**: Increase interval settings in .env
2. **Too many concurrent actions**: Reduce `MAX_CONCURRENT_ACTIONS`
3. **Memory leaks**: Restart services, check for connection pool issues
4. **Database performance**: Add indexes, optimize queries

### Problem: Logs Not Being Detected

**Symptoms:**
- No incidents created despite obvious errors
- Log patterns not matching
- Polling service not running

**Diagnostic Steps:**
```bash
# Check log poller status
python cli.py logs status

# Test pattern matching
python cli.py test-patterns --file /path/to/test.log

# Verify log file permissions
ls -la /var/log/app/

# Check monitoring service
python cli.py monitor-status
```

**Common Solutions:**
1. **Wrong log paths**: Update `LOG_PATHS` in .env to correct locations
2. **Pattern syntax errors**: Test regex patterns separately
3. **File permissions**: Ensure read access to log files
4. **Service not started**: Restart log monitoring service

### Emergency Procedures

#### Stop All Automation
```bash
# Immediate stop
python cli.py emergency-stop

# Graceful shutdown
python cli.py stop-services

# Docker stop
docker-compose down
```

#### Manual Override
```bash
# Disable specific action types
python cli.py disable-action --type restart_service

# Set manual mode
export AI_ENABLED=false

# Review and approve actions
export REQUIRE_APPROVAL=true
```

#### Recovery Mode
```bash
# Reset AI model
python cli.py reset-ai --confirm

# Clear incident queue
python cli.py clear-queue

# Restore from backup
python cli.py restore --backup-file backup.sql
```

---

## 10. Quick Reference

### Essential Commands

```bash
# System Management
python start.py                    # Start with validation
python cli.py start-services       # Start all services
python cli.py stop-services        # Stop all services
python cli.py status               # System status
python cli.py restart              # Restart everything

# Monitoring
python cli.py incidents            # List incidents
python cli.py actions              # List actions
python cli.py monitor-logs         # Real-time monitoring
python cli.py health               # Health check

# AI Management
python cli.py ai-status            # AI engine status
python cli.py retrain              # Retrain models
python cli.py confidence --set 0.8 # Set confidence threshold

# Testing
python cli.py test-action          # Test specific action
python cli.py simulate-incident    # Create test incident
python cli.py test-patterns        # Test alert patterns
```

### Configuration Quick Reference

#### Environment Variables
```bash
# Core Settings
DEBUG=false
LOG_LEVEL=INFO
API_PORT=8000

# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# AI Settings
CONFIDENCE_THRESHOLD=0.75
AI_ENABLED=true

# Monitoring
LOG_PATHS=/var/log/app/*.log
LOG_POLL_INTERVAL_SECONDS=5

# Actions
MAX_CONCURRENT_ACTIONS=5
ACTION_TIMEOUT_SECONDS=300
```

#### Key File Locations
```
Configuration:     .env, src/core/config.py
Training Data:     data/sample_training.json
Logs:             logs/app.log, logs/ai_decisions.log
Alert Patterns:   src/monitoring/__init__.py
Actions:          src/actions/__init__.py
API Docs:         http://localhost:8000/docs
```

### API Endpoints

```bash
# System Status
GET  /health                       # Health check
GET  /api/v1/monitoring/status     # Detailed status

# Incidents
GET  /api/v1/incidents             # List incidents
POST /api/v1/incidents             # Create incident
GET  /api/v1/incidents/{id}        # Get incident details

# Actions  
GET  /api/v1/actions               # List actions
POST /api/v1/actions               # Execute action

# AI Engine
GET  /api/v1/ai/learning-stats     # Learning statistics
POST /api/v1/ai/retrain            # Retrain models

# Log Management
GET  /api/v1/logs/polling/status   # Polling status
POST /api/v1/logs/polling/start    # Start polling
POST /api/v1/logs/polling/stop     # Stop polling
```

### Common Log Patterns

```python
# Database Issues
r"(?i)(connection.*timeout|pool.*exhausted|database.*error)"

# Airflow Problems  
r"(?i)(airflow.*failed|dag.*error|task.*failed)"

# Spark Issues
r"(?i)(spark.*failed|executor.*lost|yarn.*killed)"

# API Errors
r"HTTP\/\d\.\d\"\s+5\d\d"

# Memory Issues
r"(?i)(out of memory|memory.*exceeded|heap.*space)"

# File System
r"(?i)(no space left|disk.*full|permission denied)"
```

### Performance Tuning

```bash
# High Volume Logs
LOG_POLL_INTERVAL_SECONDS=10
MAX_CONCURRENT_ACTIONS=10

# Low Latency Response
LOG_POLL_INTERVAL_SECONDS=1  
AI_DECISION_LOOP_INTERVAL_SECONDS=1

# Resource Conservation
LOG_POLL_INTERVAL_SECONDS=30
MAX_CONCURRENT_ACTIONS=3
```

---

## 🎯 Summary

The AI On-Call Agent is your intelligent infrastructure assistant that:

1. **Monitors** your systems 24/7 using configurable log sources
2. **Learns** from examples to recognize and solve problems automatically  
3. **Acts** quickly to fix issues using real infrastructure APIs
4. **Reports** everything for transparency and continuous improvement

Key points to remember:
- **Training data** is crucial for AI performance - keep adding examples
- **Log patterns** determine what problems get detected - refine them regularly
- **Polling frequencies** balance responsiveness with resource usage
- **Action permissions** need to be configured for your infrastructure
- **Monitoring and logging** help you understand and improve the system

Start with the defaults, monitor performance, and gradually customize to fit your specific infrastructure and requirements.

---

**Need help?** Check the logs, use the CLI tools, and refer to the API documentation at http://localhost:8000/docs

**Made with ❤️ for keeping your systems running smoothly**
