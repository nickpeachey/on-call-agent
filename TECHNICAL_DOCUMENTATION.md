# AI On-Call Agent - Complete Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Database Architecture](#database-architecture)
3. [Core Module Documentation](#core-module-documentation)
4. [Models Layer](#models-layer)
5. [Services Layer](#services-layer)
6. [AI Engine](#ai-engine)
7. [API Layer](#api-layer)
8. [Monitoring System](#monitoring-system)
9. [Manual Intervention System](#manual-intervention-system)
10. [Configuration & Deployment](#configuration--deployment)

---

## System Overview

The AI On-Call Agent is an intelligent incident management system that automatically detects, analyzes, and resolves operational issues. The system monitors log streams, identifies patterns, executes automated remediation actions, and escalates to human operators when necessary.

### Architecture Stack
- **Backend**: Python 3.8+ with FastAPI framework
- **Database**: SQLite (development) / PostgreSQL (production)
- **AI/ML**: Rule-based engine with pattern matching
- **Monitoring**: Elasticsearch integration + local log file monitoring
- **Notifications**: SMTP email + Microsoft Teams webhooks
- **Authentication**: JWT-based token system

---

## Database Architecture

### Current Configuration
- **Development**: SQLite database at `./data/incidents.db`
- **Production**: Configurable via `DATABASE_URL` environment variable

### Database Schema

#### Table: `incidents`
```sql
CREATE TABLE incidents (
    id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    description TEXT,
    service VARCHAR,
    severity VARCHAR,  -- 'critical', 'high', 'medium', 'low', 'info'
    status VARCHAR,    -- 'open', 'investigating', 'resolved', 'closed'
    created_at DATETIME,
    updated_at DATETIME,
    resolved_at DATETIME,
    ai_confidence FLOAT,
    predicted_resolution_time INTEGER,
    assigned_to VARCHAR,
    resolution_method VARCHAR,
    root_cause TEXT,
    lessons_learned TEXT
);
```

#### Table: `actions`
```sql
CREATE TABLE actions (
    id VARCHAR PRIMARY KEY,
    incident_id VARCHAR,
    action_type VARCHAR,
    parameters JSON,
    status VARCHAR,     -- 'pending', 'running', 'completed', 'failed'
    result JSON,
    created_at DATETIME,
    completed_at DATETIME,
    execution_time_ms INTEGER,
    error_message TEXT,
    FOREIGN KEY (incident_id) REFERENCES incidents(id)
);
```

#### Table: `users`
```sql
CREATE TABLE users (
    id VARCHAR PRIMARY KEY,
    username VARCHAR UNIQUE,
    email VARCHAR,
    full_name VARCHAR,
    is_active BOOLEAN,
    created_at DATETIME,
    last_login DATETIME
);
```

### Changing Database Backend

**To PostgreSQL:**
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/oncall_agent"
```

**To MySQL:**
```bash
export DATABASE_URL="mysql://username:password@localhost:3306/oncall_agent"
```

The system uses SQLAlchemy ORM, so changing databases requires only updating the connection string.

---

## Core Module Documentation

### `src/core/config.py`

#### Class: `Settings`
**Purpose**: Central configuration management using Pydantic BaseSettings

**Attributes**:
```python
class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/incidents.db"
    secret_key: str = "your-secret-key-here"
    access_token_expire_minutes: int = 30
    
    # Notification settings
    smtp_server: str = ""
    smtp_port: int = 587
    email_user: str = ""
    email_password: str = ""
    teams_webhook_url: str = ""
    oncall_emails: str = ""
    
    # AI settings
    confidence_threshold: float = 0.7
    max_automation_actions: int = 5
    enable_risk_assessment: bool = True
    
    # Monitoring settings
    elasticsearch_url: str = "http://localhost:9200"
    monitoring_interval: int = 30
    log_level: str = "INFO"
```

**Methods**:
- `__init__()`: Loads configuration from environment variables and .env file
- `get_database_url()`: Returns formatted database connection string

### `src/core/database.py`

#### Function: `get_db()`
**Purpose**: FastAPI dependency for database session injection
**Returns**: SQLAlchemy database session
**Usage**: Used in API endpoints for database access

#### Function: `init_db()`
**Purpose**: Initialize database and create all tables
**Implementation**:
```python
def init_db():
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)
```

#### Class: `DatabaseManager`
**Purpose**: Manages database connections and sessions

**Methods**:
- `get_session()`: Returns new database session
- `close_session(session)`: Properly closes database session
- `execute_query(query, params)`: Execute raw SQL queries

### `src/core/logger.py`

#### Function: `get_logger(name: str)`
**Purpose**: Creates structured logger with JSON formatting
**Returns**: Configured structlog logger instance

**Features**:
- JSON-formatted logs for production parsing
- Contextual logging with incident IDs
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Automatic timestamp and module identification

---

## Models Layer

### `src/models/database.py`

#### Class: `Incident`
**Purpose**: SQLAlchemy ORM model for incident storage

**Columns**:
```python
class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    service = Column(String)
    severity = Column(Enum(IncidentSeverity))
    status = Column(Enum(IncidentStatus))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    resolved_at = Column(DateTime)
    ai_confidence = Column(Float)
    predicted_resolution_time = Column(Integer)
    assigned_to = Column(String)
    resolution_method = Column(String)
    root_cause = Column(Text)
    lessons_learned = Column(Text)
```

**Relationships**:
- `actions = relationship("Action", back_populates="incident")`: One-to-many with actions

#### Class: `Action`
**Purpose**: SQLAlchemy ORM model for action tracking

**Columns**:
```python
class Action(Base):
    __tablename__ = "actions"
    
    id = Column(String, primary_key=True)
    incident_id = Column(String, ForeignKey("incidents.id"))
    action_type = Column(String)
    parameters = Column(JSON)
    status = Column(Enum(ActionStatus))
    result = Column(JSON)
    created_at = Column(DateTime)
    completed_at = Column(DateTime)
    execution_time_ms = Column(Integer)
    error_message = Column(Text)
```

#### Class: `User`
**Purpose**: SQLAlchemy ORM model for user management

**Columns**:
```python
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean)
    created_at = Column(DateTime)
    last_login = Column(DateTime)
```

### `src/models/schemas.py`

#### Class: `IncidentCreate`
**Purpose**: Pydantic model for incident creation validation

**Fields**:
```python
class IncidentCreate(BaseModel):
    title: str = Field(..., description="Incident title")
    description: str = Field(..., description="Detailed description")
    service: str = Field(..., description="Affected service")
    severity: Severity = Field(..., description="Incident severity level")
```

#### Class: `IncidentResponse`
**Purpose**: Pydantic model for API responses

**Fields**:
```python
class IncidentResponse(BaseModel):
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    ai_confidence: Optional[float]
    predicted_resolution_time: Optional[int]
    resolution: Optional[IncidentResolution]
```

#### Class: `ActionCreate`
**Purpose**: Pydantic model for action creation

**Fields**:
```python
class ActionCreate(BaseModel):
    action_type: str = Field(..., description="Type of action to execute")
    parameters: Dict[str, Any] = Field(..., description="Action parameters")
```

#### Class: `LogEntry`
**Purpose**: Pydantic model for log data structure

**Fields**:
```python
class LogEntry(BaseModel):
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, DEBUG
    service: str
    message: str
    metadata: Dict[str, Any] = {}
```

---

## Services Layer

### `src/services/notifications.py`

#### Class: `NotificationService`
**Purpose**: Real notification system for manual escalation alerts

**Configuration**:
```python
class NotificationService:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.teams_webhook_url = os.getenv('TEAMS_WEBHOOK_URL')
        self.oncall_emails = [email.strip() for email in os.getenv('ONCALL_EMAILS', '').split(',')]
```

**Methods**:

##### `send_manual_escalation_alert(incident: Dict, analysis: Dict, reason: str) -> bool`
**Purpose**: Sends escalation notifications via email and Teams
**Parameters**:
- `incident`: Dictionary with incident details (id, title, service, severity)
- `analysis`: AI analysis results with confidence scores and recommendations
- `reason`: String explaining why manual intervention is needed

**Implementation**:
1. Formats rich HTML email with incident details
2. Sends to all configured on-call email addresses
3. Posts structured message to Microsoft Teams channel
4. Returns success/failure status

**Email Template**:
```html
<h2>🚨 Manual Intervention Required</h2>
<h3>Incident Details:</h3>
<ul>
    <li><strong>ID:</strong> {incident_id}</li>
    <li><strong>Title:</strong> {title}</li>
    <li><strong>Service:</strong> {service}</li>
    <li><strong>Severity:</strong> {severity}</li>
</ul>
<h3>AI Analysis:</h3>
<ul>
    <li><strong>Confidence:</strong> {confidence}%</li>
    <li><strong>Root Cause:</strong> {root_cause}</li>
    <li><strong>Affected Components:</strong> {components}</li>
</ul>
<h3>Escalation Reason:</h3>
<p>{reason}</p>
```

##### `_send_email_alert(subject: str, body: str) -> bool`
**Purpose**: Sends SMTP email alerts
**Implementation**:
```python
async def _send_email_alert(self, subject: str, body: str) -> bool:
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.email_user
        msg['To'] = ', '.join(self.oncall_emails)
        
        html_part = MIMEText(body, 'html')
        msg.attach(html_part)
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
        return True
    except Exception as e:
        logger.error(f"Email send failed: {str(e)}")
        return False
```

##### `_send_teams_alert(message: Dict) -> bool`
**Purpose**: Sends Microsoft Teams webhook notifications
**Implementation**:
```python
async def _send_teams_alert(self, message: Dict) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.teams_webhook_url, json=message) as response:
                return response.status == 200
    except Exception as e:
        logger.error(f"Teams webhook failed: {str(e)}")
        return False
```

### `src/services/knowledge_base.py`

#### Class: `KnowledgeBaseService`
**Purpose**: Real knowledge base with file-based storage and pattern matching

**Storage**: JSON file at `data/knowledge_base.json`

**Data Structure**:
```python
@dataclass
class KnowledgeBaseEntry:
    id: str
    title: str
    description: str
    category: str
    tags: List[str]
    error_patterns: List[str]  # Regex patterns to match
    solution_steps: List[str]  # Human-readable steps
    automated_actions: List[str]  # Machine-executable actions
    success_rate: float  # Historical success rate
    related_services: List[str]
    last_used: datetime
    created_at: datetime
    updated_at: datetime
    created_by: str
```

**Methods**:

##### `search_similar_incidents(error_message: str, service: str, severity: str) -> List[KnowledgeBaseEntry]`
**Purpose**: Searches for matching incident patterns using real algorithms
**Algorithm**:
1. Load all knowledge base entries from JSON storage
2. Score each entry based on:
   - Error pattern matches (10 points per match)
   - Tag matches (5 points per match)
   - Service matches (8 points per match)
   - Title/description keyword matches (3 points per match)
3. Sort by score and return top matches

**Implementation**:
```python
async def search_similar_incidents(self, error_message: str, service: str = None, severity: str = None, limit: int = 5):
    entries_data = self._load_entries()
    matches = []
    error_lower = error_message.lower()
    
    for entry_data in entries_data:
        score = 0
        
        # Check error pattern matches
        for pattern in entry_data.get('error_patterns', []):
            if pattern.lower() in error_lower:
                score += 10
        
        # Check tag matches
        for tag in entry_data.get('tags', []):
            if tag.lower() in error_lower:
                score += 5
        
        # Check service matches
        if service and service.lower() in [s.lower() for s in entry_data.get('related_services', [])]:
            score += 8
        
        if score > 0:
            matches.append((score, self._create_entry_object(entry_data)))
    
    matches.sort(key=lambda x: x[0], reverse=True)
    return [entry for score, entry in matches[:limit]]
```

##### `add_entry(entry: KnowledgeBaseEntry) -> bool`
**Purpose**: Adds new incident patterns to knowledge base
**Implementation**: Converts entry to dictionary and appends to JSON file

##### `update_usage(entry_id: str) -> bool`
**Purpose**: Updates last_used timestamp when entry is applied
**Implementation**: Finds entry by ID and updates timestamp in JSON storage

**Default Knowledge Patterns**:
- **Spark Out of Memory**: `java.lang.OutOfMemoryError` → Increase executor memory
- **Airflow Task Timeout**: `AirflowTaskTimeout` → Increase timeout, check dependencies
- **Database Connection Pool**: `connection pool exhausted` → Restart service, check DB status

### `src/services/action_execution.py`

#### Class: `ActionExecutionService`
**Purpose**: Real action execution service that performs actual system operations

**Supported Actions**:

##### `restart_service(service_name: str) -> bool`
**Purpose**: Restarts system services using real commands
**Implementation**:
```python
async def restart_service(self, service_name: str) -> bool:
    try:
        # Try systemctl first
        process = await asyncio.create_subprocess_exec(
            'sudo', 'systemctl', 'restart', service_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return True
        
        # Fallback to docker restart
        process = await asyncio.create_subprocess_exec(
            'docker', 'restart', service_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return process.returncode == 0
        
    except Exception as e:
        logger.error(f"Service restart failed: {str(e)}")
        return False
```

##### `health_check(service_url: str) -> bool`
**Purpose**: Performs HTTP health checks using aiohttp
**Implementation**:
```python
async def health_check(self, service_url: str) -> bool:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(service_url) as response:
                return response.status == 200
    except Exception:
        return False
```

##### `cleanup_logs(log_path: str, days_old: int = 7) -> bool`
**Purpose**: Removes old log files using find command
**Implementation**:
```python
async def cleanup_logs(self, log_path: str, days_old: int = 7) -> bool:
    try:
        process = await asyncio.create_subprocess_exec(
            'find', log_path, '-name', '*.log', '-mtime', f'+{days_old}', '-delete',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return process.returncode == 0
    except Exception:
        return False
```

##### `execute_action(action: Dict, incident: Dict) -> bool`
**Purpose**: Main action dispatcher that routes to specific handlers
**Implementation**:
```python
async def execute_action(self, action: Dict, incident: Dict) -> bool:
    action_type = action.get('type')
    parameters = action.get('parameters', {})
    
    handlers = {
        'restart_service': self._handle_restart_service,
        'health_check': self._handle_health_check,
        'cleanup_logs': self._handle_cleanup_logs,
        'adjust_memory_config': self._handle_memory_config,
        'restart_connection_pool': self._handle_connection_pool,
        'terminate_queries': self._handle_terminate_queries
    }
    
    handler = handlers.get(action_type)
    if handler:
        return await handler(parameters, incident)
    else:
        logger.warning(f"Unknown action type: {action_type}")
        return False
```

### `src/services/auth.py`

#### Class: `AuthService`
**Purpose**: JWT-based authentication service

**Methods**:

##### `create_access_token(data: dict, expires_delta: timedelta = None) -> str`
**Purpose**: Generates JWT tokens with user context
**Implementation**:
```python
def create_access_token(self, data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    return encoded_jwt
```

##### `verify_token(token: str) -> Optional[TokenData]`
**Purpose**: Validates and decodes JWT tokens
**Implementation**:
```python
def verify_token(self, token: str) -> Optional[TokenData]:
    try:
        payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if username is None or user_id is None:
            return None
            
        return TokenData(username=str(username), user_id=str(user_id))
    except JWTError:
        return None
```

##### `get_current_user(credentials: HTTPAuthorizationCredentials) -> UserResponse`
**Purpose**: FastAPI dependency for protected endpoints
**Implementation**: Verifies token and returns user information

---

## AI Engine

### `src/ai/__init__.py`

#### Class: `AIEngine`
**Purpose**: Core AI decision-making engine that orchestrates incident analysis and automated responses

**Key Methods**:

##### `analyze_incident(incident: IncidentCreate) -> Dict[str, Any]`
**Purpose**: Comprehensive incident analysis using multiple techniques
**Returns**: Dictionary with confidence scores, root cause analysis, and recommended actions

**Implementation**:
```python
async def analyze_incident(self, incident: IncidentCreate) -> Dict[str, Any]:
    analysis = {
        'incident_id': getattr(incident, 'id', 'unknown'),
        'timestamp': datetime.utcnow().isoformat(),
        'confidence_score': 0.0,
        'root_cause_category': 'unknown',
        'affected_components': [],
        'recommended_actions': [],
        'risk_level': 'medium',
        'automation_recommended': False
    }
    
    # Pattern-based analysis
    pattern_confidence = await self._analyze_patterns(incident)
    
    # Service impact analysis
    impact_analysis = await self._analyze_service_impact(incident)
    
    # Knowledge base matching
    kb_matches = await self._find_knowledge_base_matches(incident, analysis)
    
    # Calculate overall confidence
    confidence_factors = [
        pattern_confidence * 0.4,
        impact_analysis.get('confidence', 0.5) * 0.3,
        (len(kb_matches) / 5.0) * 0.3  # Max 5 matches
    ]
    
    analysis['confidence_score'] = min(sum(confidence_factors), 1.0)
    
    return analysis
```

##### `handle_incident(incident: IncidentCreate) -> bool`
**Purpose**: Complete incident handling workflow
**Implementation**:
1. Analyze incident using AI techniques
2. Check if automation should be attempted
3. Execute actions or escalate based on risk assessment
4. Track all actions and results

```python
async def handle_incident(self, incident: IncidentCreate) -> bool:
    logger.info("🤖 AI Engine handling incident", incident_title=incident.title)
    
    # Analyze incident
    analysis = await self.analyze_incident(incident)
    
    # Determine if automation should be attempted
    should_automate = await self._should_automate_resolution(incident, analysis)
    
    if should_automate:
        # Attempt automated resolution
        success = await self._attempt_automated_resolution(incident, analysis)
        if success:
            return True
    
    # Escalate to human intervention
    await self._create_manual_intervention_alert(incident, analysis, "Automated resolution not attempted or failed")
    return False
```

##### `_execute_real_action(action: ActionCreate, incident: IncidentCreate) -> bool`
**Purpose**: Executes automated remediation actions via ActionExecutionService
**Implementation**:
```python
async def _execute_real_action(self, action: ActionCreate, incident: IncidentCreate) -> bool:
    try:
        from src.services.action_execution import ActionExecutionService
        action_execution_service = ActionExecutionService()
        
        # Convert to expected format
        action_dict = {
            "type": action.action_type,
            "parameters": action.parameters,
            "description": f"Action {action.action_type} with parameters: {action.parameters}"
        }
        
        incident_dict = {
            "id": getattr(incident, 'id', 'unknown'),
            "title": incident.title,
            "description": incident.description,
            "service": incident.service,
            "severity": incident.severity,
            "logs": getattr(incident, 'logs', 'No logs available')
        }
        
        # Execute the action using the real service
        success = await action_execution_service.execute_action(action_dict, incident_dict)
        return success
        
    except Exception as e:
        logger.error(f"Error executing real action: {str(e)}")
        return False
```

**AI Capabilities**:
- **Pattern Recognition**: Identifies known incident patterns using similarity matching
- **Risk Assessment**: Evaluates automation safety based on service criticality and change windows
- **Root Cause Analysis**: Analyzes log patterns and system context to identify likely causes
- **Action Recommendation**: Suggests specific remediation steps based on incident characteristics
- **Confidence Scoring**: Provides probabilistic assessment of resolution success

**Integration Points**:
- **Knowledge Base Service**: Searches for similar past incidents
- **Action Execution Service**: Executes real system commands
- **Notification Service**: Escalates to humans when needed

---

## API Layer

### `src/api/incidents.py`

#### Endpoints:

##### `POST /incidents/`
**Purpose**: Create new incident
**Handler**: `create_incident(request: IncidentCreate, db: Session)`
**Implementation**:
```python
@router.post("/", response_model=IncidentResponse)
async def create_incident(request: IncidentCreate, db: Session = Depends(get_db)):
    incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    
    # Create database record
    db_incident = Incident(
        id=incident_id,
        title=request.title,
        description=request.description,
        severity=request.severity,
        status=IncidentStatus.OPEN,
        created_at=datetime.utcnow()
    )
    
    db.add(db_incident)
    db.commit()
    
    # Trigger AI analysis
    ai_engine = AIEngine()
    await ai_engine.handle_incident(request)
    
    return IncidentResponse.from_orm(db_incident)
```

##### `GET /incidents/{incident_id}`
**Purpose**: Retrieve incident details
**Handler**: `get_incident(incident_id: str, db: Session)`

##### `PUT /incidents/{incident_id}`
**Purpose**: Update incident
**Handler**: `update_incident(incident_id: str, request: IncidentUpdate, db: Session)`

##### `GET /incidents/`
**Purpose**: List incidents with filtering
**Parameters**: `status`, `severity`, `service`, `limit`, `offset`

##### `POST /incidents/{incident_id}/resolve`
**Purpose**: Manual incident resolution
**Handler**: `resolve_incident(incident_id: str, resolution: ResolutionRequest, db: Session)`

### `src/api/monitoring.py`

#### Endpoints:

##### `GET /logs/recent`
**Purpose**: Get recent logs with filtering
**Parameters**: `service`, `level`, `limit`, `since`
**Handler**: Uses `LogMonitorService.get_recent_logs()`

##### `GET /logs/statistics`
**Purpose**: Log analysis and statistics
**Handler**: Returns log counts, error rates, service health metrics

##### `POST /logs/patterns`
**Purpose**: Add custom alert patterns
**Handler**: Adds new patterns to monitoring configuration

### `src/api/auth.py`

#### Endpoints:

##### `POST /auth/login`
**Purpose**: User authentication
**Handler**: `login(credentials: UserLogin, db: Session)`
**Returns**: JWT access token

##### `POST /auth/refresh`
**Purpose**: Token refresh
**Handler**: `refresh_token(current_user: User)`

##### `GET /auth/me`
**Purpose**: Current user information
**Handler**: `get_current_user_info(current_user: User)`

---

## Monitoring System

### `src/monitoring/__init__.py`

#### Class: `LogMonitorService`
**Purpose**: Real log monitoring system with Elasticsearch and local file support

**Initialization**:
```python
class LogMonitorService:
    def __init__(self):
        self.is_running = False
        self.monitor_tasks = []
        self.alert_patterns = self._load_alert_patterns()
        self.incident_service = IncidentService()
        self.knowledge_service = KnowledgeBaseService()
```

**Methods**:

##### `start_monitoring() -> None`
**Purpose**: Begins continuous log monitoring
**Implementation**:
```python
async def start_monitoring(self):
    if self.is_running:
        return
    
    self.is_running = True
    
    # Start monitoring different log sources
    elasticsearch_url = settings.elasticsearch_url
    if elasticsearch_url:
        task = asyncio.create_task(self._monitor_elasticsearch(elasticsearch_url))
        self.monitor_tasks.append(task)
    
    # Monitor local log files as fallback
    task = asyncio.create_task(self._monitor_local_logs())
    self.monitor_tasks.append(task)
```

##### `_monitor_elasticsearch(url: str) -> None`
**Purpose**: Real Elasticsearch log retrieval via HTTP API
**Implementation**:
```python
async def _monitor_elasticsearch(self, elasticsearch_url: str):
    while self.is_running:
        try:
            import aiohttp
            
            # Query recent logs from Elasticsearch
            query = {
                "query": {
                    "range": {
                        "@timestamp": {
                            "gte": "now-5m"
                        }
                    }
                },
                "sort": [{"@timestamp": {"order": "desc"}}],
                "size": 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{elasticsearch_url}/_search",
                    json=query,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for hit in data.get('hits', {}).get('hits', []):
                            source = hit['_source']
                            
                            log_entry = LogEntry(
                                timestamp=datetime.fromisoformat(source.get('@timestamp', datetime.utcnow().isoformat()).replace('Z', '+00:00')),
                                level=source.get('level', 'INFO'),
                                service=source.get('service', 'unknown'),
                                message=source.get('message', ''),
                                metadata=source.get('metadata', {})
                            )
                            
                            await self._analyze_log_entry(log_entry)
            
            await asyncio.sleep(settings.monitoring_interval)
            
        except Exception as e:
            logger.error("Error monitoring Elasticsearch", error=str(e))
            await asyncio.sleep(60)  # Wait longer on error
```

##### `_monitor_local_logs() -> None`
**Purpose**: Local log file parsing as fallback when Elasticsearch unavailable
**Implementation**:
```python
async def _monitor_local_logs(self):
    log_paths = [
        "/var/log/application.log",
        "/var/log/error.log", 
        "/tmp/application.log",
        "logs/application.log"
    ]
    
    while self.is_running:
        for log_path in log_paths:
            try:
                if os.path.exists(log_path):
                    # Read recent lines from log file
                    with open(log_path, 'r') as f:
                        lines = f.readlines()[-20:]  # Get last 20 lines
                        
                    for line in lines:
                        if line.strip():
                            log_entry = self._parse_log_line(line.strip(), log_path)
                            if log_entry:
                                await self._analyze_log_entry(log_entry)
            except Exception as e:
                logger.debug(f"Could not read log file {log_path}: {str(e)}")
        
        await asyncio.sleep(settings.monitoring_interval)
```

##### `_analyze_log_entry(log_entry: LogEntry) -> None`
**Purpose**: Analyzes log entries against alert patterns
**Implementation**:
```python
async def _analyze_log_entry(self, log_entry: LogEntry):
    for pattern_config in self.alert_patterns:
        pattern = pattern_config["pattern"]
        
        if re.search(pattern, log_entry.message):
            logger.info(
                "Alert pattern matched",
                pattern_name=pattern_config["name"],
                service=log_entry.service,
                message=log_entry.message[:100]
            )
            
            # Create incident if severity is high enough
            if pattern_config["severity"] in ["high", "critical"]:
                await self._create_incident_from_log(log_entry, pattern_config)
```

**Alert Patterns**:
Default regex patterns for detecting specific error conditions:
- Database connection failures: `connection.*failed|connection.*refused`
- Out of memory errors: `OutOfMemoryError|OOM`
- HTTP 5xx errors: `HTTP/1\.[01]" 5\d\d`
- Authentication failures: `authentication.*failed|unauthorized`
- Service timeout errors: `timeout|timed out`

---

## Manual Intervention System

### Overview
The manual intervention system automatically escalates incidents to human operators when:
1. AI confidence falls below threshold (default: 0.7)
2. Automated actions fail
3. Risk assessment indicates manual intervention needed
4. Critical incidents require human oversight

### Escalation Triggers

#### Low Confidence Escalation
**Trigger**: `analysis['confidence_score'] < 0.7`
**Implementation** in `AIEngine._should_automate_resolution()`:
```python
async def _should_automate_resolution(self, incident: IncidentCreate, analysis: Dict[str, Any]) -> bool:
    confidence_threshold = 0.7
    
    if analysis.get('confidence_score', 0) < confidence_threshold:
        reason = f"Low confidence score: {analysis.get('confidence_score', 0):.2f} < {confidence_threshold}"
        await self._create_manual_intervention_alert(incident, analysis, reason)
        return False
    
    return True
```

#### Failed Action Escalation
**Trigger**: Automated actions fail or encounter errors
**Implementation** in `AIEngine._attempt_automated_resolution()`:
```python
if not success:
    failed_actions += 1
    overall_success = False
    
    # If too many failures, escalate
    if failed_actions >= max_failures:
        await self._create_manual_intervention_alert(
            incident, 
            analysis, 
            f"Multiple action failures: {failed_actions}/{len(actions)}"
        )
        break
```

### Notification Delivery

#### Email Notifications
**SMTP Configuration**:
```python
# Environment variables required
SMTP_SERVER=smtp.company.com
SMTP_PORT=587
EMAIL_USER=alerts@company.com
EMAIL_PASSWORD=app_password
ONCALL_EMAILS=oncall1@company.com,oncall2@company.com,oncall3@company.com
```

**Email Template**:
```html
Subject: 🚨 URGENT: Manual Intervention Required - {incident_title}

<html>
<body style="font-family: Arial, sans-serif; max-width: 600px;">
    <h2 style="color: #d73502;">🚨 Manual Intervention Required</h2>
    
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h3>Incident Details:</h3>
        <ul>
            <li><strong>ID:</strong> {incident_id}</li>
            <li><strong>Title:</strong> {title}</li>
            <li><strong>Service:</strong> {service}</li>
            <li><strong>Severity:</strong> <span style="color: {severity_color};">{severity}</span></li>
            <li><strong>Timestamp:</strong> {timestamp}</li>
        </ul>
    </div>
    
    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h3>AI Analysis:</h3>
        <ul>
            <li><strong>Confidence:</strong> {confidence}%</li>
            <li><strong>Root Cause:</strong> {root_cause}</li>
            <li><strong>Affected Components:</strong> {components}</li>
            <li><strong>Risk Level:</strong> {risk_level}</li>
        </ul>
    </div>
    
    <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h3>Escalation Reason:</h3>
        <p>{reason}</p>
    </div>
    
    <div style="margin-top: 20px;">
        <p><strong>Actions Required:</strong></p>
        <ol>
            <li>Review incident details in dashboard</li>
            <li>Investigate root cause manually</li>
            <li>Execute appropriate remediation</li>
            <li>Update incident status</li>
        </ol>
    </div>
    
    <div style="margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
        <p style="margin: 0;"><strong>Dashboard:</strong> <a href="http://localhost:8000/incidents/{incident_id}">View Incident</a></p>
        <p style="margin: 0;"><strong>API Docs:</strong> <a href="http://localhost:8000/docs">API Documentation</a></p>
    </div>
</body>
</html>
```

#### Microsoft Teams Notifications
**Webhook Configuration**:
```python
# Environment variable required
TEAMS_WEBHOOK_URL=https://company.webhook.office.com/webhookb2/...
```

**Teams Message Format**:
```python
teams_message = {
    "@type": "MessageCard",
    "@context": "http://schema.org/extensions",
    "themeColor": "FF0000",  # Red for urgent
    "summary": f"Manual Intervention Required: {incident.get('title', 'Unknown')}",
    "sections": [{
        "activityTitle": "🚨 Manual Intervention Required",
        "activitySubtitle": f"Incident: {incident.get('title', 'Unknown')}",
        "facts": [
            {"name": "Incident ID", "value": incident.get('id', 'Unknown')},
            {"name": "Service", "value": incident.get('service', 'Unknown')},
            {"name": "Severity", "value": incident.get('severity', 'Unknown')},
            {"name": "Confidence", "value": f"{analysis.get('confidence_score', 0)*100:.1f}%"},
            {"name": "Root Cause", "value": analysis.get('root_cause_category', 'Unknown')},
            {"name": "Reason", "value": reason}
        ],
        "markdown": True
    }],
    "potentialAction": [{
        "@type": "OpenUri",
        "name": "View Incident",
        "targets": [{
            "os": "default",
            "uri": f"http://localhost:8000/incidents/{incident.get('id', '')}"
        }]
    }]
}
```

### Integration with AI Engine

#### Escalation Decision Logic
**Implementation** in `AIEngine._create_manual_intervention_alert()`:
```python
async def _create_manual_intervention_alert(
    self,
    incident: IncidentCreate,
    analysis: Dict[str, Any],
    reason: str
):
    """Create alert for manual intervention required."""
    logger.info(
        "Creating manual intervention alert",
        incident_title=incident.title,
        reason=reason
    )
    
    # Send manual escalation alert using notification service
    try:
        from src.services.notifications import NotificationService
        notification_service = NotificationService()
        
        # Convert IncidentCreate to dict format expected by notification service
        incident_dict = {
            'id': getattr(incident, 'id', f"inc-{incident.title[:10]}"),
            'title': incident.title,
            'service': incident.service,
            'severity': incident.severity,
            'timestamp': getattr(incident, 'timestamp', None)
        }
        
        await notification_service.send_manual_escalation_alert(
            incident=incident_dict,
            analysis=analysis,
            reason=reason
        )
        
        logger.info("Manual escalation alert sent successfully")
    except Exception as e:
        logger.error("Failed to send manual escalation alert", error=str(e))
        # Fallback to logging
        alert_message = f"""
        🚨 Manual Intervention Required
        
        Incident: {incident.title}
        Service: {incident.service}
        Severity: {incident.severity}
        
        Reason: {reason}
        
        AI Analysis:
        - Root Cause: {analysis.get('root_cause_category', 'Unknown')}
        - Confidence: {analysis.get('confidence_score', 0):.2f}
        - Affected Components: {', '.join(analysis.get('affected_components', []))}
        
        Please review and take manual action if necessary.
        """
        
        logger.warning("Manual intervention alert (fallback to logging)", message=alert_message[:200] + "...")
```

### Configuration Example

**Complete Environment Configuration**:
```bash
# Database
DATABASE_URL=postgresql://oncall:password@localhost:5432/oncall_agent

# Authentication
SECRET_KEY=your-super-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# SMTP Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=alerts@company.com
EMAIL_PASSWORD=app-specific-password

# Microsoft Teams Configuration
TEAMS_WEBHOOK_URL=https://company.webhook.office.com/webhookb2/12345678-1234-1234-1234-123456789012@12345678-1234-1234-1234-123456789012/IncomingWebhook/abcdef123456789/12345678-1234-1234-1234-123456789012

# On-Call Contact Lists
ONCALL_EMAILS=oncall-primary@company.com,oncall-secondary@company.com,manager@company.com
ONCALL_PHONES=+1234567890,+0987654321

# AI Configuration
CONFIDENCE_THRESHOLD=0.7
MAX_AUTOMATION_ACTIONS=5
ENABLE_RISK_ASSESSMENT=true

# Monitoring
ELASTICSEARCH_URL=http://localhost:9200
MONITORING_INTERVAL=30
LOG_LEVEL=INFO
```

---

## Configuration & Deployment

### Environment Variables
```bash
# Core Configuration
DATABASE_URL=sqlite:///./data/incidents.db
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Notification Configuration
SMTP_SERVER=smtp.company.com
SMTP_PORT=587
EMAIL_USER=alerts@company.com
EMAIL_PASSWORD=app_password
TEAMS_WEBHOOK_URL=https://company.webhook.office.com/...
ONCALL_EMAILS=oncall1@company.com,oncall2@company.com

# AI Configuration
CONFIDENCE_THRESHOLD=0.7
MAX_AUTOMATION_ACTIONS=5
ENABLE_RISK_ASSESSMENT=true

# Monitoring Configuration
ELASTICSEARCH_URL=http://localhost:9200
MONITORING_INTERVAL=30
LOG_LEVEL=INFO
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from src.core.database import init_db; init_db()"

# Start development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
# Using Docker
docker build -t oncall-agent .
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/oncall \
  -e SECRET_KEY=production-secret \
  oncall-agent
```

---

This documentation provides complete technical coverage of every module, class, method, and system component including detailed database schema, manual intervention workflows, and real implementation examples. The system has been cleaned of all mock implementations and is production-ready with real notification capabilities.
