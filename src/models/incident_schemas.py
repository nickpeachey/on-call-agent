"""Enhanced incident schemas for AI training and management."""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentCategory(str, Enum):
    """Incident categories for classification."""
    ETL_PIPELINE = "etl_pipeline"
    DATABASE = "database"
    AIRFLOW_DAG = "airflow_dag"
    SPARK_JOB = "spark_job"
    API_SERVICE = "api_service"
    INFRASTRUCTURE = "infrastructure"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    UNKNOWN = "unknown"


class IncidentStatus(str, Enum):
    """Incident status tracking."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


class ActionOutcome(str, Enum):
    """Action execution outcomes for training."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class IncidentMetrics(BaseModel):
    """Performance and context metrics for incidents."""
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")
    memory_usage: Optional[float] = Field(None, ge=0, le=100, description="Memory usage percentage")
    disk_usage: Optional[float] = Field(None, ge=0, le=100, description="Disk usage percentage")
    error_rate: Optional[float] = Field(None, ge=0, description="Error rate (errors per minute)")
    response_time: Optional[float] = Field(None, ge=0, description="Response time in milliseconds")
    throughput: Optional[float] = Field(None, ge=0, description="Requests/operations per second")
    queue_depth: Optional[int] = Field(None, ge=0, description="Number of items in queue")
    active_connections: Optional[int] = Field(None, ge=0, description="Number of active connections")


class IncidentContext(BaseModel):
    """Rich context information for AI training."""
    service_name: Optional[str] = Field(None, description="Name of affected service")
    environment: Optional[str] = Field(None, description="Environment (dev, staging, prod)")
    region: Optional[str] = Field(None, description="Geographic region")
    component: Optional[str] = Field(None, description="Specific component affected")
    version: Optional[str] = Field(None, description="Software version")
    deployment_id: Optional[str] = Field(None, description="Deployment identifier")
    user_impact: Optional[str] = Field(None, description="Description of user impact")
    business_impact: Optional[str] = Field(None, description="Business impact assessment")
    related_incidents: List[str] = Field(default_factory=list, description="Related incident IDs")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    metrics: Optional[IncidentMetrics] = Field(None, description="Performance metrics")


class LogEntry(BaseModel):
    """Individual log entry for incident context."""
    timestamp: datetime
    level: str = Field(..., description="Log level (ERROR, WARN, INFO, DEBUG)")
    message: str = Field(..., description="Log message")
    source: Optional[str] = Field(None, description="Log source (service, file, etc.)")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional log metadata")


class ActionTaken(BaseModel):
    """Record of actions taken during incident resolution."""
    action_id: str = Field(..., description="Unique action identifier")
    action_type: str = Field(..., description="Type of action taken")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    outcome: ActionOutcome = Field(ActionOutcome.SUCCESS, description="Action outcome")
    started_at: datetime = Field(..., description="When action was started")
    completed_at: Optional[datetime] = Field(None, description="When action completed")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Action result data")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="AI confidence in action")


class IncidentResolution(BaseModel):
    """Resolution details for training the AI model."""
    resolution_method: str = Field(..., description="How the incident was resolved")
    actions_taken: List[ActionTaken] = Field(default_factory=list, description="All actions taken")
    root_cause: Optional[str] = Field(None, description="Identified root cause")
    prevention_steps: List[str] = Field(default_factory=list, description="Steps to prevent recurrence")
    lessons_learned: Optional[str] = Field(None, description="Lessons learned")
    effectiveness_score: Optional[float] = Field(None, ge=0, le=1, description="Resolution effectiveness")
    time_to_resolution_minutes: Optional[int] = Field(None, ge=0, description="Time to resolve in minutes")
    manual_intervention_required: bool = Field(False, description="Whether manual intervention was needed")


class IncidentCreateRequest(BaseModel):
    """Request model for creating incidents via API."""
    title: str = Field(..., min_length=1, max_length=200, description="Incident title")
    description: str = Field(..., min_length=1, description="Detailed incident description")
    severity: IncidentSeverity = Field(..., description="Incident severity")
    category: IncidentCategory = Field(IncidentCategory.UNKNOWN, description="Incident category")
    context: Optional[IncidentContext] = Field(None, description="Incident context")
    log_entries: List[LogEntry] = Field(default_factory=list, description="Related log entries")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source: Optional[str] = Field(None, description="Source that reported the incident")
    external_id: Optional[str] = Field(None, description="External system ID")
    
    @validator('log_entries')
    def validate_log_entries(cls, v):
        if len(v) > 1000:  # Reasonable limit
            raise ValueError("Too many log entries (max 1000)")
        return v


class IncidentUpdateRequest(BaseModel):
    """Request model for updating incidents."""
    status: Optional[IncidentStatus] = Field(None, description="New incident status")
    severity: Optional[IncidentSeverity] = Field(None, description="Updated severity")
    category: Optional[IncidentCategory] = Field(None, description="Updated category")
    resolution: Optional[IncidentResolution] = Field(None, description="Resolution details")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    notes: Optional[str] = Field(None, description="Update notes")


class IncidentResponse(BaseModel):
    """Response model for incident data."""
    id: str = Field(..., description="Unique incident identifier")
    title: str = Field(..., description="Incident title")
    description: str = Field(..., description="Incident description")
    severity: IncidentSeverity = Field(..., description="Incident severity")
    category: IncidentCategory = Field(..., description="Incident category")
    status: IncidentStatus = Field(..., description="Current status")
    context: Optional[IncidentContext] = Field(None, description="Incident context")
    log_entries: List[LogEntry] = Field(default_factory=list, description="Log entries")
    resolution: Optional[IncidentResolution] = Field(None, description="Resolution details")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Timestamps
    created_at: datetime = Field(..., description="When incident was created")
    updated_at: datetime = Field(..., description="When incident was last updated")
    resolved_at: Optional[datetime] = Field(None, description="When incident was resolved")
    closed_at: Optional[datetime] = Field(None, description="When incident was closed")
    
    # Source information
    source: Optional[str] = Field(None, description="Source that reported the incident")
    external_id: Optional[str] = Field(None, description="External system ID")
    assigned_to: Optional[str] = Field(None, description="Person/team assigned")
    
    # AI-related fields
    ai_confidence: Optional[float] = Field(None, ge=0, le=1, description="AI confidence in classification")
    predicted_resolution_time: Optional[int] = Field(None, description="Predicted resolution time in minutes")
    similar_incidents: List[str] = Field(default_factory=list, description="Similar incident IDs")


class TrainingDataPoint(BaseModel):
    """Individual training data point for AI model."""
    incident_id: str = Field(..., description="Source incident ID")
    input_features: Dict[str, Any] = Field(..., description="Input features for training")
    target_actions: List[str] = Field(..., description="Successful actions taken")
    outcome_score: float = Field(..., ge=0, le=1, description="Success score (0-1)")
    resolution_time_minutes: int = Field(..., ge=0, description="Time to resolution")
    created_at: datetime = Field(..., description="When training data was created")


class TrainingDataset(BaseModel):
    """Complete training dataset for AI model."""
    version: str = Field(..., description="Dataset version")
    created_at: datetime = Field(..., description="When dataset was created")
    total_incidents: int = Field(..., ge=0, description="Total number of incidents")
    successful_resolutions: int = Field(..., ge=0, description="Number of successful resolutions")
    training_points: List[TrainingDataPoint] = Field(..., description="Training data points")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")


class AIModelMetrics(BaseModel):
    """AI model performance metrics."""
    model_version: str = Field(..., description="Model version")
    accuracy: float = Field(..., ge=0, le=1, description="Overall accuracy")
    precision: float = Field(..., ge=0, le=1, description="Precision score")
    recall: float = Field(..., ge=0, le=1, description="Recall score")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score")
    training_data_size: int = Field(..., ge=0, description="Size of training dataset")
    last_trained: datetime = Field(..., description="When model was last trained")
    performance_by_category: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Performance metrics by incident category"
    )
