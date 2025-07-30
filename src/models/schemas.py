"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class IncidentStatus(str, Enum):
    """Incident status enumeration."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ActionStatus(str, Enum):
    """Action execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class ActionType(str, Enum):
    """Types of automated actions."""
    RESTART_SERVICE = "restart_service"
    RESTART_AIRFLOW_DAG = "restart_airflow_dag"
    RESTART_SPARK_JOB = "restart_spark_job"
    CALL_API_ENDPOINT = "call_api_endpoint"
    SCALE_RESOURCES = "scale_resources"
    CLEAR_CACHE = "clear_cache"
    RESTART_DATABASE_CONNECTION = "restart_database_connection"


class Severity(str, Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentBase(BaseModel):
    """Base incident model."""
    title: str = Field(..., description="Brief description of the incident")
    description: str = Field(..., description="Detailed incident description")
    severity: Severity = Field(..., description="Incident severity level")
    service: str = Field(..., description="Affected service or component")
    tags: List[str] = Field(default=[], description="Incident tags")


class IncidentCreate(IncidentBase):
    """Schema for creating incidents."""
    pass


class IncidentResponse(IncidentBase):
    """Schema for incident responses."""
    id: str = Field(..., description="Unique incident identifier")
    status: IncidentStatus = Field(..., description="Current incident status")
    created_at: datetime = Field(..., description="Incident creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    assigned_to: Optional[str] = Field(None, description="Assigned engineer ID")
    resolution_notes: Optional[str] = Field(None, description="Resolution details")
    actions_taken: List[str] = Field(default=[], description="List of action IDs")
    
    class Config:
        from_attributes = True


class ActionBase(BaseModel):
    """Base action model."""
    action_type: ActionType = Field(..., description="Type of action to execute")
    parameters: Dict[str, Any] = Field(..., description="Action parameters")
    timeout_seconds: int = Field(300, description="Action timeout in seconds")


class ActionCreate(ActionBase):
    """Schema for creating actions."""
    incident_id: Optional[str] = Field(None, description="Related incident ID")


class ActionResponse(ActionBase):
    """Schema for action responses."""
    id: str = Field(..., description="Unique action identifier")
    status: ActionStatus = Field(..., description="Current action status")
    incident_id: Optional[str] = Field(None, description="Related incident ID")
    created_at: datetime = Field(..., description="Action creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    result: Optional[Dict[str, Any]] = Field(None, description="Action execution result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    executed_by: Optional[str] = Field(None, description="User or system that executed")
    is_manual: bool = Field(False, description="Whether action was manually triggered")
    
    class Config:
        from_attributes = True


class KnowledgeBaseEntry(BaseModel):
    """Knowledge base entry schema."""
    id: Optional[str] = Field(None, description="Unique entry identifier")
    title: str = Field(..., description="Entry title")
    description: str = Field(..., description="Detailed description")
    category: str = Field(..., description="Entry category")
    tags: List[str] = Field(default=[], description="Entry tags")
    error_patterns: List[str] = Field(..., description="Error patterns to match")
    solution_steps: List[str] = Field(..., description="Step-by-step solution")
    automated_actions: List[ActionBase] = Field(default=[], description="Automated actions")
    prerequisites: List[str] = Field(default=[], description="Prerequisites for solution")
    related_services: List[str] = Field(default=[], description="Related services")
    success_rate: float = Field(0.0, description="Historical success rate")
    last_used: Optional[datetime] = Field(None, description="Last time solution was used")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    created_by: Optional[str] = Field(None, description="Creator user ID")
    
    class Config:
        from_attributes = True


class SystemStatus(BaseModel):
    """System status schema."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Status check timestamp")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Individual service statuses")
    metrics: Dict[str, Any] = Field(..., description="System metrics")


class AlertCreate(BaseModel):
    """Schema for creating alerts."""
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    severity: Severity = Field(..., description="Alert severity")
    service: str = Field(..., description="Affected service")
    metadata: Dict[str, Any] = Field(default={}, description="Additional alert metadata")


class LogEntry(BaseModel):
    """Log entry schema."""
    timestamp: datetime = Field(..., description="Log timestamp")
    level: str = Field(..., description="Log level")
    service: str = Field(..., description="Source service")
    message: str = Field(..., description="Log message")
    metadata: Dict[str, Any] = Field(default={}, description="Additional log metadata")


class MonitoringMetrics(BaseModel):
    """Monitoring metrics schema."""
    service: str = Field(..., description="Service name")
    metrics: Dict[str, float] = Field(..., description="Metric values")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    
    
class UserBase(BaseModel):
    """Base user model."""
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    full_name: str = Field(..., description="Full name")
    is_active: bool = Field(True, description="Whether user is active")


class UserCreate(UserBase):
    """Schema for creating users."""
    password: str = Field(..., description="User password")


class UserResponse(UserBase):
    """Schema for user responses."""
    id: str = Field(..., description="Unique user identifier")
    created_at: datetime = Field(..., description="User creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")


class AlertPattern(BaseModel):
    """Alert pattern for log monitoring."""
    name: str = Field(..., description="Pattern name")
    pattern: str = Field(..., description="Regex pattern to match")
    severity: str = Field(..., description="Alert severity")
    action_required: bool = Field(True, description="Whether action is required")
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Authentication token schema."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class TokenData(BaseModel):
    """Token data schema."""
    username: Optional[str] = None
    user_id: Optional[str] = None
