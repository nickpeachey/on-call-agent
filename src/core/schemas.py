"""
Core schemas for the on-call agent system.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class KnowledgeBaseEntryCreate(BaseModel):
    """Schema for creating a new knowledge base entry."""
    
    title: str = Field(..., description="Title of the knowledge base entry")
    description: str = Field(..., description="Detailed description of the issue or solution")
    category: str = Field(..., description="Category or type of the entry")
    tags: List[str] = Field(default=[], description="Tags for categorization and search")
    solution: str = Field(..., description="Step-by-step solution or remediation")
    severity: str = Field(default="medium", description="Severity level: low, medium, high, critical")
    confidence: float = Field(default=0.8, description="Confidence score for the solution")
    conditions: Dict[str, Any] = Field(default={}, description="Conditions when this solution applies")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "High Memory Usage in Spark Job",
                "description": "Spark job consuming excessive memory causing OOM errors",
                "category": "spark",
                "tags": ["memory", "spark", "performance"],
                "solution": "1. Increase executor memory\n2. Adjust spark.sql.adaptive.coalescePartitions.enabled\n3. Check for data skew",
                "severity": "high",
                "confidence": 0.9,
                "conditions": {
                    "log_patterns": ["OutOfMemoryError", "GC overhead limit exceeded"],
                    "metrics": {"memory_usage": ">80%"}
                }
            }
        }


class KnowledgeBaseEntry(KnowledgeBaseEntryCreate):
    """Schema for a complete knowledge base entry with metadata."""
    
    id: str = Field(..., description="Unique identifier for the entry")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    usage_count: int = Field(default=0, description="Number of times this solution was used")
    success_rate: float = Field(default=1.0, description="Success rate of this solution")
    last_used: Optional[datetime] = Field(default=None, description="Last time this solution was used")


class KnowledgeBaseUpdate(BaseModel):
    """Schema for updating a knowledge base entry."""
    
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    solution: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    conditions: Optional[Dict[str, Any]] = None


class KnowledgeBaseQuery(BaseModel):
    """Schema for querying knowledge base entries."""
    
    query: Optional[str] = Field(default=None, description="Search query")
    category: Optional[str] = Field(default=None, description="Filter by category")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    severity: Optional[str] = Field(default=None, description="Filter by severity")
    min_confidence: Optional[float] = Field(default=None, description="Minimum confidence score")
    limit: int = Field(default=10, description="Maximum number of results")
    offset: int = Field(default=0, description="Offset for pagination")


class KnowledgeBaseResponse(BaseModel):
    """Schema for knowledge base query response."""
    
    entries: List[KnowledgeBaseEntry]
    total: int
    limit: int
    offset: int


class AlertData(BaseModel):
    """Schema for alert data from monitoring systems."""
    
    id: str
    source: str
    severity: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default={})


class LogData(BaseModel):
    """Schema for log data from various systems."""
    
    timestamp: datetime
    level: str
    message: str
    source: str
    service: Optional[str] = None
    metadata: Dict[str, Any] = Field(default={})


class ActionResult(BaseModel):
    """Schema for automation action results."""
    
    action_id: str
    action_type: str
    status: str  # success, failed, pending
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default={})


class HealthCheck(BaseModel):
    """Schema for health check responses."""
    
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str] = Field(default={})
    uptime: float
