"""API route definitions."""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..core import get_logger
from ..models.schemas import (
    IncidentResponse,
    ActionResponse,
    KnowledgeBaseEntry,
    SystemStatus,
    AlertCreate,
)
from ..services.incidents import IncidentService
from ..services.knowledge_base import KnowledgeBaseService
from ..services.actions import ActionService


logger = get_logger(__name__)
security = HTTPBearer()


# Optional auth dependency for now
async def get_optional_user():
    """Optional user dependency - returns None if no auth."""
    return None


def create_api_router() -> APIRouter:
    """Create the main API router with all endpoints."""
    router = APIRouter()
    
    # Include all route modules
    from . import logs
    router.include_router(logs.router)
    
    # Include AI training endpoints
    from . import simple_ai_training
    router.include_router(simple_ai_training.router)
    
    # Include resolution monitoring
    from .resolution_monitor import create_resolution_monitor_router
    router.include_router(create_resolution_monitor_router(), prefix="/resolutions", tags=["Resolution Monitoring"])
    
    # Include enhanced incidents (new AI-powered system)
    from .enhanced_incidents import create_enhanced_incidents_router, create_legacy_incidents_router
    router.include_router(create_enhanced_incidents_router(), prefix="/enhanced-incidents", tags=["Enhanced Incidents"])
    router.include_router(create_legacy_incidents_router(), prefix="/incidents", tags=["Legacy Incidents"])
    
    # Include testing endpoints
    from .testing_clean import router as testing_router
    router.include_router(testing_router, tags=["Testing & Demo"])
    
    # Include other routers
    router.include_router(create_knowledge_base_router(), prefix="/knowledge", tags=["Knowledge Base"])
    router.include_router(create_actions_router(), prefix="/actions", tags=["Actions"])
    router.include_router(create_monitoring_router(), prefix="/monitoring", tags=["Monitoring"])
    
    # Root endpoint
    @router.get("/")
    async def root():
        return {
            "message": "AI On-Call Agent API",
            "version": "0.1.0",
            "status": "operational"
        }
    
    return router


def create_incidents_router() -> APIRouter:
    """Create incidents management router."""
    router = APIRouter()
    
    @router.get("/", response_model=List[IncidentResponse])
    async def list_incidents(
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        user=Depends(get_optional_user)
    ):
        """List incidents with optional filtering."""
        service = IncidentService()
        return await service.list_incidents(skip=skip, limit=limit, status=status)
    
    @router.get("/{incident_id}", response_model=IncidentResponse)
    async def get_incident(
        incident_id: str,
        user=Depends(get_optional_user)
    ):
        """Get specific incident details."""
        service = IncidentService()
        incident = await service.get_incident(incident_id)
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Incident not found"
            )
        return incident
    
    @router.post("/{incident_id}/resolve")
    async def resolve_incident(
        incident_id: str,
        resolution_notes: str,
        user=Depends(get_optional_user)
    ):
        """Mark incident as resolved."""
        service = IncidentService()
        success = await service.resolve_incident(incident_id, resolution_notes, "system")
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Incident not found"
            )
        return {"message": "Incident resolved successfully"}
    
    return router


def create_knowledge_base_router() -> APIRouter:
    """Create knowledge base management router."""
    router = APIRouter()
    
    @router.get("/", response_model=List[KnowledgeBaseEntry])
    async def list_knowledge_entries(
        skip: int = 0,
        limit: int = 100,
        category: Optional[str] = None,
        search: Optional[str] = None,
        user=Depends(get_optional_user)
    ):
        """List knowledge base entries."""
        service = KnowledgeBaseService()
        return await service.search_entries(
            skip=skip,
            limit=limit,
            category=category,
            search=search
        )
    
    @router.post("/", response_model=KnowledgeBaseEntry)
    async def create_knowledge_entry(
        entry: KnowledgeBaseEntry,
        user=Depends(get_optional_user)
    ):
        """Create new knowledge base entry."""
        service = KnowledgeBaseService()
        return await service.create_entry(entry, "system")
    
    @router.put("/{entry_id}", response_model=KnowledgeBaseEntry)
    async def update_knowledge_entry(
        entry_id: str,
        entry: KnowledgeBaseEntry,
        user=Depends(get_optional_user)
    ):
        """Update knowledge base entry."""
        service = KnowledgeBaseService()
        updated_entry = await service.update_entry(entry_id, entry, "system")
        if not updated_entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge entry not found"
            )
        return updated_entry
    
    @router.delete("/{entry_id}")
    async def delete_knowledge_entry(
        entry_id: str,
        user=Depends(get_optional_user)
    ):
        """Delete knowledge base entry."""
        service = KnowledgeBaseService()
        success = await service.delete_entry(entry_id, "system")
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge entry not found"
            )
        return {"message": "Knowledge entry deleted successfully"}
    
    return router


def create_actions_router() -> APIRouter:
    """Create actions management router."""
    router = APIRouter()
    
    @router.get("/", response_model=List[ActionResponse])
    async def list_actions(
        request: Request,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        user=Depends(get_optional_user)
    ):
        """List executed actions."""
        action_engine = request.app.state.action_engine
        return await action_engine.action_service.list_actions(skip=skip, limit=limit, status=status)
    
    @router.post("/execute")
    async def execute_manual_action(
        request: Request,
        action_type: str,
        parameters: Dict[str, Any],
        incident_id: Optional[str] = None,
        user=Depends(get_optional_user)
    ):
        """Execute a manual action."""
        action_engine = request.app.state.action_engine
        action_id = await action_engine.action_service.execute_action(
            action_type=action_type,
            parameters=parameters,
            incident_id=incident_id,
            user_id="system",
            is_manual=True
        )
        return {"action_id": action_id, "message": "Action queued for execution"}
    
    @router.get("/{action_id}", response_model=ActionResponse)
    async def get_action(
        request: Request,
        action_id: str,
        user=Depends(get_optional_user)
    ):
        """Get action execution details."""
        action_engine = request.app.state.action_engine
        action = await action_engine.action_service.get_action(action_id)
        if not action:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Action not found"
            )
        return action
    
    return router


def create_monitoring_router() -> APIRouter:
    """Create monitoring and system status router."""
    router = APIRouter()
    
    @router.get("/status", response_model=SystemStatus)
    async def get_system_status(user=Depends(get_optional_user)):
        """Get overall system status."""
        # This would integrate with your monitoring service
        return SystemStatus(
            status="healthy",
            timestamp=datetime.utcnow(),
            services={
                "log_monitor": {"status": "running", "last_check": datetime.utcnow()},
                "ai_engine": {"status": "running", "last_check": datetime.utcnow()},
                "action_engine": {"status": "running", "last_check": datetime.utcnow()},
            },
            metrics={
                "incidents_last_24h": 5,
                "actions_executed": 12,
                "success_rate": 0.95
            }
        )
    
    @router.get("/logs")
    async def get_recent_logs(
        limit: int = 100,
        level: Optional[str] = None,
        service: Optional[str] = None,
        user=Depends(get_optional_user)
    ):
        """Get recent system logs."""
        # This would integrate with your logging system
        return {
            "logs": [],
            "total": 0,
            "filters": {
                "level": level,
                "service": service,
                "limit": limit
            }
        }
    
    @router.post("/alerts")
    async def create_alert(
        alert: AlertCreate,
        user=Depends(get_optional_user)
    ):
        """Create a manual alert."""
        # This would integrate with your alerting system
        return {"message": "Alert created successfully", "alert_id": "alert_123"}
    
    return router
