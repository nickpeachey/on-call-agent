"""API endpoints for incident management with AI integration."""

from fastapi import APIRouter, HTTPException, Depends, Request, status
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..core import get_logger
from ..models.incident_schemas import (
    IncidentCreateRequest, IncidentUpdateRequest, IncidentResponse,
    IncidentStatus, IncidentSeverity, IncidentCategory, TrainingDataPoint
)
from ..services.enhanced_incident_service import EnhancedIncidentService


logger = get_logger(__name__)


def create_enhanced_incidents_router() -> APIRouter:
    """Create enhanced incidents management router."""
    router = APIRouter()
    
    # Get incident service from app state
    def get_incident_service(request: Request) -> EnhancedIncidentService:
        """Get incident service from app state."""
        if hasattr(request.app.state, 'enhanced_incident_service'):
            return request.app.state.enhanced_incident_service
        else:
            # Fallback: create new instance
            return EnhancedIncidentService()
    
    @router.post("/", response_model=Dict[str, Any])
    async def create_incident_from_airflow(
        request_data: IncidentCreateRequest,
        auto_resolve: bool = True,
        service: EnhancedIncidentService = Depends(get_incident_service)
    ):
        """
        Create incident from Airflow DAG failure with AI analysis and async resolution.
        
        This endpoint is designed to be called by Airflow when a DAG fails.
        It will:
        1. Create the incident in the database
        2. Trigger async AI analysis and action execution in background
        3. Return immediately with incident ID and execution status
        """
        try:
            logger.info("🚨 NEW INCIDENT FROM AIRFLOW", 
                       title=request_data.title,
                       severity=request_data.severity,
                       auto_resolve=auto_resolve)
            
            # Create incident and trigger async resolution
            result = await service.create_incident_with_async_resolution(
                request_data, auto_resolve=auto_resolve
            )
            
            return result
            
        except Exception as e:
            logger.error("❌ FAILED TO CREATE INCIDENT", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create incident: {str(e)}"
            )
    
    @router.get("/{incident_id}", response_model=IncidentResponse)
    async def get_incident(
        incident_id: str,
        service: EnhancedIncidentService = Depends(get_incident_service)
    ):
        """Get incident by ID."""
        incident = await service.get_incident(incident_id)
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Incident not found"
            )
        return incident
    
    @router.get("/", response_model=List[IncidentResponse])
    async def list_incidents(
        skip: int = 0,
        limit: int = 100,
        status: Optional[IncidentStatus] = None,
        severity: Optional[IncidentSeverity] = None,
        category: Optional[IncidentCategory] = None,
        service: EnhancedIncidentService = Depends(get_incident_service)
    ):
        """List incidents with filtering."""
        return await service.list_incidents(
            skip=skip,
            limit=limit,
            status=status,
            severity=severity,
            category=category
        )
    
    @router.put("/{incident_id}", response_model=IncidentResponse)
    async def update_incident(
        incident_id: str,
        update_data: IncidentUpdateRequest,
        service: EnhancedIncidentService = Depends(get_incident_service)
    ):
        """Update incident details."""
        incident = await service.get_incident(incident_id)
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Incident not found"
            )
        
        # Apply updates
        if update_data.status is not None:
            incident.status = update_data.status
        if update_data.severity is not None:
            incident.severity = update_data.severity
        if update_data.category is not None:
            incident.category = update_data.category
        if update_data.resolution is not None:
            incident.resolution = update_data.resolution
        if update_data.additional_context is not None:
            incident.metadata.update(update_data.additional_context)
        
        incident.updated_at = datetime.utcnow()
        
        # Handle resolution
        if update_data.resolution and update_data.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
            return await service.update_incident_resolution(
                incident_id, update_data.resolution, update_data.status or IncidentStatus.RESOLVED
            )
        
        # Store updated incident (would use database)
        return incident
    
    @router.post("/{incident_id}/actions/{action_id}/success")
    async def mark_action_success(
        incident_id: str,
        action_id: str,
        result_data: Dict[str, Any],
        service: EnhancedIncidentService = Depends(get_incident_service)
    ):
        """Mark an action as successful."""
        await service.mark_action_success(incident_id, action_id, result_data)
        return {"message": "Action marked as successful"}
    
    @router.post("/{incident_id}/actions/{action_id}/failure")
    async def mark_action_failure(
        incident_id: str,
        action_id: str,
        error_data: Dict[str, str],
        service: EnhancedIncidentService = Depends(get_incident_service)
    ):
        """Mark an action as failed."""
        error_message = error_data.get("error_message", "Unknown error")
        await service.mark_action_failure(incident_id, action_id, error_message)
        return {"message": "Action marked as failed"}
    
    @router.get("/training/data", response_model=List[TrainingDataPoint])
    async def get_training_data(
        limit: int = 1000,
        min_effectiveness_score: float = 0.7,
        service: EnhancedIncidentService = Depends(get_incident_service)
    ):
        """Get training data for AI model retraining."""
        return await service.get_training_data(limit, min_effectiveness_score)
    
    @router.get("/ai/metrics")
    async def get_ai_metrics(
        service: EnhancedIncidentService = Depends(get_incident_service)
    ):
        """Get AI model performance metrics."""
        return await service.get_ai_metrics()
    
    @router.post("/webhook/airflow")
    async def airflow_webhook(
        webhook_data: Dict[str, Any],
        service: EnhancedIncidentService = Depends(get_incident_service)
    ):
        """
        Webhook endpoint specifically for Airflow DAG failures.
        
        Expected payload from Airflow:
        {
            "dag_id": "my_dag",
            "task_id": "failed_task",
            "execution_date": "2025-07-30T10:00:00Z",
            "state": "failed",
            "log_url": "http://airflow/log/...",
            "error_message": "Task failed due to...",
            "context": {
                "environment": "production",
                "cluster": "spark-cluster-1"
            }
        }
        """
        try:
            # Transform Airflow webhook data to incident request
            dag_id = webhook_data.get("dag_id", "unknown")
            task_id = webhook_data.get("task_id", "unknown")
            error_message = webhook_data.get("error_message", "Unknown error")
            
            # Determine severity based on context
            severity = IncidentSeverity.HIGH
            environment = webhook_data.get("context", {}).get("environment", "unknown")
            if environment == "production":
                severity = IncidentSeverity.CRITICAL
            elif environment in ["staging", "test"]:
                severity = IncidentSeverity.MEDIUM
            
            from ..models.incident_schemas import IncidentContext
            
            incident_request = IncidentCreateRequest(
                title=f"Airflow DAG Failed: {dag_id}",
                description=f"DAG '{dag_id}' task '{task_id}' failed with error: {error_message}",
                severity=severity,
                category=IncidentCategory.AIRFLOW_DAG,
                context=IncidentContext(
                    service_name=dag_id,
                    environment=environment,
                    component=task_id,
                    tags=[],
                    region=None,
                    version=None,
                    deployment_id=None,
                    user_impact=None,
                    business_impact=None,
                    metrics=None
                ),
                source="airflow",
                external_id=f"{dag_id}_{task_id}_{webhook_data.get('execution_date', '')}",
                metadata={
                    "dag_id": dag_id,
                    "task_id": task_id,
                    "execution_date": webhook_data.get("execution_date"),
                    "state": webhook_data.get("state"),
                    "log_url": webhook_data.get("log_url"),
                    "airflow_context": webhook_data.get("context", {})
                }
            )
            
            # Create incident with auto-resolution
            incident, action_ids = await service.create_incident_from_airflow(
                incident_request, auto_resolve=True
            )
            
            return {
                "status": "success",
                "incident_id": incident.id,
                "action_ids": action_ids,
                "message": f"Incident created for DAG {dag_id}",
                "auto_resolution_attempted": len(action_ids) > 0
            }
            
        except Exception as e:
            logger.error("❌ AIRFLOW WEBHOOK FAILED", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process Airflow webhook: {str(e)}"
            )
    
    return router


def create_legacy_incidents_router() -> APIRouter:
    """Create legacy incidents router for backward compatibility."""
    router = APIRouter()
    
    @router.get("/")
    async def list_legacy_incidents():
        """Legacy endpoint - redirects to enhanced version."""
        return {"message": "Please use /api/v1/enhanced-incidents/ for full functionality"}
    
    @router.post("/")
    async def create_legacy_incident():
        """Legacy endpoint - redirects to enhanced version."""
        return {"message": "Please use /api/v1/enhanced-incidents/ for incident creation"}
    
    return router
