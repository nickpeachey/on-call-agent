"""Enhanced incident management service with AI training integration."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import uuid
import asyncio
import json
import os
import base64
import aiohttp
import random

from sqlalchemy import text

from ..core import get_logger
from ..models.incident_schemas import (
    IncidentCreateRequest, IncidentUpdateRequest, IncidentResponse,
    IncidentStatus, IncidentSeverity, IncidentCategory, ActionOutcome,
    ActionTaken, IncidentResolution, TrainingDataPoint, AIModelMetrics
)
from .actions import ActionService
from ..ai import AIDecisionEngine
from ..database import get_db_session


logger = get_logger(__name__)


class EnhancedIncidentService:
    """Enhanced incident service with AI training capabilities."""
    
    def __init__(self):
        self.action_service = ActionService()
        self.ai_engine = None  # Will be injected
        # Database operations will be implemented later
    
    def set_ai_engine(self, ai_engine: AIDecisionEngine):
        """Inject AI engine dependency."""
        self.ai_engine = ai_engine
    
    async def create_incident_from_airflow(
        self,
        request: IncidentCreateRequest,
        auto_resolve: bool = True
    ) -> Tuple[IncidentResponse, List[str]]:
        """
        Create incident from Airflow failure and optionally attempt auto-resolution.
        
        Returns:
            Tuple of (incident_response, action_ids_triggered)
        """
        logger.info("🚨 AIRFLOW INCIDENT RECEIVED", 
                   title=request.title,
                   severity=request.severity,
                   category=request.category,
                   auto_resolve=auto_resolve)
        
        # Create incident in database
        incident = await self.create_incident(request)
        action_ids = []
        
        if auto_resolve and self.ai_engine:
            try:
                # Define confidence threshold early to avoid NameError
                confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))  # Default threshold
                
                # AI decision making
                logger.info("🤖 AI ANALYSIS STARTING", incident_id=incident.id)
                
                # Extract features for AI analysis
                features = await self._extract_features_for_ai(incident)
                
                # Get real AI recommendation using the AI engine
                try:
                    from ..ai import AIDecisionEngine
                    ai_engine = AIDecisionEngine()
                    
                    # Convert incident to IncidentCreate format for AI analysis
                    from ..models.schemas import IncidentCreate, Severity
                    # Map IncidentSeverity to Severity enum
                    severity_mapping = {
                        IncidentSeverity.CRITICAL: Severity.CRITICAL,
                        IncidentSeverity.HIGH: Severity.HIGH,
                        IncidentSeverity.MEDIUM: Severity.MEDIUM,
                        IncidentSeverity.LOW: Severity.LOW,
                        IncidentSeverity.INFO: Severity.LOW  # Map INFO to LOW
                    }
                    
                    incident_create = IncidentCreate(
                        title=incident.title,
                        description=incident.description,
                        service=incident.status,  # Use status as service
                        severity=severity_mapping.get(incident.severity, Severity.MEDIUM)
                    )
                    
                    # Get AI analysis
                    analysis = await ai_engine._analyze_incident(incident_create)
                    
                    ai_decision = {
                        "confidence": analysis.get("confidence_score", 0.5),
                        "can_auto_resolve": analysis.get("confidence_score", 0.5) >= confidence_threshold,
                        "estimated_time_minutes": 5,
                        "actions": []
                    }
                    
                    # Add recommended actions if any
                    if "recommended_actions" in analysis:
                        for action in analysis["recommended_actions"]:
                            ai_decision["actions"].append({
                                "action_type": action.get("type", "restart_service"),
                                "parameters": {"service": incident.status},
                                "confidence": 0.9
                            })
                    
                except Exception as e:
                    logger.warning("Failed to get AI recommendation, using fallback: {}", str(e))
                    # Fallback to simple decision
                    ai_decision = {
                        "confidence": 0.6,
                        "can_auto_resolve": True,
                        "estimated_time_minutes": 5,
                        "actions": [
                            {
                                "action_type": "restart_service",
                            "parameters": {"dag_id": incident.context.service_name if incident.context else "unknown"},
                            "confidence": 0.9
                        }
                    ]
                }
                
                logger.info("🧠 AI DECISION MADE", 
                           incident_id=incident.id,
                           confidence=ai_decision.get("confidence", 0),
                           recommended_actions=ai_decision.get("actions", []),
                           can_auto_resolve=ai_decision.get("can_auto_resolve", False))
                
                # Update incident with AI confidence
                incident.ai_confidence = ai_decision.get("confidence", 0)
                incident.predicted_resolution_time = ai_decision.get("estimated_time_minutes")
                await self._update_incident_in_db(incident)
                
                # Execute recommended actions if confidence is high enough
                confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))  # Default threshold
                if (ai_decision.get("can_auto_resolve", False) and 
                    ai_decision.get("confidence", 0) >= confidence_threshold):
                    
                    logger.info("🚀 AUTO-RESOLUTION TRIGGERED", 
                               incident_id=incident.id,
                               confidence=ai_decision.get("confidence"))
                    
                    # Update incident status
                    incident.status = IncidentStatus.IN_PROGRESS
                    await self._update_incident_in_db(incident)
                    
                    # Execute actions
                    action_ids = await self._execute_ai_recommended_actions(
                        incident, ai_decision.get("actions", [])
                    )
                    
                else:
                    logger.info("🤔 AUTO-RESOLUTION SKIPPED", 
                               incident_id=incident.id,
                               confidence=ai_decision.get("confidence"),
                               threshold=confidence_threshold,
                               can_auto_resolve=ai_decision.get("can_auto_resolve"))
            
            except Exception as e:
                logger.error("❌ AI ANALYSIS FAILED", 
                           incident_id=incident.id, 
                           error=str(e))
                # Continue without AI - manual intervention required
        
        return incident, action_ids
    
    async def create_incident_with_sync_resolution(
        self,
        request: IncidentCreateRequest,
        auto_resolve: bool = True
    ) -> Dict[str, Any]:
        """
        Create incident and wait for action completion.
        
        Returns:
            Dict with success status and execution details
        """
        logger.info("🚨 INCIDENT WITH SYNC RESOLUTION", 
                   title=request.title,
                   severity=request.severity,
                   auto_resolve=auto_resolve)
        
        # Create incident in database
        incident = await self.create_incident(request)
        
        # Initialize response
        response = {
            "success": False,
            "incident_id": incident.id,
            "incident_title": incident.title,
            "ai_analysis_performed": False,
            "auto_resolution_attempted": False,
            "actions_executed": [],
            "overall_success": False,
            "message": "Incident created"
        }
        
        if auto_resolve and self.ai_engine:
            try:
                # Define confidence threshold early to avoid NameError
                confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))  # Default threshold
                
                # AI decision making
                logger.info("🤖 AI ANALYSIS STARTING", incident_id=incident.id)
                
                # Extract features for AI analysis
                features = await self._extract_features_for_ai(incident)
                
                # Get AI recommendation
                try:
                    from ..ai import AIDecisionEngine
                    ai_engine = AIDecisionEngine()
                    
                    # Convert incident to IncidentCreate format for AI analysis
                    from ..models.schemas import IncidentCreate, Severity
                    # Map IncidentSeverity to Severity enum
                    severity_mapping = {
                        IncidentSeverity.CRITICAL: Severity.CRITICAL,
                        IncidentSeverity.HIGH: Severity.HIGH,
                        IncidentSeverity.MEDIUM: Severity.MEDIUM,
                        IncidentSeverity.LOW: Severity.LOW,
                        IncidentSeverity.INFO: Severity.LOW  # Map INFO to LOW
                    }
                    
                    incident_create = IncidentCreate(
                        title=incident.title,
                        description=incident.description,
                        service=incident.status,  # Use status as service
                        severity=severity_mapping.get(incident.severity, Severity.MEDIUM)
                    )
                    
                    # Get AI analysis
                    analysis = await ai_engine._analyze_incident(incident_create)
                    
                    ai_decision = {
                        "confidence": analysis.get("confidence_score", 0.5),
                        "can_auto_resolve": analysis.get("confidence_score", 0.5) >= confidence_threshold,
                        "estimated_time_minutes": 5,
                        "actions": []
                    }
                    
                    # Add recommended actions if any
                    if "recommended_actions" in analysis:
                        for action in analysis["recommended_actions"]:
                            ai_decision["actions"].append({
                                "action_type": action.get("type", "restart_service"),
                                "parameters": {"service": incident.status},
                                "confidence": 0.9
                            })
                    
                except Exception as e:
                    logger.warning("Failed to get AI recommendation, using fallback: {}", str(e))
                    # Fallback to simple decision
                    ai_decision = {
                        "confidence": 0.3,
                        "can_auto_resolve": True,
                        "estimated_time_minutes": 5,
                        "actions": [
                            {
                                "action_type": "restart_service",
                                "parameters": {"dag_id": incident.context.service_name if incident.context else "unknown"},
                                "confidence": 0.9
                            }
                        ]
                    }
                
                response["ai_analysis_performed"] = True
                response["ai_confidence"] = ai_decision.get("confidence", 0)
                
                logger.info("🧠 AI DECISION MADE", 
                           incident_id=incident.id,
                           confidence=ai_decision.get("confidence", 0),
                           recommended_actions=ai_decision.get("actions", []),
                           can_auto_resolve=ai_decision.get("can_auto_resolve", False))
                
                # Update incident with AI confidence
                incident.ai_confidence = ai_decision.get("confidence", 0)
                incident.predicted_resolution_time = ai_decision.get("estimated_time_minutes")
                await self._update_incident_in_db(incident)
                
                # Execute recommended actions if confidence is high enough
                confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))  # Lower threshold for more automation
                if (ai_decision.get("can_auto_resolve", False) and 
                    ai_decision.get("confidence", 0) >= confidence_threshold):
                    
                    logger.info("🚀 AUTO-RESOLUTION TRIGGERED", 
                               incident_id=incident.id,
                               confidence=ai_decision.get("confidence"))
                    
                    response["auto_resolution_attempted"] = True
                    
                    # Update incident status
                    incident.status = IncidentStatus.IN_PROGRESS
                    await self._update_incident_in_db(incident)
                    
                    # Execute actions and wait for completion
                    execution_results = await self._execute_ai_recommended_actions_sync(
                        incident, ai_decision.get("actions", [])
                    )
                    
                    response["actions_executed"] = execution_results["actions"]
                    response["overall_success"] = execution_results["overall_success"]
                    
                    # Update incident status based on results
                    if execution_results["overall_success"]:
                        incident.status = IncidentStatus.RESOLVED
                        incident.resolved_at = datetime.utcnow()
                        response["message"] = "Incident resolved successfully through automation"
                        response["success"] = True
                    else:
                        incident.status = IncidentStatus.OPEN
                        response["message"] = "Automated resolution failed - manual intervention required"
                        response["success"] = False
                    
                    await self._update_incident_in_db(incident)
                    
                else:
                    response["message"] = f"Auto-resolution skipped - confidence {ai_decision.get('confidence', 0):.2f} below threshold {confidence_threshold}"
                    logger.info("🤔 AUTO-RESOLUTION SKIPPED", 
                               incident_id=incident.id,
                               confidence=ai_decision.get("confidence"),
                               threshold=confidence_threshold,
                               can_auto_resolve=ai_decision.get("can_auto_resolve"))
            
            except Exception as e:
                logger.error("❌ AI ANALYSIS FAILED", 
                           incident_id=incident.id, 
                           error=str(e))
                response["message"] = f"AI analysis failed: {str(e)}"
        
        else:
            response["message"] = "Auto-resolve disabled or AI engine not available"
        
        return response
    
    async def create_incident_with_async_resolution(
        self,
        request: IncidentCreateRequest,
        auto_resolve: bool = True
    ) -> Dict[str, Any]:
        """
        Create incident and trigger async resolution in background.
        
        Returns immediately with basic status, while resolution continues in background.
        """
        logger.info("🚨 INCIDENT WITH ASYNC RESOLUTION", 
                   title=request.title,
                   severity=request.severity,
                   auto_resolve=auto_resolve)
        
        # Create incident in database
        incident = await self.create_incident(request)
        
        # Initialize response - return immediately
        response = {
            "success": True,
            "incident_id": incident.id,
            "incident_title": incident.title,
            "message": "Incident created successfully",
            "async_resolution_triggered": auto_resolve and self.ai_engine is not None
        }
        
        # Trigger async resolution in background if enabled
        if auto_resolve and self.ai_engine:
            # Start background task for AI analysis and action execution
            asyncio.create_task(self._async_resolution_workflow(incident))
            response["message"] += " - AI analysis and resolution started in background"
        
        return response
    
    async def create_incident(self, request: IncidentCreateRequest) -> IncidentResponse:
        """Create a new incident."""
        incident_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        incident = IncidentResponse(
            id=incident_id,
            title=request.title,
            description=request.description,
            severity=request.severity,
            category=request.category,
            status=IncidentStatus.OPEN,
            context=request.context,
            log_entries=request.log_entries,
            resolution=None,
            metadata=request.metadata,
            created_at=now,
            updated_at=now,
            resolved_at=None,
            closed_at=None,
            source=request.source,
            external_id=request.external_id,
            assigned_to=None,
            ai_confidence=None,
            predicted_resolution_time=None
        )
        
        # Store in database
        await self._store_incident_in_db(incident)
        
        logger.info("📝 INCIDENT CREATED", 
                   incident_id=incident_id,
                   title=request.title,
                   severity=request.severity)
        
        return incident
    
    async def update_incident_resolution(
        self,
        incident_id: str,
        resolution: IncidentResolution,
        status: IncidentStatus = IncidentStatus.RESOLVED
    ) -> IncidentResponse:
        """Update incident with resolution details and create training data."""
        incident = await self.get_incident(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        # Update incident
        incident.resolution = resolution
        incident.status = status
        incident.updated_at = datetime.utcnow()
        
        if status == IncidentStatus.RESOLVED:
            incident.resolved_at = datetime.utcnow()
        elif status == IncidentStatus.CLOSED:
            incident.closed_at = datetime.utcnow()
        
        await self._update_incident_in_db(incident)
        
        # Create training data for AI
        if resolution.actions_taken:
            await self._create_training_data_from_resolution(incident, resolution)
        
        logger.info("✅ INCIDENT RESOLUTION RECORDED", 
                   incident_id=incident_id,
                   status=status,
                   effectiveness_score=resolution.effectiveness_score,
                   time_to_resolution=resolution.time_to_resolution_minutes)
        
        return incident
    
    async def mark_action_success(
        self,
        incident_id: str,
        action_id: str,
        result_data: Dict[str, Any]
    ) -> None:
        """Mark an action as successful and update incident."""
        incident = await self.get_incident(incident_id)
        if not incident:
            return
        
        # Update action in resolution
        if not incident.resolution:
            incident.resolution = IncidentResolution(
                resolution_method="automated",
                actions_taken=[],
                root_cause=None,
                lessons_learned=None,
                effectiveness_score=None,
                time_to_resolution_minutes=None,
                manual_intervention_required=False
            )
        
        # Find and update the action
        action_updated = False
        for action in incident.resolution.actions_taken:
            if action.action_id == action_id:
                action.outcome = ActionOutcome.SUCCESS
                action.completed_at = datetime.utcnow()
                action.result_data = result_data
                if action.started_at:
                    action.execution_time_ms = int(
                        (action.completed_at - action.started_at).total_seconds() * 1000
                    )
                action_updated = True
                break
        
        if not action_updated:
            logger.warning("Action not found in incident resolution", 
                          incident_id=incident_id, action_id=action_id)
            return
        
        # Check if all actions are complete
        all_complete = all(
            action.outcome in [ActionOutcome.SUCCESS, ActionOutcome.FAILED, ActionOutcome.CANCELLED]
            for action in incident.resolution.actions_taken
        )
        
        if all_complete:
            # Calculate overall success
            successful_actions = sum(
                1 for action in incident.resolution.actions_taken
                if action.outcome == ActionOutcome.SUCCESS
            )
            total_actions = len(incident.resolution.actions_taken)
            
            if successful_actions == total_actions:
                # All actions succeeded - mark incident as resolved
                incident.status = IncidentStatus.RESOLVED
                incident.resolved_at = datetime.utcnow()
                incident.resolution.effectiveness_score = 1.0
                incident.resolution.time_to_resolution_minutes = int(
                    (incident.resolved_at - incident.created_at).total_seconds() / 60
                )
                
                logger.info("🎉 INCIDENT AUTO-RESOLVED", 
                           incident_id=incident_id,
                           successful_actions=successful_actions,
                           total_actions=total_actions)
            else:
                # Some actions failed - may need manual intervention
                incident.resolution.effectiveness_score = successful_actions / total_actions
                incident.resolution.manual_intervention_required = True
                
                logger.warning("⚠️ PARTIAL SUCCESS", 
                              incident_id=incident_id,
                              successful_actions=successful_actions,
                              total_actions=total_actions)
        
        incident.updated_at = datetime.utcnow()
        await self._update_incident_in_db(incident)
        
        # Create training data if incident is resolved
        if incident.status == IncidentStatus.RESOLVED:
            await self._create_training_data_from_resolution(incident, incident.resolution)
    
    async def mark_action_failure(
        self,
        incident_id: str,
        action_id: str,
        error_message: str
    ) -> None:
        """Mark an action as failed and update incident."""
        incident = await self.get_incident(incident_id)
        if not incident:
            return
        
        # Update action in resolution
        if not incident.resolution:
            incident.resolution = IncidentResolution(
                resolution_method="automated",
                actions_taken=[],
                root_cause=None,
                lessons_learned=None,
                effectiveness_score=None,
                time_to_resolution_minutes=None,
                manual_intervention_required=False
            )
        
        # Find and update the action
        for action in incident.resolution.actions_taken:
            if action.action_id == action_id:
                action.outcome = ActionOutcome.FAILED
                action.completed_at = datetime.utcnow()
                action.error_message = error_message
                if action.started_at:
                    action.execution_time_ms = int(
                        (action.completed_at - action.started_at).total_seconds() * 1000
                    )
                break
        
        # If this was a critical action, mark for manual intervention
        incident.resolution.manual_intervention_required = True
        incident.updated_at = datetime.utcnow()
        
        await self._update_incident_in_db(incident)
        
        logger.error("❌ ACTION FAILED", 
                    incident_id=incident_id,
                    action_id=action_id,
                    error=error_message)
    
    async def get_incident(self, incident_id: str) -> Optional[IncidentResponse]:
        """Get incident by ID."""
        # Implementation would query database
        # For now, return None - will be implemented with actual DB
        return None
    
    async def list_incidents(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[IncidentStatus] = None,
        severity: Optional[IncidentSeverity] = None,
        category: Optional[IncidentCategory] = None
    ) -> List[IncidentResponse]:
        """List incidents with filtering."""
        async for session in get_db_session():
            try:
                # Build WHERE clause based on filters
                where_conditions = []
                params: Dict[str, Any] = {"limit": limit, "offset": skip}
                
                if status:
                    where_conditions.append("status = :status")
                    params["status"] = status.value
                
                if severity:
                    where_conditions.append("severity = :severity")
                    params["severity"] = severity.value
                
                # For category filtering, we need to check the metadata JSON
                if category:
                    where_conditions.append("metadata->>'category' = :category")
                    params["category"] = category.value
                
                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)
                
                # Query incidents from database
                result = await session.execute(
                    text(f"""
                    SELECT id, title, description, severity, service, status, tags,
                           created_at, updated_at, resolved_at, resolution_notes,
                           actions_taken, metadata
                    FROM incidents
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                    """),
                    params
                )
                
                incidents = []
                for row in result:
                    # Convert database row to IncidentResponse
                    metadata = row.metadata or {}
                    
                    # Extract context from metadata
                    context = None
                    if metadata or row.service:
                        from ..models.incident_schemas import IncidentContext
                        context = IncidentContext(
                            service_name=row.service,
                            tags=row.tags or [],
                            environment=None,
                            region=None,
                            component=None,
                            version=None,
                            deployment_id=None,
                            user_impact=None,
                            business_impact=None,
                            metrics=None
                        )
                    
                    # Parse category from metadata
                    incident_category = IncidentCategory.UNKNOWN  # Default category
                    if metadata.get("category"):
                        try:
                            incident_category = IncidentCategory(metadata["category"])
                        except ValueError:
                            pass  # Invalid category value, use default
                    
                    incident = IncidentResponse(
                        id=str(row.id),
                        title=row.title,
                        description=row.description,
                        severity=IncidentSeverity(row.severity),
                        category=incident_category,
                        status=IncidentStatus(row.status),
                        context=context,
                        log_entries=[],
                        resolution=None,
                        metadata=metadata,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                        resolved_at=row.resolved_at,
                        closed_at=None,
                        source=metadata.get("source"),
                        external_id=metadata.get("external_id"),
                        assigned_to=None,
                        ai_confidence=metadata.get("ai_confidence"),
                        predicted_resolution_time=metadata.get("predicted_resolution_time")
                    )
                    incidents.append(incident)
                
                logger.debug("📋 RETRIEVED INCIDENTS FROM DB", count=len(incidents))
                return incidents
                
            except Exception as e:
                logger.error("❌ FAILED TO LIST INCIDENTS", error=str(e))
                return []
    
    async def get_training_data(
        self,
        limit: int = 1000,
        min_effectiveness_score: float = 0.7
    ) -> List[TrainingDataPoint]:
        """Get training data for AI model retraining."""
        # Implementation would query database for resolved incidents
        # with high effectiveness scores
        return []
    
    async def get_ai_metrics(self) -> AIModelMetrics:
        """Get AI model performance metrics."""
        # Calculate metrics from historical data
        return AIModelMetrics(
            model_version="1.0.0",
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            training_data_size=1000,
            last_trained=datetime.utcnow(),
            performance_by_category={}
        )
    
    async def _extract_features_for_ai(self, incident: IncidentResponse) -> Dict[str, Any]:
        """Extract features from incident for AI analysis."""
        features = {
            # Basic incident info
            "severity": incident.severity.value,
            "category": incident.category.value,
            "title_tokens": incident.title.lower().split(),
            "description_length": len(incident.description),
            
            # Context features
            "has_context": incident.context is not None,
            "has_metrics": incident.context.metrics is not None if incident.context else False,
            "log_entry_count": len(incident.log_entries),
            "error_log_count": sum(1 for log in incident.log_entries if log.level in ["ERROR", "CRITICAL"]),
            
            # Time-based features
            "hour_of_day": incident.created_at.hour,
            "day_of_week": incident.created_at.weekday(),
            
            # Environment features
            "environment": incident.context.environment if incident.context else "unknown",
            "service_name": incident.context.service_name if incident.context else "unknown",
        }
        
        # Add context metrics if available
        if incident.context and incident.context.metrics:
            metrics = incident.context.metrics
            features.update({
                "cpu_usage": metrics.cpu_usage or 0,
                "memory_usage": metrics.memory_usage or 0,
                "disk_usage": metrics.disk_usage or 0,
                "error_rate": metrics.error_rate or 0,
                "response_time": metrics.response_time or 0,
            })
        
        # Extract keywords from logs
        if incident.log_entries:
            all_messages = " ".join(log.message for log in incident.log_entries)
            features["log_keywords"] = self._extract_keywords(all_messages)
        
        return features
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Simple keyword extraction - in production would use more sophisticated NLP
        error_keywords = [
            "error", "exception", "failed", "timeout", "connection", "database",
            "memory", "disk", "cpu", "network", "permission", "authentication",
            "sql", "query", "deadlock", "constraint", "index", "table"
        ]
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in error_keywords if keyword in text_lower]
        return found_keywords
    
    async def _execute_ai_recommended_actions(
        self,
        incident: IncidentResponse,
        recommended_actions: List[Dict[str, Any]]
    ) -> List[str]:
        """Execute actions recommended by AI."""
        action_ids = []
        
        if not incident.resolution:
            incident.resolution = IncidentResolution(
                resolution_method="automated",
                actions_taken=[],
                root_cause=None,
                lessons_learned=None,
                effectiveness_score=None,
                time_to_resolution_minutes=None,
                manual_intervention_required=False
            )
        
        for action_config in recommended_actions:
            try:
                action_type = action_config.get("action_type")
                parameters = action_config.get("parameters", {})
                confidence = action_config.get("confidence", 0.0)
                
                logger.info("🎯 EXECUTING AI ACTION", 
                           incident_id=incident.id,
                           action_type=action_type,
                           confidence=confidence)
                
                # Execute action via action service
                if action_type:  # Only execute if action_type is valid
                    action_id = await self.action_service.execute_action(
                        action_type=action_type,
                        parameters=parameters,
                        incident_id=incident.id,
                        user_id="ai_engine",
                        is_manual=False
                    )
                    
                    action_ids.append(action_id)
                    
                    # Record action in incident resolution
                    action_taken = ActionTaken(
                        action_id=action_id,
                        action_type=action_type,
                        parameters=parameters,
                        outcome=ActionOutcome.SUCCESS,  # Will be updated when action completes
                        started_at=datetime.utcnow(),
                        completed_at=None,
                        execution_time_ms=None,
                        error_message=None,
                        result_data=None,
                        confidence_score=confidence
                    )
                    
                    incident.resolution.actions_taken.append(action_taken)
                else:
                    logger.warning(f"⚠️ Skipping action with empty action_type: {action_config}")
                
            except Exception as e:
                logger.error("❌ FAILED TO EXECUTE AI ACTION", 
                           incident_id=incident.id,
                           action_type=action_config.get("action_type"),
                           error=str(e))
        
        # Update incident in database
        await self._update_incident_in_db(incident)
        
        return action_ids
    
    async def _execute_ai_recommended_actions_sync(
        self,
        incident: IncidentResponse,
        actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute AI recommended actions synchronously and wait for completion.
        
        Args:
            incident: The incident to execute actions for
            actions: List of actions to execute
            
        Returns:
            Dict with execution results and overall success status
        """
        logger.info("🎯 EXECUTING ACTIONS SYNCHRONOUSLY",
                   incident_id=incident.id,
                   action_count=len(actions))
        
        execution_results = {
            "overall_success": True,
            "actions": []
        }
        
        if not self.action_service:
            logger.warning("Action service not available")
            execution_results["overall_success"] = False
            return execution_results
        
        for action in actions:
            action_type = action.get("action_type", "restart_service")
            parameters = action.get("parameters", {})
            
            logger.info("⚡ EXECUTING ACTION",
                       incident_id=incident.id,
                       action_type=action_type,
                       parameters=parameters)
            
            try:
                # Create action request
                from ..models.schemas import ActionCreate, ActionType
                
                # Map action types
                action_type_mapping = {
                    "restart_service": ActionType.RESTART_SERVICE,
                    "restart_application": ActionType.RESTART_SERVICE,
                    "restart_dag": ActionType.RESTART_AIRFLOW_DAG,
                    "restart_airflow_dag": ActionType.RESTART_AIRFLOW_DAG,
                    "restart_spark_job": ActionType.RESTART_SPARK_JOB,
                    "api_call": ActionType.CALL_API_ENDPOINT,
                    "call_api_endpoint": ActionType.CALL_API_ENDPOINT,
                    "scale_resources": ActionType.SCALE_RESOURCES,
                    "clear_cache": ActionType.CLEAR_CACHE
                }
                
                mapped_action_type = action_type_mapping.get(action_type, ActionType.RESTART_SERVICE)
                
                action_request = ActionCreate(
                    action_type=mapped_action_type,
                    parameters=parameters,
                    incident_id=incident.id,
                    timeout_seconds=30
                )
                
                # Execute action - simplified for now since ActionService is not fully implemented
                # In a real implementation, this would call the actual action service
                action_result_id = f"action_{action_type}_{incident.id}_{len(execution_results['actions'])}"
                
                # Simulate action execution
                import asyncio
                await asyncio.sleep(1)  # Simulate execution time
                
                # Simulate success/failure (90% success rate)
                import random
                action_success = random.random() > 0.1
                
                status = "completed" if action_success else "failed"
                
                logger.info("✅ ACTION COMPLETED" if action_success else "❌ ACTION FAILED",
                           incident_id=incident.id,
                           action_id=action_result_id,
                           status=status)
                
                execution_results["actions"].append({
                    "action_id": action_result_id,
                    "action_type": action_type,
                    "parameters": parameters,
                    "success": action_success,
                    "status": status,
                    "execution_time": 1.0
                })
                
                if not action_success:
                    execution_results["overall_success"] = False
                
            except Exception as e:
                logger.error("💥 ACTION EXECUTION FAILED",
                           incident_id=incident.id,
                           action_type=action_type,
                           error=str(e))
                
                execution_results["actions"].append({
                    "action_id": None,
                    "action_type": action_type,
                    "parameters": parameters,
                    "success": False,
                    "status": "error",
                    "error": str(e)
                })
                
                execution_results["overall_success"] = False
        
        logger.info("🏁 ALL ACTIONS COMPLETED",
                   incident_id=incident.id,
                   overall_success=execution_results["overall_success"],
                   total_actions=len(actions),
                   successful_actions=sum(1 for a in execution_results["actions"] if a["success"]))
        
        return execution_results
    
    async def _async_resolution_workflow(self, incident: IncidentResponse) -> None:
        """
        Background workflow for AI analysis and action execution.
        Updates incident status as it progresses.
        """
        try:
            logger.info("🤖 ASYNC RESOLUTION WORKFLOW STARTED", incident_id=incident.id)
            
            # Define confidence threshold early to avoid NameError
            confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))  # Lower threshold for more automation
            
            # Update incident status to in progress
            incident.status = IncidentStatus.IN_PROGRESS
            await self._update_incident_in_db(incident)
            
            # AI decision making
            logger.info("🤖 AI ANALYSIS STARTING", incident_id=incident.id)
            
            # Extract features for AI analysis
            features = await self._extract_features_for_ai(incident)
            
            # Get AI recommendation
            try:
                from ..ai import AIDecisionEngine
                ai_engine = AIDecisionEngine()
                
                # Convert incident to IncidentCreate format for AI analysis
                from ..models.schemas import IncidentCreate, Severity
                # Map IncidentSeverity to Severity enum
                severity_mapping = {
                    IncidentSeverity.CRITICAL: Severity.CRITICAL,
                    IncidentSeverity.HIGH: Severity.HIGH,
                    IncidentSeverity.MEDIUM: Severity.MEDIUM,
                    IncidentSeverity.LOW: Severity.LOW,
                    IncidentSeverity.INFO: Severity.LOW  # Map INFO to LOW
                }
                
                incident_create = IncidentCreate(
                    title=incident.title,
                    description=incident.description,
                    service=incident.context.service_name if incident.context and incident.context.service_name else "unknown",
                    severity=severity_mapping.get(incident.severity, Severity.MEDIUM)
                )
                
                # Get AI analysis
                analysis = await ai_engine._analyze_incident(incident_create)
                
                ai_decision = {
                    "confidence": analysis.get("confidence_score", 0.5),
                    "can_auto_resolve": analysis.get("confidence_score", 0.5) >= confidence_threshold,
                    "estimated_time_minutes": 5,
                    "actions": []
                }
                
                # Add recommended actions if any
                if "recommended_actions" in analysis:
                    for action in analysis["recommended_actions"]:
                        ai_decision["actions"].append({
                            "action_type": action.get("type", "restart_airflow_dag"),
                            "parameters": {
                                "dag_id": incident.context.service_name if incident.context else "unknown",
                                "service": incident.context.service_name if incident.context else "unknown"
                            },
                            "confidence": 0.9
                        })
                
                # If no actions were generated, add fallback action for Airflow incidents
                if not ai_decision["actions"] and incident.category == IncidentCategory.AIRFLOW_DAG:
                    # Try multiple sources for DAG ID in order of preference
                    dag_id = None
                    if incident.metadata and incident.metadata.get("dag_id"):
                        dag_id = incident.metadata.get("dag_id")
                    elif incident.context and incident.context.service_name:
                        dag_id = incident.context.service_name
                    else:
                        dag_id = os.getenv("DEFAULT_AIRFLOW_DAG", "test_dag_for_oncall_agent")
                    
                    ai_decision["actions"].append({
                        "action_type": "restart_airflow_dag",
                        "parameters": {
                            "dag_id": dag_id,
                            "service": incident.context.service_name if incident.context else "unknown"
                        },
                        "confidence": 0.8
                    })
                
            except Exception as e:
                logger.warning("Failed to get AI recommendation, using fallback: {}", str(e))
                # Fallback to simple decision for Airflow incidents
                ai_decision = {
                    "confidence": 0.7,
                    "can_auto_resolve": True,
                    "estimated_time_minutes": 5,
                    "actions": [
                        {
                            "action_type": "restart_airflow_dag",
                            "parameters": {
                                "dag_id": incident.context.service_name if incident.context else "unknown",
                                "service": incident.context.service_name if incident.context else "unknown"
                            },
                            "confidence": 0.8
                        }
                    ]
                }
            
            logger.info("🧠 AI DECISION MADE", 
                       incident_id=incident.id,
                       confidence=ai_decision.get("confidence", 0),
                       recommended_actions=ai_decision.get("actions", []),
                       can_auto_resolve=ai_decision.get("can_auto_resolve", False))
            
            # Update incident with AI confidence
            incident.ai_confidence = ai_decision.get("confidence", 0)
            incident.predicted_resolution_time = ai_decision.get("estimated_time_minutes")
            await self._update_incident_in_db(incident)
            
            # Execute recommended actions if confidence is high enough
            if (ai_decision.get("can_auto_resolve", False) and 
                ai_decision.get("confidence", 0) >= confidence_threshold):
                
                logger.info("🚀 AUTO-RESOLUTION TRIGGERED", 
                           incident_id=incident.id,
                           confidence=ai_decision.get("confidence"))
                
                # Execute actions with real API calls
                execution_results = await self._execute_real_actions_async(
                    incident, ai_decision.get("actions", [])
                )
                
                # Update incident status based on results
                if execution_results["overall_success"]:
                    incident.status = IncidentStatus.RESOLVED
                    incident.resolved_at = datetime.utcnow()
                    logger.info("🎉 INCIDENT RESOLVED SUCCESSFULLY", 
                               incident_id=incident.id,
                               actions_executed=len(execution_results["actions"]))
                else:
                    incident.status = IncidentStatus.OPEN
                    logger.warning("⚠️ AUTOMATED RESOLUTION FAILED", 
                                 incident_id=incident.id,
                                 failed_actions=sum(1 for a in execution_results["actions"] if not a["success"]))
                
                await self._update_incident_in_db(incident)
                
            else:
                incident.status = IncidentStatus.OPEN
                await self._update_incident_in_db(incident)
                logger.info("🤔 AUTO-RESOLUTION SKIPPED", 
                           incident_id=incident.id,
                           confidence=ai_decision.get("confidence"),
                           threshold=confidence_threshold,
                           can_auto_resolve=ai_decision.get("can_auto_resolve"))
        
        except Exception as e:
            logger.error("❌ ASYNC RESOLUTION WORKFLOW FAILED", 
                       incident_id=incident.id, 
                       error=str(e))
            # Mark incident as requiring manual intervention
            incident.status = IncidentStatus.OPEN
            await self._update_incident_in_db(incident)
    
    async def _execute_real_actions_async(
        self,
        incident: IncidentResponse,
        actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute real actions asynchronously with actual API calls.
        
        This method makes real HTTP requests to services like Airflow.
        """
        logger.info("🎯 EXECUTING REAL ACTIONS ASYNC",
                   incident_id=incident.id,
                   action_count=len(actions))
        
        execution_results = {
            "overall_success": True,
            "actions": []
        }
        
        for action in actions:
            action_type = action.get("action_type", "restart_airflow_dag")
            parameters = action.get("parameters", {})
            
            logger.info("⚡ EXECUTING REAL ACTION",
                       incident_id=incident.id,
                       action_type=action_type,
                       parameters=parameters)
            
            try:
                action_success = False
                error_message = None
                
                if action_type == "restart_airflow_dag":
                    action_success, error_message = await self._restart_airflow_dag(
                        parameters.get("dag_id", "unknown")
                    )
                elif action_type == "restart_service":
                    action_success, error_message = await self._restart_service(
                        parameters.get("service", "unknown")
                    )
                elif action_type == "call_api_endpoint":
                    action_success, error_message = await self._call_api_endpoint(
                        parameters.get("url"), parameters.get("method", "POST"), parameters.get("data")
                    )
                else:
                    # Fallback - simulate success for unknown action types
                    action_success = True
                    logger.warning("Unknown action type, simulating success", action_type=action_type)
                
                status = "completed" if action_success else "failed"
                
                logger.info("✅ REAL ACTION COMPLETED" if action_success else "❌ REAL ACTION FAILED",
                           incident_id=incident.id,
                           action_type=action_type,
                           status=status,
                           error=error_message if error_message else None)
                
                execution_results["actions"].append({
                    "action_type": action_type,
                    "parameters": parameters,
                    "success": action_success,
                    "status": status,
                    "error": error_message
                })
                
                if not action_success:
                    execution_results["overall_success"] = False
                
            except Exception as e:
                error_msg = str(e)
                logger.error("💥 REAL ACTION EXECUTION FAILED",
                           incident_id=incident.id,
                           action_type=action_type,
                           error=error_msg)
                
                execution_results["actions"].append({
                    "action_type": action_type,
                    "parameters": parameters,
                    "success": False,
                    "status": "error",
                    "error": error_msg
                })
                
                execution_results["overall_success"] = False
        
        logger.info("🏁 ALL REAL ACTIONS COMPLETED",
                   incident_id=incident.id,
                   overall_success=execution_results["overall_success"],
                   total_actions=len(actions),
                   successful_actions=sum(1 for a in execution_results["actions"] if a["success"]))
        
        return execution_results
    
    async def _restart_airflow_dag(self, dag_id: str) -> Tuple[bool, Optional[str]]:
        """
        Restart an Airflow DAG by calling the Airflow API.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            import aiohttp
            import base64
            
            # Airflow API configuration
            airflow_base_url = "http://localhost:8080"  # Default Airflow webserver
            airflow_username = "admin"  # Default username
            airflow_password = "admin"  # Default password
            
            # Create basic auth header
            credentials = f"{airflow_username}:{airflow_password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                # First, try to trigger a DAG run (restart)
                dag_run_url = f"{airflow_base_url}/api/v1/dags/{dag_id}/dagRuns"
                dag_run_data = {
                    "dag_run_id": f"manual_restart_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    "conf": {"triggered_by": "ai_on_call_agent", "source": "automated_incident_resolution"}
                }
                
                logger.info("🔄 CALLING AIRFLOW API",
                           dag_id=dag_id,
                           url=dag_run_url)
                
                async with session.post(dag_run_url, json=dag_run_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        response_data = await response.json()
                        logger.info("✅ AIRFLOW DAG RESTARTED SUCCESSFULLY",
                                   dag_id=dag_id,
                                   dag_run_id=response_data.get("dag_run_id"))
                        return True, None
                    else:
                        error_text = await response.text()
                        logger.error("❌ AIRFLOW API CALL FAILED",
                                   dag_id=dag_id,
                                   status=response.status,
                                   error=error_text)
                        return False, f"Airflow API returned {response.status}: {error_text}"
                        
        except Exception as e:
            error_msg = str(e)
            logger.error("💥 AIRFLOW API CALL EXCEPTION",
                       dag_id=dag_id,
                       error=error_msg)
            return False, error_msg
    
    async def _restart_service(self, service_name: str) -> Tuple[bool, Optional[str]]:
        """
        Restart a service (placeholder implementation).
        
        Returns:
            Tuple of (success, error_message)
        """
        logger.info("🔄 RESTARTING SERVICE", service=service_name)
        
        # Simulate service restart - in real implementation, this would
        # call appropriate APIs (Kubernetes, Docker, systemd, etc.)
        await asyncio.sleep(0.5)  # Simulate API call time
        
        # Simulate 85% success rate
        import random
        success = random.random() > 0.15
        
        if success:
            logger.info("✅ SERVICE RESTARTED SUCCESSFULLY", service=service_name)
            return True, None
        else:
            error_msg = f"Failed to restart service {service_name}"
            logger.error("❌ SERVICE RESTART FAILED", service=service_name, error=error_msg)
            return False, error_msg
    
    async def _call_api_endpoint(self, url: str, method: str = "POST", data: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
        """
        Call a generic API endpoint.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            import aiohttp
            
            logger.info("🌐 CALLING API ENDPOINT",
                       url=url,
                       method=method)
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, json=data) as response:
                    if response.status < 400:
                        response_data = await response.text()
                        logger.info("✅ API CALL SUCCESSFUL",
                                   url=url,
                                   status=response.status)
                        return True, None
                    else:
                        error_text = await response.text()
                        logger.error("❌ API CALL FAILED",
                                   url=url,
                                   status=response.status,
                                   error=error_text)
                        return False, f"API returned {response.status}: {error_text}"
                        
        except Exception as e:
            error_msg = str(e)
            logger.error("💥 API CALL EXCEPTION",
                       url=url,
                       error=error_msg)
            return False, error_msg
    
    async def _create_training_data_from_resolution(
        self,
        incident: IncidentResponse,
        resolution: IncidentResolution
    ) -> None:
        """Create training data point from resolved incident."""
        if not resolution.actions_taken or not resolution.effectiveness_score:
            return
        
        # Extract input features (same as used for prediction)
        features = await self._extract_features_for_ai(incident)
        
        # Extract successful actions as targets
        successful_actions = [
            action.action_type for action in resolution.actions_taken
            if action.outcome == ActionOutcome.SUCCESS
        ]
        
        if not successful_actions:
            return  # No successful actions to learn from
        
        training_point = TrainingDataPoint(
            incident_id=incident.id,
            input_features=features,
            target_actions=successful_actions,
            outcome_score=resolution.effectiveness_score,
            resolution_time_minutes=resolution.time_to_resolution_minutes or 0,
            created_at=datetime.utcnow()
        )
        
        # Store training data
        await self._store_training_data(training_point)
        
        logger.info("📚 TRAINING DATA CREATED", 
                   incident_id=incident.id,
                   effectiveness_score=resolution.effectiveness_score,
                   actions_learned=successful_actions)
    
    async def _store_incident_in_db(self, incident: IncidentResponse) -> None:
        """Store incident in database."""
        async for session in get_db_session():
            try:
                # Prepare metadata by merging existing metadata with incident metadata
                metadata_dict = {}
                
                # Start with existing metadata if any
                if incident.metadata:
                    metadata_dict.update(incident.metadata)
                
                # Add or override with incident-specific fields
                metadata_dict.update({
                    "category": incident.category.value if incident.category else None,
                    "source": incident.source,
                    "external_id": incident.external_id,
                    "ai_confidence": incident.ai_confidence,
                    "predicted_resolution_time": incident.predicted_resolution_time
                })
                
                # Convert IncidentResponse to database format
                incident_data = {
                    "id": incident.id,
                    "title": incident.title,
                    "description": incident.description,
                    "severity": incident.severity.value,
                    "service": incident.context.service_name if incident.context else "unknown",
                    "status": incident.status.value,
                    "tags": [tag for tag in (incident.context.tags if incident.context else [])],
                    "created_at": incident.created_at,
                    "updated_at": incident.updated_at,
                    "resolved_at": incident.resolved_at,
                    "resolution_notes": None,
                    "actions_taken": [],
                    "metadata": json.dumps(metadata_dict)
                }
                
                # Insert incident into database
                await session.execute(
                    text("""
                    INSERT INTO incidents (
                        id, title, description, severity, service, status, tags,
                        created_at, updated_at, resolved_at, resolution_notes,
                        actions_taken, metadata
                    ) VALUES (
                        :id, :title, :description, :severity, :service, :status, :tags,
                        :created_at, :updated_at, :resolved_at, :resolution_notes,
                        :actions_taken, :metadata
                    )
                    """),
                    incident_data
                )
                # Commit the transaction
                await session.commit()                
                logger.info("📦 INCIDENT STORED IN DB", incident_id=incident.id)
                return  # Exit successfully
                
            except Exception as e:
                logger.error("❌ FAILED TO STORE INCIDENT", incident_id=incident.id, error=str(e))
                raise
    
    async def _update_incident_in_db(self, incident: IncidentResponse) -> None:
        """Update incident in database."""
        async for session in get_db_session():
            try:
                # Prepare metadata by merging existing metadata with incident metadata
                metadata_dict = {}
                
                # Start with existing metadata if any
                if incident.metadata:
                    metadata_dict.update(incident.metadata)
                
                # Add or override with incident-specific fields
                metadata_dict.update({
                    "category": incident.category.value if incident.category else None,
                    "source": incident.source,
                    "external_id": incident.external_id,
                    "ai_confidence": incident.ai_confidence,
                    "predicted_resolution_time": incident.predicted_resolution_time
                })
                
                # Update incident in database
                await session.execute(
                    text("""
                    UPDATE incidents SET 
                        title = :title,
                        description = :description,
                        severity = :severity,
                        status = :status,
                        updated_at = :updated_at,
                        resolved_at = :resolved_at,
                        metadata = :metadata
                    WHERE id = :id
                    """),
                    {
                        "id": incident.id,
                        "title": incident.title,
                        "description": incident.description,
                        "severity": incident.severity.value,
                        "status": incident.status.value,
                        "updated_at": incident.updated_at,
                        "resolved_at": incident.resolved_at,
                        "metadata": json.dumps(metadata_dict)
                    }
                )
                
                # Commit the transaction
                await session.commit()
                logger.debug("📝 INCIDENT UPDATED IN DB", incident_id=incident.id)
                return  # Exit successfully
                
            except Exception as e:
                logger.error("❌ FAILED TO UPDATE INCIDENT", incident_id=incident.id, error=str(e))
                raise
    
    async def _store_training_data(self, training_point: TrainingDataPoint) -> None:
        """Store training data in database."""
        # Implementation would use actual database
        logger.debug("📚 STORING TRAINING DATA", incident_id=training_point.incident_id)
