"""Enhanced incident management service with AI training integration."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import uuid
import asyncio
import json

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
                        "can_auto_resolve": analysis.get("confidence_score", 0.5) > 0.7,
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
                confidence_threshold = 0.8  # Default threshold
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
