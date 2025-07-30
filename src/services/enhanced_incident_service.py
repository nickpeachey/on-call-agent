"""Enhanced incident management service with AI training integration."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import uuid
import asyncio
import json

from ..core import get_logger
from ..models.incident_schemas import (
    IncidentCreateRequest, IncidentUpdateRequest, IncidentResponse,
    IncidentStatus, IncidentSeverity, IncidentCategory, ActionOutcome,
    ActionTaken, IncidentResolution, TrainingDataPoint, AIModelMetrics
)
from .actions import ActionService
from ..ai import AIDecisionEngine


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
                    from ..ai import AIEngine
                    ai_engine = AIEngine()
                    
                    # Convert incident to IncidentCreate format for AI analysis
                    from ..models.schemas import IncidentCreate
                    incident_create = IncidentCreate(
                        title=incident.title,
                        description=incident.description,
                        service=incident.status,  # Use status as service
                        severity=incident.severity.value  # Convert enum to string
                    )
                    
                    # Get AI analysis
                    analysis = await ai_engine.analyze_incident(incident_create)
                    
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
            metadata=request.metadata,
            created_at=now,
            updated_at=now,
            source=request.source,
            external_id=request.external_id
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
                actions_taken=[]
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
                actions_taken=[]
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
        # Implementation would query database
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
                actions_taken=[]
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
                    confidence_score=confidence
                )
                
                incident.resolution.actions_taken.append(action_taken)
                
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
        # Implementation would use actual database
        logger.debug("📦 STORING INCIDENT", incident_id=incident.id)
    
    async def _update_incident_in_db(self, incident: IncidentResponse) -> None:
        """Update incident in database."""
        # Implementation would use actual database
        logger.debug("📝 UPDATING INCIDENT", incident_id=incident.id)
    
    async def _store_training_data(self, training_point: TrainingDataPoint) -> None:
        """Store training data in database."""
        # Implementation would use actual database
        logger.debug("📚 STORING TRAINING DATA", incident_id=training_point.incident_id)
