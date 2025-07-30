"""Incident management service."""

from typing import List, Optional
from datetime import datetime
import uuid

from ..core import get_logger, settings
from ..models.schemas import IncidentResponse, IncidentStatus, Severity


logger = get_logger(__name__)


class IncidentService:
    """Service for managing incidents with database persistence."""
    
    def __init__(self):
        # Initialize database connection pool when needed
        self._pool = None
    
    async def _get_db_connection(self):
        """Get database connection. This would integrate with your actual database."""
        # For production, this would use SQLAlchemy or your preferred ORM
        # For now, implement a basic in-memory store that persists to files
        return None
    
    async def list_incidents(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[IncidentResponse]:
        """List incidents with optional filtering."""
        logger.info("Listing incidents", skip=skip, limit=limit, status=status)
        
        try:
            # In production, this would query the database
            # For now, return structured data that mimics database results
            incidents = await self._load_incidents_from_storage()
            
            # Apply status filter
            if status:
                incidents = [i for i in incidents if i.status.value == status]
            
            # Apply pagination
            return incidents[skip:skip + limit]
                
        except Exception as e:
            logger.error("Error listing incidents", error=str(e))
            return []
    
    async def _load_incidents_from_storage(self) -> List[IncidentResponse]:
        """Load incidents from storage (file-based for development)."""
        import json
        import os
        
        storage_file = "data/incidents.json"
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        try:
            if os.path.exists(storage_file):
                with open(storage_file, 'r') as f:
                    data = json.load(f)
                    
                incidents = []
                for item in data:
                    incident = IncidentResponse(
                        id=item['id'],
                        title=item['title'],
                        description=item['description'],
                        severity=item['severity'],
                        service=item['service'],
                        status=IncidentStatus(item['status']),
                        tags=item.get('tags', []),
                        created_at=datetime.fromisoformat(item['created_at']),
                        updated_at=datetime.fromisoformat(item['updated_at']),
                        resolved_at=datetime.fromisoformat(item['resolved_at']) if item.get('resolved_at') else None,
                        assigned_to=item.get('assigned_to'),
                        resolution_notes=item.get('resolution_notes'),
                        actions_taken=item.get('actions_taken', [])
                    )
                    incidents.append(incident)
                    
                return incidents
            else:
                # Create initial incidents
                return await self._create_initial_incidents()
                
        except Exception as e:
            logger.error("Error loading incidents from storage", error=str(e))
            return await self._create_initial_incidents()
    
    async def _create_initial_incidents(self) -> List[IncidentResponse]:
        """Create some initial incidents for demonstration."""
        incidents = [
            IncidentResponse(
                id="incident_" + str(uuid.uuid4())[:8],
                title="Database Connection Pool Exhausted",
                description="PostgreSQL connection pool reached maximum capacity, causing application timeouts",
                severity=Severity.HIGH,
                service="database",
                status=IncidentStatus.OPEN,
                tags=["database", "postgresql", "connection-pool"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                resolved_at=None,
                assigned_to=None,
                resolution_notes=None,
                actions_taken=[]
            ),
            IncidentResponse(
                id="incident_" + str(uuid.uuid4())[:8],
                title="Spark Job Memory Leak",
                description="Spark executor memory usage continuously increasing, leading to OOM errors",
                severity=Severity.MEDIUM,
                service="spark-cluster",
                status=IncidentStatus.IN_PROGRESS,
                tags=["spark", "memory", "oom"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                resolved_at=None,
                assigned_to="admin",
                resolution_notes=None,
                actions_taken=["restart_spark_executor", "increase_memory_limit"]
            )
        ]
        
        # Save to storage
        await self._save_incidents_to_storage(incidents)
        return incidents
    
    async def _save_incidents_to_storage(self, incidents: List[IncidentResponse]):
        """Save incidents to storage."""
        import json
        import os
        
        storage_file = "data/incidents.json"
        os.makedirs("data", exist_ok=True)
        
        try:
            data = []
            for incident in incidents:
                data.append({
                    'id': incident.id,
                    'title': incident.title,
                    'description': incident.description,
                    'severity': incident.severity,
                    'service': incident.service,
                    'status': incident.status.value,
                    'tags': incident.tags,
                    'created_at': incident.created_at.isoformat(),
                    'updated_at': incident.updated_at.isoformat(),
                    'resolved_at': incident.resolved_at.isoformat() if incident.resolved_at else None,
                    'assigned_to': incident.assigned_to,
                    'resolution_notes': incident.resolution_notes,
                    'actions_taken': incident.actions_taken
                })
            
            with open(storage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error("Error saving incidents to storage", error=str(e))
    
    async def get_incident(self, incident_id: str) -> Optional[IncidentResponse]:
        """Get specific incident by ID."""
        logger.info("Getting incident", incident_id=incident_id)
        
        try:
            incidents = await self._load_incidents_from_storage()
            for incident in incidents:
                if incident.id == incident_id:
                    return incident
            return None
            
        except Exception as e:
            logger.error("Error getting incident", incident_id=incident_id, error=str(e))
            return None
    
    async def create_incident(
        self,
        title: str,
        description: str,
        severity: str,
        service: str,
        tags: Optional[List[str]] = None
    ) -> IncidentResponse:
        """Create a new incident."""
        incident_id = "incident_" + str(uuid.uuid4())[:8]
        
        # Convert string severity to enum
        severity_enum = Severity(severity) if severity in [s.value for s in Severity] else Severity.MEDIUM
        
        incident = IncidentResponse(
            id=incident_id,
            title=title,
            description=description,
            severity=severity_enum,
            service=service,
            status=IncidentStatus.OPEN,
            tags=tags or [],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resolved_at=None,
            assigned_to=None,
            resolution_notes=None,
            actions_taken=[]
        )
        
        # Add to storage
        incidents = await self._load_incidents_from_storage()
        incidents.append(incident)
        await self._save_incidents_to_storage(incidents)
        
        logger.info("Created new incident", incident_id=incident_id, title=title)
        return incident
    
    async def update_incident_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        resolution_notes: Optional[str] = None
    ) -> Optional[IncidentResponse]:
        """Update incident status."""
        try:
            incidents = await self._load_incidents_from_storage()
            
            for i, incident in enumerate(incidents):
                if incident.id == incident_id:
                    # Update incident
                    updated_incident = incident.model_copy(update={
                        'status': status,
                        'updated_at': datetime.utcnow(),
                        'resolved_at': datetime.utcnow() if status == IncidentStatus.RESOLVED else incident.resolved_at,
                        'resolution_notes': resolution_notes or incident.resolution_notes
                    })
                    
                    incidents[i] = updated_incident
                    await self._save_incidents_to_storage(incidents)
                    
                    logger.info("Updated incident status", incident_id=incident_id, status=status.value)
                    return updated_incident
                    
            return None
            
        except Exception as e:
            logger.error("Error updating incident status", incident_id=incident_id, error=str(e))
            return None
    
    async def assign_incident(
        self,
        incident_id: str,
        user_id: Optional[str] = None
    ) -> Optional[IncidentResponse]:
        """Assign incident to user."""
        try:
            incidents = await self._load_incidents_from_storage()
            
            for i, incident in enumerate(incidents):
                if incident.id == incident_id:
                    updated_incident = incident.model_copy(update={
                        'assigned_to': user_id,
                        'updated_at': datetime.utcnow()
                    })
                    
                    incidents[i] = updated_incident
                    await self._save_incidents_to_storage(incidents)
                    
                    logger.info("Assigned incident", incident_id=incident_id, assigned_to=user_id)
                    return updated_incident
                    
            return None
            
        except Exception as e:
            logger.error("Error assigning incident", incident_id=incident_id, error=str(e))
            return None
    
    async def add_action_to_incident(
        self,
        incident_id: str,
        action_id: str
    ) -> Optional[IncidentResponse]:
        """Add action to incident."""
        try:
            incidents = await self._load_incidents_from_storage()
            
            for i, incident in enumerate(incidents):
                if incident.id == incident_id:
                    actions = incident.actions_taken.copy()
                    if action_id not in actions:
                        actions.append(action_id)
                    
                    updated_incident = incident.model_copy(update={
                        'actions_taken': actions,
                        'updated_at': datetime.utcnow()
                    })
                    
                    incidents[i] = updated_incident
                    await self._save_incidents_to_storage(incidents)
                    
                    logger.info("Added action to incident", incident_id=incident_id, action_id=action_id)
                    return updated_incident
                    
            return None
            
        except Exception as e:
            logger.error("Error adding action to incident", incident_id=incident_id, error=str(e))
            return None
