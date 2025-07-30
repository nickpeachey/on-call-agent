"""
Knowledge Base Service - Real implementation for incident pattern matching and resolution storage.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class KnowledgeBaseEntry:
    """Knowledge base entry representing a known incident pattern and resolution."""
    id: str
    title: str
    description: str
    category: str
    tags: List[str]
    error_patterns: List[str]
    solution_steps: List[str]
    automated_actions: List[str]
    success_rate: float
    related_services: List[str]
    last_used: datetime
    created_at: datetime
    updated_at: datetime
    created_by: str


class KnowledgeBaseService:
    """Real knowledge base service for storing and retrieving incident patterns."""
    
    def __init__(self, storage_path: str = "data/knowledge_base.json"):
        self.storage_path = storage_path
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self):
        """Ensure the storage directory and file exist."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        if not os.path.exists(self.storage_path):
            # Create with default entries
            default_entries = self._get_default_entries()
            self._save_entries(default_entries)
    
    def _get_default_entries(self) -> List[Dict[str, Any]]:
        """Get default knowledge base entries."""
        return [
            {
                "id": "kb_spark_oom",
                "title": "Spark Out of Memory Error",
                "description": "Spark jobs failing due to out of memory errors in executors",
                "category": "performance",
                "tags": ["spark", "memory", "oom", "bigdata"],
                "error_patterns": ["java.lang.OutOfMemoryError", "spark.sql.execution.OutOfMemoryError", "Container killed by YARN"],
                "solution_steps": [
                    "Increase executor memory (spark.executor.memory)",
                    "Optimize data partitioning",
                    "Check for data skew",
                    "Restart Spark job with adjusted parameters"
                ],
                "automated_actions": ["restart_service", "adjust_memory_config"],
                "success_rate": 0.85,
                "related_services": ["spark", "yarn", "hdfs"],
                "last_used": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "created_by": "system"
            },
            {
                "id": "kb_airflow_timeout",
                "title": "Airflow Task Timeout",
                "description": "Airflow tasks timing out due to long-running operations",
                "category": "orchestration",
                "tags": ["airflow", "timeout", "dag", "workflow"],
                "error_patterns": ["AirflowTaskTimeout", "Task timed out", "DagRun timeout"],
                "solution_steps": [
                    "Increase task timeout configuration",
                    "Check downstream dependencies",
                    "Optimize task execution",
                    "Consider task splitting"
                ],
                "automated_actions": ["restart_task", "adjust_timeout"],
                "success_rate": 0.78,
                "related_services": ["airflow", "scheduler", "executor"],
                "last_used": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "created_by": "system"
            },
            {
                "id": "kb_db_connection",
                "title": "Database Connection Pool Exhausted",
                "description": "Database connection pool exhausted leading to connection failures",
                "category": "database",
                "tags": ["database", "connection", "pool", "timeout"],
                "error_patterns": ["connection pool exhausted", "psycopg2.OperationalError", "Connection refused"],
                "solution_steps": [
                    "Restart application to reset connection pool",
                    "Check database server status",
                    "Increase connection pool size",
                    "Optimize connection usage"
                ],
                "automated_actions": ["restart_service", "health_check"],
                "success_rate": 0.92,
                "related_services": ["postgresql", "mysql", "application"],
                "last_used": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "created_by": "system"
            }
        ]
    
    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load entries from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except Exception:
            return self._get_default_entries()
    
    def _save_entries(self, entries: List[Dict[str, Any]]):
        """Save entries to storage file."""
        with open(self.storage_path, 'w') as f:
            json.dump(entries, f, indent=2)

    def _create_entry_object(self, entry_data: Dict[str, Any]) -> KnowledgeBaseEntry:
        """Convert dictionary data to KnowledgeBaseEntry object."""
        return KnowledgeBaseEntry(
            id=entry_data.get('id', ''),
            title=entry_data.get('title', ''),
            description=entry_data.get('description', ''),
            category=entry_data.get('category', ''),
            tags=entry_data.get('tags', []),
            error_patterns=entry_data.get('error_patterns', []),
            solution_steps=entry_data.get('solution_steps', []),
            automated_actions=entry_data.get('automated_actions', []),
            success_rate=entry_data.get('success_rate', 0.0),
            related_services=entry_data.get('related_services', []),
            last_used=datetime.fromisoformat(entry_data.get('last_used', datetime.now().isoformat())),
            created_at=datetime.fromisoformat(entry_data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(entry_data.get('updated_at', datetime.now().isoformat())),
            created_by=entry_data.get('created_by', 'system')
        )
    
    async def search_similar_incidents(
        self,
        error_message: str,
        service: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 5
    ) -> List[KnowledgeBaseEntry]:
        """
        Search for similar incidents in the knowledge base.
        
        Args:
            error_message: The error message to match against
            service: Service name to filter by
            severity: Severity level to consider
            limit: Maximum number of results to return
            
        Returns:
            List of matching KnowledgeBaseEntry objects
        """
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
            
            # Check title/description matches
            if any(word in entry_data.get('title', '').lower() for word in error_lower.split()):
                score += 3
            
            if score > 0:
                # Convert to KnowledgeBaseEntry object
                entry = KnowledgeBaseEntry(
                    id=entry_data['id'],
                    title=entry_data['title'],
                    description=entry_data['description'],
                    category=entry_data['category'],
                    tags=entry_data['tags'],
                    error_patterns=entry_data['error_patterns'],
                    solution_steps=entry_data['solution_steps'],
                    automated_actions=entry_data['automated_actions'],
                    success_rate=entry_data['success_rate'],
                    related_services=entry_data['related_services'],
                    last_used=datetime.fromisoformat(entry_data['last_used']) if isinstance(entry_data['last_used'], str) else entry_data['last_used'],
                    created_at=datetime.fromisoformat(entry_data['created_at']) if isinstance(entry_data['created_at'], str) else entry_data['created_at'],
                    updated_at=datetime.fromisoformat(entry_data['updated_at']) if isinstance(entry_data['updated_at'], str) else entry_data['updated_at'],
                    created_by=entry_data['created_by']
                )
                matches.append((score, entry))
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x[0], reverse=True)
        return [entry for score, entry in matches[:limit]]
    
    async def add_entry(self, entry: KnowledgeBaseEntry) -> bool:
        """Add a new entry to the knowledge base."""
        try:
            entries = self._load_entries()
            
            # Convert entry to dict
            entry_dict = {
                'id': entry.id,
                'title': entry.title,
                'description': entry.description,
                'category': entry.category,
                'tags': entry.tags,
                'error_patterns': entry.error_patterns,
                'solution_steps': entry.solution_steps,
                'automated_actions': entry.automated_actions,
                'success_rate': entry.success_rate,
                'related_services': entry.related_services,
                'last_used': entry.last_used.isoformat(),
                'created_at': entry.created_at.isoformat(),
                'updated_at': entry.updated_at.isoformat(),
                'created_by': entry.created_by
            }
            
            entries.append(entry_dict)
            self._save_entries(entries)
            return True
        except Exception:
            return False
    
    async def update_usage(self, entry_id: str) -> bool:
        """Update the last used timestamp for an entry."""
        try:
            entries = self._load_entries()
            
            for entry in entries:
                if entry['id'] == entry_id:
                    entry['last_used'] = datetime.now().isoformat()
                    entry['updated_at'] = datetime.now().isoformat()
                    break
            
            self._save_entries(entries)
            return True
        except Exception:
            return False

    async def search_entries(
        self,
        skip: int = 0,
        limit: int = 100,
        category: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[KnowledgeBaseEntry]:
        """Search and list knowledge base entries with pagination and filtering."""
        try:
            entries_data = self._load_entries()
            
            # Filter by category if provided
            if category:
                entries_data = [e for e in entries_data if e.get('category', '').lower() == category.lower()]
            
            # Filter by search term if provided
            if search:
                search_lower = search.lower()
                filtered_entries = []
                for entry in entries_data:
                    # Search in title, description, tags, and error patterns
                    if (search_lower in entry.get('title', '').lower() or
                        search_lower in entry.get('description', '').lower() or
                        any(search_lower in tag.lower() for tag in entry.get('tags', [])) or
                        any(search_lower in pattern.lower() for pattern in entry.get('error_patterns', []))):
                        filtered_entries.append(entry)
                entries_data = filtered_entries
            
            # Apply pagination
            total_entries = len(entries_data)
            entries_data = entries_data[skip:skip + limit]
            
            # Convert to KnowledgeBaseEntry objects
            entries = []
            for entry_data in entries_data:
                entry = self._create_entry_object(entry_data)
                entries.append(entry)
            
            return entries
        except Exception as e:
            # Return empty list on error
            return []

    async def create_entry(self, entry: KnowledgeBaseEntry, created_by: str) -> KnowledgeBaseEntry:
        """Create a new knowledge base entry."""
        # Set creation metadata
        entry.created_by = created_by
        entry.created_at = datetime.now()
        entry.updated_at = datetime.now()
        entry.last_used = datetime.now()
        
        # Add to storage
        await self.add_entry(entry)
        return entry

    async def get_entry_by_id(self, entry_id: str) -> Optional[KnowledgeBaseEntry]:
        """Get a knowledge base entry by ID."""
        try:
            entries_data = self._load_entries()
            
            for entry_data in entries_data:
                if entry_data.get('id') == entry_id:
                    return self._create_entry_object(entry_data)
            
            return None
        except Exception:
            return None

    async def update_entry(self, entry_id: str, updated_entry: KnowledgeBaseEntry) -> Optional[KnowledgeBaseEntry]:
        """Update an existing knowledge base entry."""
        try:
            entries = self._load_entries()
            
            for i, entry in enumerate(entries):
                if entry['id'] == entry_id:
                    # Preserve original creation info
                    updated_entry.id = entry_id
                    updated_entry.created_at = datetime.fromisoformat(entry['created_at'])
                    updated_entry.created_by = entry['created_by']
                    updated_entry.updated_at = datetime.now()
                    
                    # Convert to dict and update
                    entries[i] = {
                        'id': updated_entry.id,
                        'title': updated_entry.title,
                        'description': updated_entry.description,
                        'category': updated_entry.category,
                        'tags': updated_entry.tags,
                        'error_patterns': updated_entry.error_patterns,
                        'solution_steps': updated_entry.solution_steps,
                        'automated_actions': updated_entry.automated_actions,
                        'success_rate': updated_entry.success_rate,
                        'related_services': updated_entry.related_services,
                        'last_used': updated_entry.last_used.isoformat(),
                        'created_at': updated_entry.created_at.isoformat(),
                        'updated_at': updated_entry.updated_at.isoformat(),
                        'created_by': updated_entry.created_by
                    }
                    
                    self._save_entries(entries)
                    return updated_entry
            
            return None
        except Exception:
            return None

    async def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge base entry."""
        try:
            entries = self._load_entries()
            original_count = len(entries)
            
            entries = [e for e in entries if e.get('id') != entry_id]
            
            if len(entries) < original_count:
                self._save_entries(entries)
                return True
            
            return False
        except Exception:
            return False
