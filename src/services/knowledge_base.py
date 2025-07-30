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
        """Save entries to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(entries, f, indent=2, default=str)
    
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
