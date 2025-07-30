"""
Log Archiving Service for On-Call Agent

This service captures, processes, and archives important system logs,
especially for automated actions and incident resolution tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path

from ..database import get_database
from ..core import get_logger

logger = get_logger(__name__)


@dataclass
class ArchivedLog:
    """Represents an archived log entry."""
    id: str
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    event: Optional[str] = None
    incident_id: Optional[str] = None
    action_id: Optional[str] = None
    action_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    archived_at: Optional[datetime] = None


class LogArchivingService:
    """Service for archiving and managing system logs."""
    
    def __init__(self):
        self.db = get_database()
        self.archive_path = Path(os.getenv("LOG_ARCHIVE_PATH", "/app/logs/archive"))
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # Track processed logs to avoid duplicates
        self.processed_logs = set()
        
        # Important log patterns to archive
        self.important_patterns = [
            "AIRFLOW",
            "ACTION",
            "INCIDENT",
            "RESOLUTION",
            "AUTOMATED",
            "FAILED",
            "ERROR",
            "CRITICAL"
        ]
    
    async def should_archive_log(self, log_entry: Dict[str, Any]) -> bool:
        """Determine if a log entry should be archived."""
        message = log_entry.get("message", "")
        event = log_entry.get("event", "")
        level = log_entry.get("level", "").upper()
        
        # Always archive error and critical logs
        if level in ["ERROR", "CRITICAL"]:
            return True
        
        # Archive logs with important patterns
        full_text = f"{message} {event}".upper()
        for pattern in self.important_patterns:
            if pattern in full_text:
                return True
        
        # Archive logs with action or incident IDs
        if any(key in log_entry for key in ["action_id", "incident_id", "action_type"]):
            return True
        
        return False
    
    async def archive_log(self, log_entry: Dict[str, Any]) -> Optional[ArchivedLog]:
        """Archive a single log entry."""
        try:
            # Generate unique ID for the log
            log_id = f"{log_entry.get('timestamp', datetime.utcnow().isoformat())}_{hash(str(log_entry))}"
            
            # Skip if already processed
            if log_id in self.processed_logs:
                return None
            
            # Check if should be archived
            if not await self.should_archive_log(log_entry):
                return None
            
            archived_log = ArchivedLog(
                id=log_id,
                timestamp=datetime.fromisoformat(log_entry.get("timestamp", datetime.utcnow().isoformat()).replace("Z", "+00:00")),
                level=log_entry.get("level", "INFO").upper(),
                logger_name=log_entry.get("logger", "unknown"),
                message=log_entry.get("message", ""),
                event=log_entry.get("event"),
                incident_id=log_entry.get("incident_id"),
                action_id=log_entry.get("action_id"),
                action_type=log_entry.get("action_type"),
                metadata=log_entry,
                archived_at=datetime.utcnow()
            )
            
            # Store in database
            await self._store_in_database(archived_log)
            
            # Store in file system
            await self._store_in_filesystem(archived_log)
            
            # Mark as processed
            self.processed_logs.add(log_id)
            
            logger.info("📁 LOG ARCHIVED", 
                       log_id=log_id,
                       log_level=archived_log.level,
                       log_event=archived_log.event,
                       action_type=archived_log.action_type)
            
            return archived_log
            
        except Exception as e:
            logger.error("❌ LOG ARCHIVING FAILED", 
                        error=str(e),
                        log_entry=log_entry)
            return None
    
    async def _store_in_database(self, archived_log: ArchivedLog):
        """Store archived log in database."""
        try:
            query = """
                INSERT INTO archived_logs (
                    id, timestamp, level, logger_name, message, event,
                    incident_id, action_id, action_type, metadata, archived_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (id) DO NOTHING
            """
            
            await self.db.execute(
                query,
                archived_log.id,
                archived_log.timestamp,
                archived_log.level,
                archived_log.logger_name,
                archived_log.message,
                archived_log.event,
                archived_log.incident_id,
                archived_log.action_id,
                archived_log.action_type,
                json.dumps(archived_log.metadata) if archived_log.metadata else None,
                archived_log.archived_at
            )
            
        except Exception as e:
            logger.error("❌ DATABASE ARCHIVE FAILED", 
                        log_id=archived_log.id,
                        error=str(e))
            raise
    
    async def _store_in_filesystem(self, archived_log: ArchivedLog):
        """Store archived log in filesystem."""
        try:
            # Organize by date
            date_folder = self.archive_path / archived_log.timestamp.strftime("%Y-%m-%d")
            date_folder.mkdir(exist_ok=True)
            
            # Separate files by log level
            level_file = date_folder / f"{archived_log.level.lower()}.jsonl"
            
            # Append to daily log file
            with open(level_file, "a") as f:
                f.write(json.dumps(asdict(archived_log), default=str) + "\n")
                
        except Exception as e:
            logger.error("❌ FILESYSTEM ARCHIVE FAILED", 
                        log_id=archived_log.id,
                        error=str(e))
            # Don't raise - database storage is more important
    
    async def get_archived_logs(self, 
                               incident_id: Optional[str] = None,
                               action_id: Optional[str] = None,
                               action_type: Optional[str] = None,
                               level: Optional[str] = None,
                               since: Optional[datetime] = None,
                               limit: int = 100) -> List[ArchivedLog]:
        """Retrieve archived logs with filters."""
        try:
            conditions = []
            params = []
            param_count = 0
            
            if incident_id:
                param_count += 1
                conditions.append(f"incident_id = ${param_count}")
                params.append(incident_id)
            
            if action_id:
                param_count += 1
                conditions.append(f"action_id = ${param_count}")
                params.append(action_id)
            
            if action_type:
                param_count += 1
                conditions.append(f"action_type = ${param_count}")
                params.append(action_type)
            
            if level:
                param_count += 1
                conditions.append(f"level = ${param_count}")
                params.append(level.upper())
            
            if since:
                param_count += 1
                conditions.append(f"timestamp >= ${param_count}")
                params.append(since)
            
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            param_count += 1
            
            query = f"""
                SELECT id, timestamp, level, logger_name, message, event,
                       incident_id, action_id, action_type, metadata, archived_at
                FROM archived_logs
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ${param_count}
            """
            
            params.append(limit)
            
            rows = await self.db.fetch(query, *params)
            
            archived_logs = []
            for row in rows:
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
                archived_logs.append(ArchivedLog(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    level=row["level"],
                    logger_name=row["logger_name"],
                    message=row["message"],
                    event=row["event"],
                    incident_id=row["incident_id"],
                    action_id=row["action_id"],
                    action_type=row["action_type"],
                    metadata=metadata,
                    archived_at=row["archived_at"]
                ))
            
            return archived_logs
            
        except Exception as e:
            logger.error("❌ ARCHIVE RETRIEVAL FAILED", error=str(e))
            return []
    
    async def get_airflow_action_logs(self, action_id: Optional[str] = None, 
                                    since: Optional[datetime] = None) -> List[ArchivedLog]:
        """Get logs specifically related to Airflow actions."""
        return await self.get_archived_logs(
            action_id=action_id,
            action_type="restart_airflow_dag",
            since=since or datetime.utcnow() - timedelta(hours=24)
        )
    
    async def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old archived logs."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            query = "DELETE FROM archived_logs WHERE archived_at < $1"
            result = await self.db.execute(query, cutoff_date)
            
            logger.info("🧹 OLD LOGS CLEANED UP", 
                       cutoff_date=cutoff_date.isoformat(),
                       days_kept=days_to_keep)
            
            # Also clean up old filesystem files
            for date_folder in self.archive_path.iterdir():
                if date_folder.is_dir():
                    try:
                        folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
                        if folder_date < cutoff_date:
                            import shutil
                            shutil.rmtree(date_folder)
                            logger.info("🗑️  REMOVED OLD LOG FOLDER", folder=date_folder.name)
                    except ValueError:
                        # Not a date folder, skip
                        continue
                        
        except Exception as e:
            logger.error("❌ LOG CLEANUP FAILED", error=str(e))
    
    async def initialize_database_table(self):
        """Initialize the archived_logs table if it doesn't exist."""
        try:
            query = """
                CREATE TABLE IF NOT EXISTS archived_logs (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    level TEXT NOT NULL,
                    logger_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    event TEXT,
                    incident_id TEXT,
                    action_id TEXT,
                    action_type TEXT,
                    metadata JSONB,
                    archived_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    -- Indexes for common queries
                    INDEX idx_archived_logs_timestamp ON archived_logs(timestamp),
                    INDEX idx_archived_logs_incident_id ON archived_logs(incident_id),
                    INDEX idx_archived_logs_action_id ON archived_logs(action_id),
                    INDEX idx_archived_logs_action_type ON archived_logs(action_type),
                    INDEX idx_archived_logs_level ON archived_logs(level)
                )
            """
            
            await self.db.execute(query)
            logger.info("📋 ARCHIVED LOGS TABLE INITIALIZED")
            
        except Exception as e:
            logger.error("❌ ARCHIVE TABLE INITIALIZATION FAILED", error=str(e))
            raise


# Global instance
_log_archiving_service = None


def get_log_archiving_service() -> LogArchivingService:
    """Get the global log archiving service instance."""
    global _log_archiving_service
    if _log_archiving_service is None:
        _log_archiving_service = LogArchivingService()
    return _log_archiving_service
