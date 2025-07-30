"""
Log Polling Service - Continuously monitors log sources for issues
"""
import asyncio
import re
import json
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import structlog

from ..core.config import settings
from ..models.schemas import IncidentCreate, LogEntry, AlertPattern
from .incidents import IncidentService
from .knowledge_base import KnowledgeBaseService
from .log_archive import LogArchiveService

logger = structlog.get_logger(__name__)

@dataclass
class LogSource:
    """Configuration for a log source"""
    name: str
    type: str  # 'file', 'api'
    config: Dict[str, Any]
    enabled: bool = True
    poll_interval: int = 30  # seconds
    filters: Optional[List[str]] = None
    
class LogPoller:
    """Main log polling service that monitors multiple log sources"""
    
    def __init__(
        self,
        incident_service: IncidentService,
        knowledge_service: KnowledgeBaseService
    ):
        self.incident_service = incident_service
        self.knowledge_service = knowledge_service
        self.archive_service = LogArchiveService()  # Initialize log archive service
        self.sources: List[LogSource] = []
        self.alert_patterns: List[AlertPattern] = []
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.last_read_positions: Dict[str, int] = {}  # Track file positions
        
        # Initialize default log sources from config
        self._load_log_sources()
        self._load_alert_patterns()
    
    def _load_log_sources(self):
        """Load log sources from configuration"""
        # File-based logs - production paths
        self.sources.append(LogSource(
            name="application_logs",
            type="file",
            config={
                "paths": [
                    "/var/log/application/*.log",
                    "/var/log/airflow/*.log", 
                    "/var/log/spark/*.log",
                    "./logs/*.log",  # Local development logs
                    "/tmp/test_logs/*.log"  # For testing
                ]
            },
            poll_interval=15
        ))
        
        # System logs
        self.sources.append(LogSource(
            name="system_logs",
            type="file",
            config={
                "paths": [
                    "/var/log/syslog",
                    "/var/log/messages",
                    "/var/log/daemon.log"
                ]
            },
            poll_interval=30
        ))
        
        # Docker container logs
        self.sources.append(LogSource(
            name="container_logs",
            type="file",
            config={
                "paths": [
                    "/var/lib/docker/containers/*/*.log",
                    "/var/log/docker.log"
                ]
            },
            poll_interval=20
        ))
        
        # ETL specific logs
        self.sources.append(LogSource(
            name="etl_logs",
            type="file", 
            config={
                "paths": [
                    "/var/log/etl/*.log",
                    "/var/log/data-pipeline/*.log",
                    "/opt/spark/logs/*.log",
                    "/opt/airflow/logs/**/*.log"
                ]
            },
            poll_interval=10  # More frequent for ETL systems
        ))
    
    def _load_alert_patterns(self):
        """Load alert patterns that trigger immediate action"""
        patterns = [
            {
                "name": "critical_error",
                "pattern": r"CRITICAL|FATAL|ERROR.*failed.*start",
                "severity": "critical",
                "action_required": True
            },
            {
                "name": "out_of_memory", 
                "pattern": r"OutOfMemoryError|heap.*space|memory.*exhausted",
                "severity": "high",
                "action_required": True
            },
            {
                "name": "connection_failed",
                "pattern": r"connection.*timeout|connection.*refused|connection.*failed",
                "severity": "high", 
                "action_required": True
            },
            {
                "name": "disk_full",
                "pattern": r"disk.*full|no.*space.*left|filesystem.*full",
                "severity": "critical",
                "action_required": True
            },
            {
                "name": "service_unavailable",
                "pattern": r"service.*unavailable|server.*not.*responding|health.*check.*failed",
                "severity": "high",
                "action_required": True
            },
            {
                "name": "database_error",
                "pattern": r"database.*error|sql.*exception|connection.*pool.*exhausted",
                "severity": "high",
                "action_required": True
            }
        ]
        
        self.alert_patterns = [AlertPattern(**pattern) for pattern in patterns]
    
    def add_alert_pattern(self, pattern: AlertPattern):
        """Add a new alert pattern to the monitoring system."""
        # Check if pattern with same name already exists
        existing_names = [p.name for p in self.alert_patterns]
        if pattern.name in existing_names:
            raise ValueError(f"Alert pattern with name '{pattern.name}' already exists")
        
        # Validate regex pattern
        try:
            re.compile(pattern.pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        
        self.alert_patterns.append(pattern)
        logger.info("Added new alert pattern", pattern_name=pattern.name, severity=pattern.severity)
    
    def remove_alert_pattern(self, pattern_name: str) -> bool:
        """Remove an alert pattern by name."""
        for i, pattern in enumerate(self.alert_patterns):
            if pattern.name == pattern_name:
                removed_pattern = self.alert_patterns.pop(i)
                logger.info("Removed alert pattern", pattern_name=removed_pattern.name)
                return True
        return False
    
    def add_log_source(self, source: LogSource):
        """Add a new log source for monitoring."""
        # Check if source with same name already exists
        existing_names = [s.name for s in self.sources]
        if source.name in existing_names:
            raise ValueError(f"Log source with name '{source.name}' already exists")
        
        self.sources.append(source)
        logger.info("Added new log source", source_name=source.name, source_type=source.type)
        
        # If poller is running, start monitoring this source
        if self.running and source.enabled:
            task = asyncio.create_task(self._poll_source(source))
            self.tasks.append(task)
            logger.info("Started polling new source", source_name=source.name)
    
    async def start_polling(self):
        """Start polling all configured log sources"""
        if self.running:
            logger.warning("Log poller already running")
            return
        
        self.running = True
        logger.info("Starting log poller with sources", source_count=len(self.sources))
        
        # Start a polling task for each source
        for source in self.sources:
            if source.enabled:
                task = asyncio.create_task(self._poll_source(source))
                self.tasks.append(task)
                logger.info("Started polling source", source_name=source.name)
        
        logger.info("Log poller started successfully")
    
    async def stop_polling(self):
        """Stop all polling tasks"""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping log poller...")
        
        # Cancel all polling tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()
        logger.info("Log poller stopped")
    
    async def _poll_source(self, source: LogSource):
        """Poll a specific log source continuously"""
        logger.info("Starting to poll source", source_name=source.name, source_type=source.type)
        
        while self.running:
            try:
                # Get new log entries from this source
                entries = await self._fetch_logs(source)
                
                if entries:
                    logger.debug("Fetched {} log entries from {}", len(entries), source.name)
                    
                    # Process each log entry for issues
                    for entry in entries:
                        await self._process_log_entry(entry, source.name)
                
                # Wait before next poll
                await asyncio.sleep(source.poll_interval)
                
            except asyncio.CancelledError:
                logger.info("Polling cancelled for source: {}", source.name)
                break
            except Exception as e:
                logger.error("Error polling source {}: {}", source.name, str(e))
                # Wait a bit before retrying
                await asyncio.sleep(min(source.poll_interval, 60))
    
    async def _fetch_logs(self, source: LogSource) -> List[LogEntry]:
        """Fetch new log entries from a source"""
        try:
            if source.type == "file":
                return await self._fetch_file_logs(source)
            elif source.type == "api":
                return await self._fetch_api_logs(source)
            elif source.type == "database":
                return await self._fetch_database_logs(source)
            else:
                logger.warning("Unknown source type: {}", source.type)
                return []
        except Exception as e:
            logger.error("Failed to fetch logs from {}: {}", source.name, str(e))
            return []
    
    async def _fetch_file_logs(self, source: LogSource) -> List[LogEntry]:
        """Fetch logs from log files"""
        entries = []
        
        try:
            for path_pattern in source.config["paths"]:
                files = glob.glob(path_pattern)
                
                for file_path in files:
                    try:
                        # Read recent lines from log file
                        recent_entries = await self._read_recent_log_lines(file_path, source.poll_interval)
                        entries.extend(recent_entries)
                    except Exception as e:
                        logger.error("Error reading file {}: {}", file_path, str(e))
            
            return entries
            
        except Exception as e:
            logger.error("File logs fetch error: {}", str(e))
            return []
    
    async def _read_recent_log_lines(self, file_path: str, seconds_back: int) -> List[LogEntry]:
        """Read recent log lines from a file"""
        entries = []
        cutoff_time = datetime.utcnow() - timedelta(seconds=seconds_back)
        
        try:
            # Get last read position for this file
            last_pos = self.last_read_positions.get(file_path, 0)
            
            with open(file_path, 'r') as f:
                # Seek to last position
                f.seek(last_pos)
                
                # Read new lines
                lines = f.readlines()
                
                # Update position
                self.last_read_positions[file_path] = f.tell()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse log line (adapt this to your log format)
                    entry = self._parse_log_line(line, file_path)
                    if entry:
                        entries.append(entry)
            
            return entries
            
        except FileNotFoundError:
            # File doesn't exist yet, that's ok
            return []
        except Exception as e:
            logger.error("Error reading log file {}: {}", file_path, str(e))
            return []
    
    def _parse_log_line(self, line: str, source_file: str) -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry"""
        try:
            # Common log patterns - adapt these to your log format
            patterns = [
                # Standard format: 2024-01-15 10:30:15 [ERROR] service: message
                r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s+(\w+):\s+(.+)',
                # JSON format
                r'^\{.*\}$',
                # Simple format: ERROR: message
                r'(\w+):\s+(.+)'
            ]
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    if pattern.endswith(r'^\{.*\}$'):  # JSON format
                        try:
                            data = json.loads(line)
                            timestamp_str = data.get('timestamp', datetime.utcnow().isoformat())
                            timestamp = self._parse_timestamp(timestamp_str)
                            return LogEntry(
                                timestamp=timestamp,
                                service=data.get('service', 'unknown'),
                                level=data.get('level', 'INFO'),
                                message=data.get('message', ''),
                                metadata=data
                            )
                        except json.JSONDecodeError:
                            continue
                    elif len(match.groups()) == 4:  # Standard format
                        timestamp_str, level, service, message = match.groups()
                        timestamp = self._parse_timestamp(timestamp_str)
                        return LogEntry(
                            timestamp=timestamp,
                            service=service,
                            level=level,
                            message=message,
                            metadata={"source_file": source_file}
                        )
                    elif len(match.groups()) == 2:  # Simple format
                        level, message = match.groups()
                        return LogEntry(
                            timestamp=datetime.utcnow(),
                            service="unknown",
                            level=level,
                            message=message,
                            metadata={"source_file": source_file}
                        )
            
            # If no pattern matches, create a basic entry
            return LogEntry(
                timestamp=datetime.utcnow(),
                service="unknown",
                level="INFO",
                message=line,
                metadata={"source_file": source_file, "raw_line": line}
            )
            
        except Exception as e:
            logger.error("Error parsing log line: {}", str(e))
            return None
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object"""
        try:
            # Try different timestamp formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # If no format matches, return current time
            return datetime.utcnow()
        except:
            return datetime.utcnow()
    
    async def _fetch_api_logs(self, source: LogSource) -> List[LogEntry]:
        """Fetch logs from a REST API endpoint"""
        import aiohttp
        
        entries = []
        config = source.config
        
        try:
            url = config.get("url")
            if not url:
                logger.warning("No URL configured for API source", source=source.name)
                return []
                
            async with aiohttp.ClientSession() as session:
                headers = config.get("headers", {})
                params = config.get("params", {})
                
                # Add timestamp filter for recent logs
                if "since_param" in config:
                    since_time = datetime.utcnow() - timedelta(seconds=source.poll_interval)
                    params[config["since_param"]] = since_time.isoformat()
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Parse API response based on format
                        logs_field = config.get("logs_field", "logs")
                        log_data = data.get(logs_field, [])
                        
                        for log_item in log_data:
                            entry = self._parse_api_log_item(log_item, source.name)
                            if entry:
                                entries.append(entry)
                    else:
                        logger.warning("API request failed", url=url, status=response.status)
                        
        except Exception as e:
            logger.error("Error fetching API logs", source=source.name, error=str(e))
            
        return entries
    
    async def _fetch_database_logs(self, source: LogSource) -> List[LogEntry]:
        """Fetch logs from a database query"""
        entries = []
        config = source.config
        
        try:
            # This would integrate with your database service
            # For now, implement basic structure
            db_type = config.get("db_type", "postgresql")
            
            if db_type == "postgresql":
                entries = await self._fetch_postgres_logs(config)
            elif db_type == "elasticsearch":
                entries = await self._fetch_elasticsearch_logs(config)
            else:
                logger.warning("Unsupported database type", db_type=db_type)
                
        except Exception as e:
            logger.error("Error fetching database logs", source=source.name, error=str(e))
            
        return entries
    
    async def _fetch_postgres_logs(self, config: Dict[str, Any]) -> List[LogEntry]:
        """Fetch logs from PostgreSQL database"""
        # Implementation would depend on your log table structure
        # This is a placeholder for production database integration
        return []
    
    async def _fetch_elasticsearch_logs(self, config: Dict[str, Any]) -> List[LogEntry]:
        """Fetch logs from Elasticsearch"""
        # Implementation for Elasticsearch integration
        # This would use elasticsearch-py library
        return []
    
    def _parse_api_log_item(self, log_item: Dict[str, Any], source_name: str) -> Optional[LogEntry]:
        """Parse a log item from API response"""
        try:
            return LogEntry(
                timestamp=datetime.fromisoformat(log_item.get("timestamp", datetime.utcnow().isoformat())),
                level=log_item.get("level", "INFO"),
                service=log_item.get("service", source_name),
                message=log_item.get("message", ""),
                metadata={
                    **log_item.get("metadata", {}),
                    "source": source_name
                }
            )
        except Exception as e:
            logger.warning("Failed to parse API log item", error=str(e))
            return None
    
    async def _process_log_entry(self, entry: LogEntry, source_name: str):
        """Process a single log entry for issues"""
        try:
            # Convert LogEntry to dict for archive service
            log_dict = {
                'timestamp': entry.timestamp.isoformat(),
                'message': entry.message,
                'service': entry.service,
                'level': entry.level,
                'source': source_name
            }
            
            # Check if this log entry has already been processed
            if self.archive_service.is_already_processed(log_dict):
                logger.debug("Skipping already processed log entry",
                           service=entry.service,
                           message_preview=entry.message[:50])
                return
            
            # Check if this log entry matches any alert patterns
            issues_found = []
            
            for pattern in self.alert_patterns:
                if re.search(pattern.pattern, entry.message, re.IGNORECASE):
                    issues_found.append(pattern)
                    logger.info(
                        "Alert pattern '{}' matched in log from {}: {}",
                        pattern.name, source_name, entry.message[:100]
                    )
            
            # Mark as processed regardless of whether issues were found
            incident_created = len(issues_found) > 0
            self.archive_service.mark_as_processed(log_dict, incident_created)
            
            # If critical issues found, create incident and potentially auto-resolve
            if issues_found:
                await self._handle_detected_issues(entry, issues_found, source_name)
            
        except Exception as e:
            logger.error("Error processing log entry: {}", str(e))
    
    async def _handle_detected_issues(self, entry: LogEntry, issues: List[AlertPattern], source_name: str):
        """Handle detected issues by creating incidents and potentially auto-resolving"""
        try:
            # Determine the most severe issue
            severity_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
            most_severe = max(issues, key=lambda x: severity_order.get(x.severity, 0))
            
            # Create incident
            incident = await self.incident_service.create_incident(
                title=f"{most_severe.name.replace('_', ' ').title()} - {entry.service}",
                description=f"Detected in {source_name}: {entry.message}",
                service=entry.service,
                severity=most_severe.severity
            )
            
            logger.info("Created incident {} for detected issue", incident.id)
            
            # Try to auto-resolve if we have knowledge about this pattern
            if most_severe.action_required:
                await self._attempt_auto_resolution(incident, most_severe, entry)
            
        except Exception as e:
            logger.error("Error handling detected issues: {}", str(e))
    
    async def _attempt_auto_resolution(self, incident, pattern: AlertPattern, entry: LogEntry):
        """Attempt to automatically resolve the incident using knowledge base"""
        try:
            logger.info("Attempting auto-resolution for incident {}", incident.id)
            
            # Search knowledge base for similar issues
            knowledge_entries = await self.knowledge_service.search_similar_incidents(
                error_message=f"{pattern.name} {entry.message}",
                service=entry.service,
                severity=getattr(pattern, 'severity', 'medium')
            )
            
            if knowledge_entries:
                # Get the best matching knowledge entry
                best_match = knowledge_entries[0]
                
                logger.info(
                    "Found knowledge base match: {} (confidence: {})",
                    best_match.title, best_match.success_rate
                )
                
                # Execute real action using action execution service
                if best_match.success_rate >= 0.7:
                    logger.info("Executing auto-resolution action", description=best_match.description)
                    
                    try:
                        from ..services.action_execution import ActionExecutionService
                        action_service = ActionExecutionService()
                        
                        # Convert knowledge base entry to action format
                        for action_type in best_match.automated_actions:
                            action_dict = {
                                "type": action_type,
                                "parameters": {"service": incident.service},
                                "description": f"Auto-resolution action: {action_type}"
                            }
                            
                            incident_dict = {
                                "id": str(incident.id),
                                "title": incident.title,
                                "service": incident.service,
                                "severity": incident.severity,
                                "description": incident.description
                            }
                            
                            success = await action_service.execute_action(action_dict, incident_dict)
                            
                            if success:
                                logger.info("Successfully auto-resolved incident {}", incident.id)
                                break
                            else:
                                logger.warning("Action {} failed for incident {}", action_type, incident.id)
                    except Exception as e:
                        logger.error("Failed to execute auto-resolution action: {}", str(e))
                else:
                    logger.info(
                        "Low confidence for auto-resolution of incident {} (confidence: {:.2f})",
                        incident.id, best_match.success_rate
                    )
            else:
                logger.info("No knowledge base match found for incident {}", incident.id)
                
                # Create a knowledge entry for future learning
                await self._create_knowledge_entry(pattern, entry)
                
        except Exception as e:
            logger.error("Error in auto-resolution attempt: {}", str(e))
    
    async def _create_knowledge_entry(self, pattern: AlertPattern, entry: LogEntry):
        """Create a new knowledge entry for learning"""
        try:
            from ..services.knowledge_base import KnowledgeBaseEntry
            from datetime import datetime
            import uuid
            
            # Create a proper KnowledgeBaseEntry object
            knowledge_entry = KnowledgeBaseEntry(
                id=f"kb_{uuid.uuid4().hex[:8]}",
                title=f"{pattern.name.replace('_', ' ').title()} Pattern",
                description=f"Auto-detected pattern for {pattern.name}",
                category=pattern.severity,
                tags=[pattern.name, entry.service, pattern.severity],
                error_patterns=[pattern.pattern],
                solution_steps=[f"Restart {entry.service}", "Monitor recovery"],
                automated_actions=["restart_service"],
                success_rate=0.6,  # Initial confidence
                related_services=[entry.service],
                last_used=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by="system"
            )
            
            await self.knowledge_service.add_entry(knowledge_entry)
            
            logger.info("Created new knowledge entry for pattern: {}", pattern.name)
            
        except Exception as e:
            logger.error("Error creating knowledge entry: {}", str(e))
    
    def add_custom_pattern(self, name: str, pattern: str, severity: str = "medium", action_required: bool = True):
        """Add a custom alert pattern"""
        custom_pattern = AlertPattern(
            name=name,
            pattern=pattern,
            severity=severity,
            action_required=action_required
        )
        self.alert_patterns.append(custom_pattern)
        logger.info("Added custom alert pattern: {}", name)
    
    def get_polling_status(self) -> Dict[str, Any]:
        """Get current polling status"""
        return {
            "running": self.running,
            "sources": [
                {
                    "name": source.name,
                    "type": source.type,
                    "enabled": source.enabled,
                    "poll_interval": source.poll_interval
                }
                for source in self.sources
            ],
            "alert_patterns": [
                {
                    "name": pattern.name,
                    "severity": pattern.severity,
                    "action_required": pattern.action_required
                }
                for pattern in self.alert_patterns
            ],
            "active_tasks": len(self.tasks),
            "archive_stats": self.archive_service.get_stats()
        }
    
    async def cleanup_old_archives(self, max_age_days: int = 30):
        """Clean up old archive files"""
        await self.archive_service.cleanup_old_archives(max_age_days)
        
    def refresh_archive_cache(self):
        """Force refresh the archive cache"""
        self.archive_service.force_refresh_cache()
        
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archive service statistics"""
        return self.archive_service.get_stats()
