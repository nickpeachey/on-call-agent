"""Log monitoring service."""

import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import re

from ..core import get_logger, settings
from ..models.schemas import LogEntry, IncidentCreate


logger = get_logger(__name__)


class LogMonitorService:
    """Service for monitoring logs and detecting issues."""
    
    def __init__(self, ai_engine=None):
        self.is_running = False
        self.monitor_task = None
        self.log_sources = []
        self.alert_patterns = self._load_alert_patterns()
        self.ai_engine = ai_engine  # Injected AI engine
    
    async def start(self):
        """Start log monitoring."""
        if not self.is_running:
            self.is_running = True
            self.monitor_task = asyncio.create_task(self._monitor_logs())
            logger.info("Log monitoring started")
    
    async def stop(self):
        """Stop log monitoring."""
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
            logger.info("Log monitoring stopped")
    
    def _load_alert_patterns(self) -> List[Dict[str, Any]]:
        """Load alert patterns for log analysis."""
        return [
            {
                "name": "Database Connection Error",
                "pattern": r"(?i)(connection.*timeout|could not connect|psycopg2\.OperationalError)",
                "severity": "high",
                "service": "database",
                "tags": ["database", "connection", "timeout"]
            },
            {
                "name": "Out of Memory Error",
                "pattern": r"(?i)(out of memory|java\.lang\.OutOfMemoryError|oom|memory.*exhausted)",
                "severity": "high",
                "service": "application",
                "tags": ["memory", "oom", "performance"]
            },
            {
                "name": "Airflow Task Failure",
                "pattern": r"(?i)(task.*failed|dag.*failed|airflow.*error)",
                "severity": "high",  # Increased to high to trigger automated resolution
                "service": "airflow",
                "tags": ["airflow", "task", "failure"]
            },
            {
                "name": "Spark Application Failure",
                "pattern": r"(?i)(spark.*failed|executor.*lost|yarn.*killed)",
                "severity": "high",  # Increased to high to trigger automated resolution
                "service": "spark",
                "tags": ["spark", "executor", "failure"]
            },
            {
                "name": "HTTP 5xx Errors",
                "pattern": r"HTTP\/\d\.\d\"\s+5\d\d",
                "severity": "medium",
                "service": "api",
                "tags": ["http", "api", "server-error"]
            },
            {
                "name": "File Not Found",
                "pattern": r"(?i)(file.*not.*found|no such file|filenotfound)",
                "severity": "low",
                "service": "filesystem",
                "tags": ["file", "missing", "filesystem"]
            },
            {
                "name": "Authentication Failure",
                "pattern": r"(?i)(authentication.*failed|unauthorized|access.*denied|invalid.*credentials)",
                "severity": "medium",
                "service": "auth",
                "tags": ["auth", "security", "access"]
            }
        ]
    
    async def _monitor_logs(self):
        """Main log monitoring loop."""
        logger.info("Starting log monitoring loop")
        
        while self.is_running:
            try:
                # Monitor each configured log source
                for source in settings.elasticsearch_urls:
                    await self._monitor_elasticsearch(source)
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                logger.info("Log monitoring cancelled")
                break
            except Exception as e:
                logger.error("Error in log monitoring loop", error=str(e))
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _monitor_elasticsearch(self, elasticsearch_url: str):
        """Monitor Elasticsearch logs."""
        logger.debug("Monitoring Elasticsearch", url=elasticsearch_url)
        
        try:
            # Real implementation using aiohttp to query Elasticsearch
            import aiohttp
            
            # Query recent logs from Elasticsearch
            query = {
                "query": {
                    "range": {
                        "@timestamp": {
                            "gte": "now-5m"
                        }
                    }
                },
                "sort": [{"@timestamp": {"order": "desc"}}],
                "size": 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{elasticsearch_url}/_search",
                    json=query,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for hit in data.get('hits', {}).get('hits', []):
                            source = hit['_source']
                            
                            log_entry = LogEntry(
                                timestamp=datetime.fromisoformat(source.get('@timestamp', datetime.utcnow().isoformat()).replace('Z', '+00:00')),
                                level=source.get('level', 'INFO'),
                                service=source.get('service', 'unknown'),
                                message=source.get('message', ''),
                                metadata=source.get('metadata', {})
                            )
                            
                            await self._analyze_log_entry(log_entry)
                    else:
                        logger.warning(f"Elasticsearch query failed with status {response.status}")
                        
        except Exception as e:
            logger.error("Error monitoring Elasticsearch", url=elasticsearch_url, error=str(e))
            # Fallback to local log monitoring if Elasticsearch is unavailable
            await self._monitor_local_logs()
    
    async def _monitor_local_logs(self):
        """Monitor local log files as fallback when Elasticsearch is unavailable."""
        log_paths = [
            "/var/log/application.log",
            "/var/log/error.log", 
            "/tmp/application.log",
            "logs/application.log"
        ]
        
        for log_path in log_paths:
            try:
                if os.path.exists(log_path):
                    # Read recent lines from log file
                    with open(log_path, 'r') as f:
                        lines = f.readlines()[-20:]  # Get last 20 lines
                        
                    for line in lines:
                        if line.strip():
                            log_entry = self._parse_log_line(line.strip(), log_path)
                            if log_entry:
                                await self._analyze_log_entry(log_entry)
            except Exception as e:
                logger.debug(f"Could not read log file {log_path}: {str(e)}")
    
    def _parse_log_line(self, line: str, log_path: str) -> Optional[LogEntry]:
        """Parse a log line into a LogEntry."""
        try:
            # Basic log parsing - adjust patterns based on your log format
            # Common patterns: timestamp level service message
            import re
            
            # Try to extract timestamp, level, and message
            patterns = [
                r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\s]*)\s+(\w+)\s+(.+)',
                r'(\w+)\s+(.+)'  # Fallback: level and message only
            ]
            
            timestamp = datetime.utcnow()
            level = "INFO"
            message = line
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    if len(match.groups()) >= 3:
                        timestamp_str, level, message = match.groups()
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                        except:
                            timestamp = datetime.utcnow()
                    elif len(match.groups()) >= 2:
                        level, message = match.groups()
                    break
            
            service = os.path.basename(log_path).replace('.log', '')
            
            return LogEntry(
                timestamp=timestamp,
                level=level.upper(),
                service=service,
                message=message,
                metadata={"log_file": log_path}
            )
            
        except Exception:
            return None
    
    async def _analyze_log_entry(self, log_entry: LogEntry):
        """Analyze a log entry for potential issues."""
        logger.debug("Analyzing log entry", service=log_entry.service, level=log_entry.level)
        
        # Check against alert patterns
        for pattern_config in self.alert_patterns:
            pattern = pattern_config["pattern"]
            
            if re.search(pattern, log_entry.message):
                logger.info(
                    "Alert pattern matched",
                    pattern_name=pattern_config["name"],
                    service=log_entry.service,
                    message=log_entry.message[:100] + "..." if len(log_entry.message) > 100 else log_entry.message
                )
                
                # Create incident if severity is high enough
                if pattern_config["severity"] in ["high", "critical"]:
                    await self._create_incident_from_log(log_entry, pattern_config)
                
                break
    
    async def _create_incident_from_log(self, log_entry: LogEntry, pattern_config: Dict[str, Any]):
        """Create an incident from a log entry that matches an alert pattern."""
        try:
            incident_data = IncidentCreate(
                title=f"{pattern_config['name']} detected in {log_entry.service}",
                description=f"Alert pattern '{pattern_config['name']}' detected in service {log_entry.service}.\n\nLog message: {log_entry.message}\n\nTimestamp: {log_entry.timestamp}",
                severity=pattern_config["severity"],
                service=log_entry.service,
                tags=pattern_config["tags"] + [f"auto-detected", f"source:{log_entry.service}"]
            )
            
            logger.info(
                "🚨 INCIDENT DETECTED - Analyzing for automated resolution",
                title=incident_data.title,
                severity=incident_data.severity,
                service=incident_data.service,
                pattern=pattern_config['name']
            )
            
            # Queue incident for AI analysis and potential automated resolution
            if self.ai_engine:
                await self.ai_engine.queue_incident(incident_data)
                if not settings.quiet_mode:
                    logger.info("✅ Incident queued to server AI engine for automated resolution")
            else:
                logger.warning("⚠️ No AI engine available - incident created but not queued for automation")
            
        except Exception as e:
            logger.error("Error creating incident from log", error=str(e))
    
    async def get_recent_logs(
        self,
        service: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[LogEntry]:
        """Get recent logs with optional filtering."""
        logger.info("Getting recent logs", service=service, level=level, limit=limit)
        
        if since is None:
            since = datetime.utcnow() - timedelta(hours=1)
        
        logs = []
        
        try:
            # Try to get logs from Elasticsearch first
            logs = await self._get_logs_from_elasticsearch(service, level, limit, since)
        except Exception as e:
            logger.debug(f"Elasticsearch not available: {str(e)}")
            
        if not logs:
            # Fallback to local log files
            logs = await self._get_logs_from_files(service, level, limit, since)
        
        return logs
    
    async def _get_logs_from_elasticsearch(
        self,
        service: Optional[str],
        level: Optional[str], 
        limit: int,
        since: datetime
    ) -> List[LogEntry]:
        """Get logs from Elasticsearch."""
        import aiohttp
        
        # Build Elasticsearch query
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": since.isoformat()}}}
                    ]
                }
            },
            "sort": [{"@timestamp": {"order": "desc"}}],
            "size": limit
        }
        
        if service:
            query["query"]["bool"]["must"].append({"term": {"service": service}})
        if level:
            query["query"]["bool"]["must"].append({"term": {"level": level}})
        
        elasticsearch_url = "http://localhost:9200"  # Default ES URL
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{elasticsearch_url}/_search",
                json=query,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logs = []
                    
                    for hit in data.get('hits', {}).get('hits', []):
                        source = hit['_source']
                        log_entry = LogEntry(
                            timestamp=datetime.fromisoformat(source.get('@timestamp', datetime.utcnow().isoformat()).replace('Z', '+00:00')),
                            level=source.get('level', 'INFO'),
                            service=source.get('service', 'unknown'),
                            message=source.get('message', ''),
                            metadata=source.get('metadata', {})
                        )
                        logs.append(log_entry)
                    
                    return logs
        
        return []
    
    async def _get_logs_from_files(
        self,
        service: Optional[str],
        level: Optional[str],
        limit: int, 
        since: datetime
    ) -> List[LogEntry]:
        """Get logs from local log files."""
        logs = []
        log_paths = [
            "/var/log/application.log",
            "/var/log/error.log",
            "/tmp/application.log", 
            "logs/application.log"
        ]
        
        for log_path in log_paths:
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()[-100:]  # Get last 100 lines
                    
                    for line in lines:
                        if line.strip():
                            log_entry = self._parse_log_line(line.strip(), log_path)
                            if log_entry and log_entry.timestamp >= since:
                                # Apply filters
                                if service and log_entry.service != service:
                                    continue
                                if level and log_entry.level != level:
                                    continue
                                logs.append(log_entry)
                                
                except Exception as e:
                    logger.debug(f"Could not read log file {log_path}: {str(e)}")
        
        # Sort by timestamp and apply limit
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[:limit]
    
    async def get_log_statistics(
        self,
        service: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get log statistics for the specified time period."""
        logger.info("Getting log statistics", service=service, hours=hours)
        
        # TODO: Implement actual statistics calculation
        # For now, return mock statistics
        
        return {
            "total_logs": 12543,
            "error_count": 45,
            "warning_count": 128,
            "info_count": 11234,
            "debug_count": 1136,
            "error_rate": 0.36,
            "top_services": [
                {"service": "web-api", "count": 4521},
                {"service": "etl-pipeline", "count": 3421},
                {"service": "auth-service", "count": 2341},
                {"service": "data-processor", "count": 2260}
            ],
            "time_period": f"Last {hours} hours",
            "timestamp": datetime.utcnow()
        }
