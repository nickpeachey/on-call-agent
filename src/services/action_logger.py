"""Enhanced action logging service for detailed resolution tracking."""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..core import get_logger
from ..models.schemas import ActionStatus


logger = get_logger(__name__)


class ActionAttempt:
    """Represents a single action execution attempt with detailed logging."""
    
    def __init__(self, action_id: str, action_type: str, parameters: Dict[str, Any], 
                 incident_id: Optional[str] = None):
        self.attempt_id = str(uuid.uuid4())
        self.action_id = action_id
        self.action_type = action_type
        self.parameters = parameters
        self.incident_id = incident_id
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.status = ActionStatus.PENDING
        self.result: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None
        self.exception_details: Optional[Dict[str, Any]] = None
        self.logs: List[Dict[str, Any]] = []
        
    def log_step(self, step: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Log a step in the action execution process."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step,
            "status": status,
            "details": details or {}
        }
        self.logs.append(log_entry)
        
        # Also log to the main logger for real-time monitoring
        logger.info(
            f"Action {self.action_type} - {step}: {status}",
            action_id=self.action_id,
            attempt_id=self.attempt_id,
            incident_id=self.incident_id,
            step=step,
            status=status,
            details=details
        )
        
    def mark_success(self, result: Dict[str, Any]):
        """Mark the action attempt as successful."""
        self.completed_at = datetime.utcnow()
        self.status = ActionStatus.SUCCESS
        self.result = result
        self.log_step("completion", "success", {"result": result})
        
    def mark_failure(self, error: str, exception_details: Optional[Dict[str, Any]] = None):
        """Mark the action attempt as failed."""
        self.completed_at = datetime.utcnow()
        self.status = ActionStatus.FAILED
        self.error_message = error
        self.exception_details = exception_details
        self.log_step("completion", "failed", {
            "error": error,
            "exception": exception_details
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the action attempt to a dictionary."""
        return {
            "attempt_id": self.attempt_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "parameters": self.parameters,
            "incident_id": self.incident_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value if isinstance(self.status, ActionStatus) else self.status,
            "result": self.result,
            "error_message": self.error_message,
            "exception_details": self.exception_details,
            "logs": self.logs,
            "execution_time_seconds": (
                (self.completed_at - self.started_at).total_seconds() 
                if self.completed_at and self.started_at 
                else None
            )
        }


class ActionLogger:
    """Service for logging detailed action execution attempts and outcomes."""
    
    def __init__(self, storage_path: str = "data/action_logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_attempts: Dict[str, ActionAttempt] = {}
        
    def start_action_attempt(self, action_id: str, action_type: str, 
                           parameters: Dict[str, Any], 
                           incident_id: Optional[str] = None) -> ActionAttempt:
        """Start logging a new action attempt."""
        attempt = ActionAttempt(action_id, action_type, parameters, incident_id)
        self.current_attempts[action_id] = attempt
        
        attempt.log_step("initialization", "started", {
            "action_type": action_type,
            "parameters_keys": list(parameters.keys()),
            "incident_id": incident_id
        })
        
        logger.info(
            "🚀 ACTION ATTEMPT STARTED",
            action_type=action_type,
            action_id=action_id,
            attempt_id=attempt.attempt_id,
            incident_id=incident_id
        )
        
        return attempt
        
    def get_action_attempt(self, action_id: str) -> Optional[ActionAttempt]:
        """Get the current action attempt for an action ID."""
        return self.current_attempts.get(action_id)
        
    def complete_action_attempt(self, action_id: str, success: bool, 
                              result: Optional[Dict[str, Any]] = None,
                              error: Optional[str] = None,
                              exception_details: Optional[Dict[str, Any]] = None):
        """Complete an action attempt and persist the logs."""
        attempt = self.current_attempts.get(action_id)
        if not attempt:
            logger.warning("No action attempt found for completion", action_id=action_id)
            return
            
        if success:
            attempt.mark_success(result or {})
            execution_time_str = (
                f"{(attempt.completed_at - attempt.started_at).total_seconds():.1f}s"
                if attempt.completed_at and attempt.started_at
                else "unknown"
            )
            logger.info(
                "✅ ACTION ATTEMPT COMPLETED SUCCESSFULLY",
                action_type=attempt.action_type,
                action_id=action_id,
                attempt_id=attempt.attempt_id,
                execution_time=execution_time_str
            )
        else:
            attempt.mark_failure(error or "Unknown error", exception_details)
            execution_time_str = (
                f"{(attempt.completed_at - attempt.started_at).total_seconds():.1f}s"
                if attempt.completed_at and attempt.started_at
                else "unknown"
            )
            logger.error(
                "❌ ACTION ATTEMPT FAILED",
                action_type=attempt.action_type,
                action_id=action_id,
                attempt_id=attempt.attempt_id,
                error=error,
                execution_time=execution_time_str
            )
            
        # Persist the attempt to storage
        self._persist_attempt(attempt)
        
        # Remove from current attempts
        del self.current_attempts[action_id]
        
    def _persist_attempt(self, attempt: ActionAttempt):
        """Persist an action attempt to storage."""
        try:
            # Organize by date for easy management
            date_str = attempt.started_at.strftime("%Y-%m-%d")
            date_dir = self.storage_path / date_str
            date_dir.mkdir(exist_ok=True)
            
            # Save attempt to individual file
            attempt_file = date_dir / f"{attempt.attempt_id}.json"
            with open(attempt_file, 'w') as f:
                json.dump(attempt.to_dict(), f, indent=2, default=str)
                
            # Also append to daily summary file
            summary_file = date_dir / "daily_summary.jsonl"
            with open(summary_file, 'a') as f:
                f.write(json.dumps(attempt.to_dict(), default=str) + '\n')
                
            logger.debug(
                "Action attempt persisted",
                attempt_id=attempt.attempt_id,
                file_path=str(attempt_file)
            )
            
        except Exception as e:
            logger.error(
                "Failed to persist action attempt",
                attempt_id=attempt.attempt_id,
                error=str(e)
            )
            
    def get_action_attempts_for_incident(self, incident_id: str, 
                                       days_back: int = 7) -> List[Dict[str, Any]]:
        """Get all action attempts for a specific incident."""
        attempts = []
        
        # Search through recent files
        for i in range(days_back):
            date = datetime.utcnow().date() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            summary_file = self.storage_path / date_str / "daily_summary.jsonl"
            
            if not summary_file.exists():
                continue
                
            try:
                with open(summary_file, 'r') as f:
                    for line in f:
                        attempt_data = json.loads(line.strip())
                        if attempt_data.get("incident_id") == incident_id:
                            attempts.append(attempt_data)
            except Exception as e:
                logger.warning(
                    "Failed to read action attempt summary",
                    file=str(summary_file),
                    error=str(e)
                )
                
        # Sort by started_at timestamp
        attempts.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return attempts
        
    def get_recent_action_attempts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent action attempts from log files."""
        try:
            attempts = []
            if self.storage_path.exists():
                log_files = sorted(self.storage_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                
                for log_file in log_files:
                    if len(attempts) >= limit:
                        break
                    
                    try:
                        with open(log_file, 'r') as f:
                            log_data = json.load(f)
                            if isinstance(log_data, list):
                                attempts.extend(log_data)
                            else:
                                attempts.append(log_data)
                    except (json.JSONDecodeError, IOError) as e:
                        logger.warning(f"Could not read log file {log_file}: {e}")
                        continue
            
            return attempts[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent action attempts: {e}")
            return []
    
    def get_action_attempts_by_incident(self, incident_id: str) -> List[Dict[str, Any]]:
        """Get action attempts for a specific incident."""
        try:
            attempts = []
            if self.storage_path.exists():
                log_files = list(self.storage_path.glob("*.json"))
                
                for log_file in log_files:
                    try:
                        with open(log_file, 'r') as f:
                            log_data = json.load(f)
                            
                            # Handle both single attempts and lists of attempts
                            if isinstance(log_data, list):
                                for attempt in log_data:
                                    if attempt.get("incident_id") == incident_id:
                                        attempts.append(attempt)
                            else:
                                if log_data.get("incident_id") == incident_id:
                                    attempts.append(log_data)
                                    
                    except (json.JSONDecodeError, IOError) as e:
                        logger.warning(f"Could not read log file {log_file}: {e}")
                        continue
            
            # Sort by started_at timestamp
            attempts.sort(key=lambda x: x.get("started_at", ""), reverse=False)
            return attempts
            
        except Exception as e:
            logger.error(f"Error getting action attempts for incident {incident_id}: {e}")
            return []
        
    def get_action_statistics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get action execution statistics."""
        attempts = self.get_recent_action_attempts(limit=1000)
        
        if not attempts:
            return {
                "total_attempts": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "action_type_stats": {},
                "failure_reasons": {}
            }
            
        total_attempts = len(attempts)
        successful_attempts = [a for a in attempts if a.get("status") == "success"]
        failed_attempts = [a for a in attempts if a.get("status") == "failed"]
        
        # Calculate execution times for successful attempts
        execution_times = [
            a.get("execution_time_seconds", 0) 
            for a in successful_attempts 
            if a.get("execution_time_seconds")
        ]
        
        # Group by action type
        action_type_stats = {}
        for attempt in attempts:
            action_type = attempt.get("action_type", "unknown")
            if action_type not in action_type_stats:
                action_type_stats[action_type] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_execution_time": 0.0
                }
                
            stats = action_type_stats[action_type]
            stats["total"] += 1
            
            if attempt.get("status") == "success":
                stats["successful"] += 1
                if attempt.get("execution_time_seconds"):
                    # Update running average
                    current_avg = stats["avg_execution_time"]
                    new_time = attempt.get("execution_time_seconds")
                    stats["avg_execution_time"] = (
                        (current_avg * (stats["successful"] - 1) + new_time) / stats["successful"]
                    )
            elif attempt.get("status") == "failed":
                stats["failed"] += 1
                
        # Calculate success rates for each action type
        for stats in action_type_stats.values():
            if stats["total"] > 0:
                stats["success_rate"] = stats["successful"] / stats["total"]
            else:
                stats["success_rate"] = 0.0
                
        # Collect failure reasons
        failure_reasons = {}
        for attempt in failed_attempts:
            error = attempt.get("error_message", "Unknown error")
            # Extract the first line of error for grouping
            error_key = error.split('\n')[0][:100] if error else "Unknown error"
            failure_reasons[error_key] = failure_reasons.get(error_key, 0) + 1
            
        return {
            "total_attempts": total_attempts,
            "successful_attempts": len(successful_attempts),
            "failed_attempts": len(failed_attempts),
            "success_rate": len(successful_attempts) / total_attempts if total_attempts > 0 else 0.0,
            "average_execution_time": (
                sum(execution_times) / len(execution_times) 
                if execution_times else 0.0
            ),
            "action_type_stats": action_type_stats,
            "failure_reasons": dict(sorted(
                failure_reasons.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])  # Top 10 failure reasons
        }


# Global action logger instance
action_logger = ActionLogger()
