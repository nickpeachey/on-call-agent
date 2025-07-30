"""API endpoints for log polling management."""

import re
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import structlog

from ..services.log_poller import LogPoller
from ..models.schemas import AlertPattern

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/logs", tags=["Log Polling"])


class LogPollingStatus(BaseModel):
    """Log polling status response."""
    running: bool
    sources: List[Dict[str, Any]]
    alert_patterns: List[Dict[str, Any]]
    active_tasks: int


class CustomPatternRequest(BaseModel):
    """Request to add custom alert pattern."""
    name: str
    pattern: str
    severity: str = "medium"
    action_required: bool = True


@router.get("/polling/status", response_model=LogPollingStatus)
async def get_polling_status(request: Request):
    """Get current log polling status."""
    try:
        # Get log poller from app state
        log_poller: LogPoller = request.app.state.log_poller
        
        # Get real source information
        sources = []
        for source in log_poller.sources:
            sources.append({
                "name": source.name,
                "type": source.type,
                "enabled": source.enabled,
                "poll_interval": source.poll_interval,
                "config": source.config
            })
        
        # Get real alert patterns
        alert_patterns = []
        for pattern in log_poller.alert_patterns:
            alert_patterns.append({
                "name": pattern.name,
                "pattern": pattern.pattern,
                "severity": pattern.severity,
                "action_required": pattern.action_required
            })
        
        return LogPollingStatus(
            running=log_poller.running,
            sources=sources,
            alert_patterns=alert_patterns,
            active_tasks=len(log_poller.tasks)
        )
        
    except Exception as e:
        logger.error("Error getting polling status", error=str(e))
        # Return basic status if service unavailable
        return LogPollingStatus(
            running=False,
            sources=[],
            alert_patterns=[],
            active_tasks=0
        )


@router.post("/polling/start")
async def start_polling(request: Request):
    """Start log polling."""
    try:
        log_poller: LogPoller = request.app.state.log_poller
        
        if log_poller.running:
            return {"message": "Log polling is already running", "status": "already_running"}
        
        await log_poller.start_polling()
        
        return {
            "message": "Log polling started successfully",
            "status": "started",
            "sources_count": len(log_poller.sources),
            "active_tasks": len(log_poller.tasks)
        }
        
    except Exception as e:
        logger.error("Failed to start log polling", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start polling: {str(e)}")


@router.post("/polling/stop")
async def stop_polling(request: Request):
    """Stop log polling."""
    try:
        log_poller: LogPoller = request.app.state.log_poller
        
        if not log_poller.running:
            return {"message": "Log polling is not running", "status": "already_stopped"}
        
        await log_poller.stop_polling()
        
        return {
            "message": "Log polling stopped successfully",
            "status": "stopped"
        }
        
    except Exception as e:
        logger.error("Failed to stop log polling", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to stop polling: {str(e)}")


@router.post("/patterns/custom")
async def add_custom_pattern(pattern_request: CustomPatternRequest, request: Request):
    """Add a custom alert pattern."""
    try:
        log_poller: LogPoller = request.app.state.log_poller
        
        # Create new alert pattern
        new_pattern = AlertPattern(
            name=pattern_request.name,
            pattern=pattern_request.pattern,
            severity=pattern_request.severity,
            action_required=pattern_request.action_required
        )
        
        # Add to log poller
        log_poller.add_alert_pattern(new_pattern)
        
        return {
            "message": f"Custom pattern '{pattern_request.name}' added successfully",
            "pattern": {
                "name": new_pattern.name,
                "pattern": new_pattern.pattern,
                "severity": new_pattern.severity,
                "action_required": new_pattern.action_required
            },
            "total_patterns": len(log_poller.alert_patterns)
        }
        
    except Exception as e:
        logger.error("Failed to add custom pattern", pattern_name=pattern_request.name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to add pattern: {str(e)}")


@router.get("/patterns")
async def get_alert_patterns(request: Request):
    """Get all configured alert patterns."""
    try:
        log_poller: LogPoller = request.app.state.log_poller
        
        patterns = []
        for pattern in log_poller.alert_patterns:
            patterns.append({
                "name": pattern.name,
                "pattern": pattern.pattern,
                "severity": pattern.severity,
                "action_required": pattern.action_required
            })
        
        return {
            "patterns": patterns,
            "total_count": len(patterns)
        }
        
    except Exception as e:
        logger.error("Failed to get alert patterns", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get patterns: {str(e)}")


@router.post("/test-log-entry")
async def test_log_entry(log_message: str, request: Request):
    """Test a log message against all patterns."""
    try:
        log_poller: LogPoller = request.app.state.log_poller
        
        matches = []
        for pattern in log_poller.alert_patterns:
            try:
                if re.search(pattern.pattern, log_message, re.IGNORECASE):
                    matches.append({
                        "pattern_name": pattern.name,
                        "pattern": pattern.pattern,
                        "severity": pattern.severity,
                        "action_required": pattern.action_required,
                        "matched": True
                    })
            except re.error as e:
                logger.warning("Invalid regex pattern", pattern_name=pattern.name, error=str(e))
                continue
        
        return {
            "log_message": log_message,
            "matches": matches,
            "total_matches": len(matches),
            "total_patterns_tested": len(log_poller.alert_patterns)
        }
        
    except Exception as e:
        logger.error("Failed to test log entry", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to test log entry: {str(e)}")
