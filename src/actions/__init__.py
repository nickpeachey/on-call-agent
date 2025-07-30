"""Action execution engine."""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from ..core import get_logger
from ..services.actions import ActionService


logger = get_logger(__name__)


class ActionEngine:
    """Engine for executing automated actions."""
    
    def __init__(self):
        self.action_service = ActionService()
        self.is_running = False
    
    async def start(self):
        """Start the action engine."""
        if not self.is_running:
            await self.action_service.start()
            self.is_running = True
            logger.info("Action Engine started")
    
    async def stop(self):
        """Stop the action engine."""
        if self.is_running:
            await self.action_service.stop()
            self.is_running = False
            logger.info("Action Engine stopped")
    
    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        incident_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_manual: bool = False
    ) -> str:
        """Execute an action."""
        return await self.action_service.execute_action(
            action_type=action_type,
            parameters=parameters,
            incident_id=incident_id,
            user_id=user_id,
            is_manual=is_manual
        )
