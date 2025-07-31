"""Main application entry point."""

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core import settings, setup_logging, get_logger
from src.api import create_api_router
from src.monitoring import LogMonitorService
from src.ai import AIDecisionEngine
from src.ai.simple_engine import SimpleAIEngine
from src.actions import ActionEngine
from src.database import init_database
from src.services.incidents import IncidentService
from src.services.knowledge_base import KnowledgeBaseService
from src.services.enhanced_incident_service import EnhancedIncidentService
from src.services.log_poller import LogPoller


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    logger.info("Starting AI On-Call Agent...")
    
    # Initialize database
    await init_database()
    
    # Initialize core services
    ai_engine = AIDecisionEngine()
    simple_ai_engine = SimpleAIEngine()
    log_monitor = LogMonitorService(ai_engine=ai_engine)
    action_engine = ActionEngine()
    
    # Initialize service dependencies for log poller
    incident_service = IncidentService()
    knowledge_service = KnowledgeBaseService()
    enhanced_incident_service = EnhancedIncidentService()
    enhanced_incident_service.set_ai_engine(ai_engine)
    log_poller = LogPoller(incident_service, knowledge_service)
    
    # Inject simple AI engine into API router
    from src.api import simple_ai_training
    simple_ai_training.set_ai_engine(simple_ai_engine)
    
    # Start background services
    await log_monitor.start()
    await ai_engine.start()
    await simple_ai_engine.start()
    await action_engine.start()
    await log_poller.start_polling()
    
    # Store services in app state
    app.state.log_monitor = log_monitor
    app.state.ai_engine = ai_engine
    app.state.simple_ai_engine = simple_ai_engine
    app.state.action_engine = action_engine
    app.state.log_poller = log_poller
    app.state.incident_service = incident_service
    app.state.knowledge_service = knowledge_service
    app.state.enhanced_incident_service = enhanced_incident_service
    
    logger.info("AI On-Call Agent started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI On-Call Agent...")
    await log_poller.stop_polling()
    await action_engine.stop()
    await simple_ai_engine.stop()
    await ai_engine.stop()
    await log_monitor.stop()
    logger.info("AI On-Call Agent stopped")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="AI On-Call Agent",
        description="Intelligent automation system for ETL infrastructure monitoring",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(create_api_router(), prefix="/api/v1")
    
    return app


app = create_app()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI On-Call Agent",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "log_monitor": getattr(app.state, "log_monitor", None) is not None,
            "ai_engine": getattr(app.state, "ai_engine", None) is not None,
            "action_engine": getattr(app.state, "action_engine", None) is not None,
        }
    }


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--dev":
        # Development mode
        uvicorn.run(
            "src.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=True,
            log_level=settings.log_level.lower(),
        )
    else:
        # Production mode
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            log_level=settings.log_level.lower(),
        )


if __name__ == "__main__":
    setup_logging()
    main()
