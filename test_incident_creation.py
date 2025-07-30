#!/usr/bin/env python3
"""
Test script for creating an incident that triggers resolution with proper DAG information.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import get_logger
from src.services.enhanced_incident_service import EnhancedIncidentService
from src.models.incident_schemas import (
    IncidentCreateRequest, IncidentSeverity, IncidentCategory, 
    IncidentContext, LogEntry, IncidentMetrics
)
from src.ai import AIDecisionEngine

logger = get_logger(__name__)


async def create_test_incident_with_dag_context():
    """Create a test incident with proper DAG and task information for resolution."""
    logger.info("🧪 Creating test incident with DAG context for AI resolution")
    
    # Initialize services
    incident_service = EnhancedIncidentService()
    ai_engine = AIDecisionEngine()
    
    # Set up AI engine in incident service
    incident_service.set_ai_engine(ai_engine)
    
    try:
        # Start AI engine
        await ai_engine.start()
        
        # Load the trained model
        model_loaded = ai_engine.load_model("data/models/ai_model.pkl")
        if model_loaded:
            logger.info("✅ AI model loaded successfully")
        else:
            logger.warning("⚠️ Failed to load AI model, continuing anyway")
        
        # Create incident context with DAG information
        incident_context = IncidentContext(
            service_name="airflow-scheduler",
            environment="production",
            region="us-west-2",
            component="dag-executor",
            version="2.5.0",
            deployment_id="airflow-prod-v2.5.0",
            user_impact="Data pipeline processing delayed",
            business_impact="Daily ETL reports will be delayed by 2+ hours",
            related_incidents=[],
            tags=["airflow", "dag", "task-failure", "etl"],
            metrics=IncidentMetrics(
                cpu_usage=85.0,
                memory_usage=92.0,
                disk_usage=45.0,
                error_rate=12.5,
                response_time=5000.0,
                throughput=0.2,
                queue_depth=25,
                active_connections=150
            )
        )
        
        # Create log entries showing the DAG task failure
        log_entries = [
            LogEntry(
                timestamp=datetime.utcnow(),
                level="ERROR",
                message="Task failed in DAG 'daily_etl_pipeline' - task_id: 'extract_customer_data'",
                source="airflow-scheduler",
                stack_trace="airflow.exceptions.AirflowException: Task extract_customer_data failed with return code 1",
                metadata={
                    "dag_id": "daily_etl_pipeline",
                    "task_id": "extract_customer_data", 
                    "execution_date": "2025-07-30T00:00:00Z",
                    "try_number": 3,
                    "max_tries": 3,
                    "duration": 1800.0,
                    "operator": "BashOperator",
                    "pool": "default_pool"
                }
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level="WARN", 
                message="DAG daily_etl_pipeline has failed tasks, marking as failed",
                source="airflow-scheduler",
                stack_trace=None,
                metadata={
                    "dag_id": "daily_etl_pipeline",
                    "failed_tasks": ["extract_customer_data"],
                    "downstream_tasks_skipped": ["transform_customer_data", "load_customer_data"]
                }
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level="ERROR",
                message="Connection timeout to database server db-prod-01.company.com:5432",
                source="extract_customer_data",
                stack_trace=None,
                metadata={
                    "connection_name": "postgres_default",
                    "host": "db-prod-01.company.com",
                    "port": 5432,
                    "database": "customer_db",
                    "timeout_seconds": 30
                }
            )
        ]
        
        # Create the incident request
        incident_request = IncidentCreateRequest(
            title="Airflow DAG Task Failure - daily_etl_pipeline.extract_customer_data",
            description="""
Critical failure in the daily ETL pipeline. The extract_customer_data task in the daily_etl_pipeline DAG has failed after 3 retry attempts.

**Impact:**
- Daily customer data processing is blocked
- Downstream analytics and reporting will be delayed
- Business intelligence dashboards will show stale data

**Error Details:**
- DAG ID: daily_etl_pipeline
- Task ID: extract_customer_data  
- Execution Date: 2025-07-30T00:00:00Z
- Error: Database connection timeout to db-prod-01.company.com:5432
- Duration: 30 minutes before timeout
- Retry Attempts: 3/3 (all failed)

**Technical Context:**
- The task is trying to connect to the customer database but timing out
- This appears to be a connectivity issue rather than a query problem
- The database server may be experiencing high load or network issues
- Downstream tasks (transform_customer_data, load_customer_data) have been automatically skipped

This incident requires immediate attention as it blocks the entire daily ETL process.
            """.strip(),
            severity=IncidentSeverity.HIGH,
            category=IncidentCategory.AIRFLOW_DAG,
            context=incident_context,
            log_entries=log_entries,
            metadata={
                "dag_id": "daily_etl_pipeline",
                "task_id": "extract_customer_data",
                "execution_date": "2025-07-30T00:00:00Z",
                "failed_task_count": 1,
                "total_task_count": 3,
                "pipeline_type": "etl",
                "criticality": "high",
                "business_unit": "analytics",
                "data_sources": ["customer_db"],
                "alert_source": "airflow_monitoring"
            },
            source="airflow_monitoring_system",
            external_id="airflow-alert-20250730-001"
        )
        
        logger.info("📝 Creating incident with DAG context:")
        logger.info(f"   Title: {incident_request.title}")
        logger.info(f"   DAG ID: {incident_request.metadata.get('dag_id')}")
        logger.info(f"   Task ID: {incident_request.metadata.get('task_id')}")
        logger.info(f"   Severity: {incident_request.severity.value}")
        
        # Create the incident
        incident = await incident_service.create_incident(incident_request)
        
        logger.info(f"✅ Incident created successfully!")
        logger.info(f"   Incident ID: {incident.id}")
        logger.info(f"   Status: {incident.status.value}")
        logger.info(f"   AI Confidence: {incident.ai_confidence}")
        logger.info(f"   Predicted Resolution Time: {incident.predicted_resolution_time} minutes")
        
        # Check if AI analysis was performed
        if hasattr(incident, 'resolution') and incident.resolution:
            logger.info(f"🤖 AI Resolution Analysis:")
            logger.info(f"   Resolution Method: {incident.resolution.resolution_method}")
            logger.info(f"   Actions Planned: {len(incident.resolution.actions_taken)}")
            
            for i, action in enumerate(incident.resolution.actions_taken):
                logger.info(f"   Action {i+1}: {action.action_type} (confidence: {action.confidence_score:.2f})")
        
        # Wait a moment for any automated actions to start
        await asyncio.sleep(2)
        
        # Get updated incident status
        updated_incident = await incident_service.get_incident(incident.id)
        if updated_incident:
            logger.info(f"📊 Updated Incident Status:")
            logger.info(f"   Status: {updated_incident.status.value}")
            if updated_incident.resolution:
                logger.info(f"   Actions Executed: {len(updated_incident.resolution.actions_taken)}")
                logger.info(f"   Manual Intervention Required: {updated_incident.resolution.manual_intervention_required}")
        
        return incident
        
    except Exception as e:
        logger.error(f"❌ Error creating test incident: {str(e)}")
        return None
    
    finally:
        if ai_engine.is_running:
            await ai_engine.stop()


async def main():
    """Main test function."""
    logger.info("🚀 Starting Incident Creation Test with DAG Context")
    
    incident = await create_test_incident_with_dag_context()
    
    if incident:
        logger.info("🎉 Test completed successfully!")
        logger.info(f"📋 Summary:")
        logger.info(f"   - Incident ID: {incident.id}")
        logger.info(f"   - Title: {incident.title[:60]}...")
        logger.info(f"   - Contains DAG Info: {'dag_id' in incident.metadata}")
        logger.info(f"   - Contains Task Info: {'task_id' in incident.metadata}")
        logger.info(f"   - AI Confidence: {incident.ai_confidence}")
        logger.info("✅ This incident should trigger automated resolution based on DAG/task context")
        return 0
    else:
        logger.error("💥 Test failed to create incident")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        sys.exit(1)
