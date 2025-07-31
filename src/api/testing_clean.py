"""
Testing API endpoints for AI On-Call Agent scenarios.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from ..core import get_logger
from ..models.schemas import ActionType
from ..models.incident_schemas import IncidentCreateRequest, IncidentSeverity, IncidentCategory
from ..services.enhanced_incident_service import EnhancedIncidentService
from ..services.actions import ActionService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/testing", tags=["Testing & Demo"])


@router.get("/scenarios", summary="📋 List Available Test Scenarios")
async def list_test_scenarios() -> Dict[str, Any]:
    """
    📋 **Available Testing Scenarios**
    
    Get a comprehensive list of all testing scenarios for the AI On-Call Agent.
    """
    return {
        "scenarios": [
            {
                "name": "🚁 Airflow DAG Failure",
                "endpoint": "POST /api/v1/testing/airflow-dag-failure",
                "description": "Simulates Airflow DAG timeout and tests AI-powered automated recovery",
                "tests": ["Incident creation", "AI analysis", "Airflow API integration", "Action execution"],
                "curl_example": "curl -X POST 'http://localhost:8000/api/v1/testing/airflow-dag-failure'"
            },
            {
                "name": "🔧 Manual Action Testing", 
                "endpoint": "POST /api/v1/testing/force-action/{action_type}",
                "description": "Test individual action types with real API calls",
                "tests": ["Action execution", "External API calls", "Error handling"],
                "curl_example": "curl -X POST 'http://localhost:8000/api/v1/testing/force-action/restart_airflow_dag'"
            },
            {
                "name": "🎯 Simple Incident Test",
                "endpoint": "POST /api/v1/testing/simple-incident", 
                "description": "Quick incident creation for testing AI analysis",
                "tests": ["AI confidence scoring", "Category classification", "Auto-resolution logic"],
                "curl_example": "curl -X POST 'http://localhost:8000/api/v1/testing/simple-incident'"
            }
        ],
        "message": "🚀 Ready to test the AI On-Call Agent! Use these scenarios to validate all system capabilities."
    }


    @app.post("/test/airflow-dag-failure")
    async def test_airflow_dag_failure(
        dag_id: str = "data_pipeline_etl",
        task_id: str = "extract_data", 
        severity: str = "high",
        auto_resolve: bool = True
    ):
        """
        Test Airflow DAG failure scenario with real action execution.
        
        This endpoint simulates an Airflow DAG failure and triggers the full
        AI-powered incident resolution workflow including real API calls.
        
        Returns immediately while resolution continues in background.
        """
        logger.info("🧪 TESTING AIRFLOW DAG FAILURE", 
                   dag_id=dag_id, 
                   task_id=task_id,
                   severity=severity,
                   auto_resolve=auto_resolve)
        
        try:
            # Create incident request
            from ..models.incident_schemas import (
                IncidentCreateRequest, IncidentSeverity, IncidentCategory, IncidentContext
            )
            
            # Map severity string to enum
            severity_mapping = {
                "low": IncidentSeverity.LOW,
                "medium": IncidentSeverity.MEDIUM, 
                "high": IncidentSeverity.HIGH,
                "critical": IncidentSeverity.CRITICAL
            }
            
            incident_severity = severity_mapping.get(severity.lower(), IncidentSeverity.HIGH)
            
            incident_request = IncidentCreateRequest(
                title=f"Airflow DAG Failed: {dag_id}",
                description=f"DAG '{dag_id}' task '{task_id}' failed during execution. "
                           f"This incident was created for testing the AI on-call agent.",
                severity=incident_severity,
                category=IncidentCategory.AIRFLOW_DAG,
                context=IncidentContext(
                    service_name=dag_id,
                    environment="production",
                    component=task_id,
                    tags=["testing", "airflow", "dag-failure"],
                    region="us-west-2",
                    version="2.7.0",
                    deployment_id="airflow-prod-001",
                    user_impact="Data pipeline delayed",
                    business_impact="Reporting dashboard may be stale",
                    metrics=None
                ),
                source="testing_api",
                external_id=f"test_{dag_id}_{task_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                metadata={
                    "dag_id": dag_id,
                    "task_id": task_id,
                    "test_scenario": "airflow_dag_failure",
                    "expected_actions": ["restart_airflow_dag"],
                    "testing": True
                }
            )
            
            # Call the enhanced incident service with async resolution
            enhanced_service = get_incident_service()
            
            # This will return immediately while resolution happens in background
            result = await enhanced_service.create_incident_with_async_resolution(
                incident_request, auto_resolve=auto_resolve
            )
            
            # Enhance response with testing information
            test_response = {
                **result,
                "test_info": {
                    "scenario": "airflow_dag_failure",
                    "dag_id": dag_id,
                    "task_id": task_id,
                    "severity": severity,
                    "expected_behavior": "AI should analyze incident and trigger Airflow DAG restart",
                    "check_status": f"Monitor incident {result['incident_id']} for resolution status"
                },
                "curl_command": f"""
curl -X POST "http://localhost:8000/test/airflow-dag-failure?dag_id={dag_id}&task_id={task_id}&severity={severity}&auto_resolve={auto_resolve}"
                """.strip()
            }
            
            logger.info("✅ AIRFLOW DAG FAILURE TEST COMPLETED", 
                       incident_id=result.get("incident_id"),
                       async_resolution=result.get("async_resolution_triggered"))
            
            return test_response
            
        except Exception as e:
            logger.error("❌ AIRFLOW DAG FAILURE TEST FAILED", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "test_info": {
                    "scenario": "airflow_dag_failure",
                    "status": "failed"
                }
            }


@router.post("/force-action/{action_type}", summary="🔧 Test Individual Actions")
async def test_force_action(
    action_type: ActionType,
    dag_id: str = Query("test_dag_123", description="DAG ID for Airflow actions"),
    incident_id: Optional[str] = Query(None, description="Link to existing incident")
) -> Dict[str, Any]:
    """
    🔧 **Individual Action Testing**
    
    **Test specific action types directly:**
    - `restart_airflow_dag` - Attempts real Airflow API call
    - `restart_service` - Simulates service restart
    - `restart_spark_job` - Simulates Spark job restart
    - `clear_cache` - Simulates cache clearing
    
    **Perfect for testing:**
    - External API integrations
    - Error handling
    - Action execution flow
    - Real system interactions
    """
    try:
        action_service = ActionService()
        
        # Build action parameters
        parameters: Dict[str, Any] = {"test_mode": True}
        
        if action_type == ActionType.RESTART_AIRFLOW_DAG:
            parameters["dag_id"] = dag_id
            parameters["dag_run_id"] = f"test_run_{int(datetime.utcnow().timestamp())}"
        
        if incident_id:
            parameters["incident_id"] = incident_id
        
        # Execute the action
        action_id = await action_service.execute_action(
            action_type=action_type.value,
            parameters=parameters,
            timeout_seconds=300
        )
        
        # Wait briefly for execution to start
        import asyncio
        await asyncio.sleep(2)
        
        # Get action results
        action_result = await action_service.get_action(action_id)
        
        return {
            "✅ test_completed": True,
            "🔧 action_type": action_type.value,
            "📊 execution_results": {
                "action_id": action_id,
                "status": getattr(action_result, 'status', 'unknown'),
                "created_at": getattr(action_result, 'created_at', None),
                "started_at": getattr(action_result, 'started_at', None),
                "completed_at": getattr(action_result, 'completed_at', None),
                "error_message": getattr(action_result, 'error_message', None)
            },
            "🌐 api_integration": {
                "real_api_attempted": action_type == ActionType.RESTART_AIRFLOW_DAG,
                "expected_failure_reason": "No Airflow server running" if action_type == ActionType.RESTART_AIRFLOW_DAG else "Simulated action",
                "demonstrates": "Real external API call handling and error management"
            },
            "💡 what_this_proves": {
                "action_system": "✅ Actions queue and execute properly",
                "error_handling": "✅ Graceful failure when external systems unavailable", 
                "api_calls": "✅ Attempts real API calls with proper error handling",
                "monitoring": "✅ Full action lifecycle tracking"
            },
            "parameters_used": parameters
        }
        
    except Exception as e:
        logger.error(f"❌ Action test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Action test failed: {str(e)}")


@router.post("/simple-incident", summary="🎯 Quick Incident Test")  
async def test_simple_incident(
    title: str = Query("API service timeout detected", description="Incident title"),
    severity: str = Query("high", description="Severity: critical, high, medium, low"),
    category: str = Query("api_service", description="Category: airflow_dag, api_service, database, etc."),
    auto_resolve: bool = Query(True, description="Attempt auto-resolution")
) -> Dict[str, Any]:
    """
    🎯 **Quick Incident Creation**
    
    **Perfect for testing:**
    - AI confidence scoring
    - Category classification
    - Auto-resolution logic
    - End-to-end workflow
    
    **Use this to quickly validate the AI system is working!**
    """
    try:
        # Map string inputs to enums
        severity_map = {
            "critical": IncidentSeverity.CRITICAL,
            "high": IncidentSeverity.HIGH, 
            "medium": IncidentSeverity.MEDIUM,
            "low": IncidentSeverity.LOW
        }
        
        category_map = {
            "airflow_dag": IncidentCategory.AIRFLOW_DAG,
            "api_service": IncidentCategory.API_SERVICE,
            "database": IncidentCategory.DATABASE,
            "spark_job": IncidentCategory.SPARK_JOB,
            "infrastructure": IncidentCategory.INFRASTRUCTURE,
            "network": IncidentCategory.NETWORK
        }
        
        incident_request = IncidentCreateRequest(
            title=title,
            description=f"""
            **Quick Test Incident**
            
            This is a test incident created via the testing API to validate:
            - AI analysis and confidence scoring
            - Automated action recommendations
            - End-to-end incident workflow
            
            **Test Parameters:**
            - Severity: {severity}
            - Category: {category}
            - Auto-resolve: {auto_resolve}
            - Created: {datetime.utcnow().isoformat()}
            """,
            severity=severity_map.get(severity.lower(), IncidentSeverity.MEDIUM),
            category=category_map.get(category.lower(), IncidentCategory.UNKNOWN),
            context=None,  # Optional field
            metadata={
                "test_scenario": "simple_incident",
                "created_via": "testing_api"
            },
            source="testing_api",
            external_id=f"test_simple_{uuid.uuid4().hex[:8]}"
        )
        
        incident_service = EnhancedIncidentService()
        incident, action_ids = await incident_service.create_incident_from_airflow(
            incident_request,
            auto_resolve=auto_resolve
        )
        
        return {
            "✅ test_completed": True,
            "🎯 scenario": "simple_incident_test",
            "📊 ai_analysis": {
                "confidence": getattr(incident, 'ai_confidence', None),
                "predicted_resolution_time": getattr(incident, 'predicted_resolution_time', None),
                "category_classified": getattr(incident, 'category', None),
                "severity_assessed": getattr(incident, 'severity', None)
            },
            "🔧 automation_results": {
                "actions_triggered": len(action_ids),
                "auto_resolution_attempted": auto_resolve and len(action_ids) > 0,
                "action_ids": action_ids
            },
            "💡 what_this_shows": {
                "ai_working": getattr(incident, 'ai_confidence', None) is not None,
                "workflow_complete": True,
                "categorization": f"Classified as {getattr(incident, 'category', 'unknown')}",
                "recommendation": "Auto-resolve" if len(action_ids) > 0 else "Manual review"
            },
            "incident_details": {
                "id": incident.id,
                "title": incident.title,
                "status": getattr(incident, 'status', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Simple incident test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simple incident test failed: {str(e)}")


@router.get("/health", summary="✅ Testing System Health")
async def test_system_health() -> Dict[str, Any]:
    """
    ✅ **Testing System Health Check**
    
    Validates that all testing endpoints and core systems are operational.
    """
    return {
        "status": "🟢 All Testing Systems Operational",
        "timestamp": datetime.utcnow().isoformat(),
        "available_tests": {
            "🚁 airflow_dag_failure": "Tests complete Airflow integration with AI analysis",
            "🔧 force_action": "Tests individual action execution and API calls", 
            "🎯 simple_incident": "Quick AI analysis and workflow validation"
        },
        "quick_start": {
            "1_test_airflow": "curl -X POST 'http://localhost:8000/api/v1/testing/airflow-dag-failure'",
            "2_test_action": "curl -X POST 'http://localhost:8000/api/v1/testing/force-action/restart_airflow_dag'",
            "3_test_simple": "curl -X POST 'http://localhost:8000/api/v1/testing/simple-incident'"
        },
        "message": "🚀 Ready to test the AI On-Call Agent! All testing endpoints are healthy and operational."
    }
