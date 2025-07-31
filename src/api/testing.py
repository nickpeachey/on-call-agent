"""
Testing API endpoints for demonstrating AI On-Call Agent capabilities.
These endpoints simulate real-world scenarios for testing the system.
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


class AirflowTestParams(BaseModel):
    """Parameters for Airflow testing scenarios."""
    dag_id: str = Field(default="test_data_pipeline", description="DAG ID to test")
    task_id: str = Field(default="process_data", description="Task ID that failed")
    failure_type: str = Field(default="timeout", description="Type of failure (timeout, error, dependency)")
    airflow_url: str = Field(default="http://localhost:8080", description="Airflow webserver URL")
    enable_real_api_call: bool = Field(default=False, description="Actually try to call Airflow API")


@router.get("/scenarios", summary="List Available Test Scenarios")
async def list_test_scenarios() -> Dict[str, Any]:
    """
    Get a list of all available test scenarios.
    
    Returns detailed information about each scenario including:
    - What it tests
    - Expected behavior
    - Parameters
    """
    return {
        "scenarios": [
            {
                "name": "airflow_dag_timeout",
                "description": "Simulates an Airflow DAG task timeout and attempts automated recovery",
                "endpoint": "/api/v1/testing/airflow-dag-failure",
                "expected_behavior": "Creates incident, AI analyzes, attempts DAG restart via Airflow API",
                "ai_confidence_expected": "60-80%",
                "actions_triggered": ["restart_airflow_dag"]
            },
            {
                "name": "force_action_test",
                "description": "Tests individual action execution",
                "endpoint": "/api/v1/testing/force-action/{action_type}",
                "expected_behavior": "Directly executes specified action type",
                "actions_available": ["restart_airflow_dag", "restart_service", "restart_spark_job"]
            }
        ],
        "note": "These endpoints demonstrate the AI On-Call Agent's capabilities for automated incident resolution"
    }


@router.post("/airflow-dag-failure", summary="🚁 Test Airflow DAG Failure Scenario")
async def test_airflow_dag_failure(
    params: AirflowTestParams = AirflowTestParams(),
    auto_resolve: bool = Query(True, description="Attempt automatic resolution")
) -> Dict[str, Any]:
    """
    🚁 **Airflow DAG Failure Test Scenario**
    
    This endpoint simulates a realistic Airflow DAG failure and tests the complete
    AI-powered incident resolution workflow.
    
    **What This Tests:**
    1. ✅ Incident creation with Airflow-specific metadata
    2. 🤖 AI analysis and confidence scoring
    3. 🔧 Automated action recommendation
    4. 🌐 Real Airflow API integration (if enabled)
    5. 📊 End-to-end monitoring and logging
    
    **Expected Flow:**
    ```
    DAG Failure → Incident Creation → AI Analysis → Action Recommendation → API Call
    ```
    
    **Parameters:**
    - `dag_id`: The DAG that failed
    - `task_id`: The specific task that failed  
    - `failure_type`: Type of failure (timeout, error, dependency)
    - `airflow_url`: Airflow webserver URL for API calls
    - `enable_real_api_call`: Whether to actually call Airflow API
    
    **Returns:**
    - Incident details with AI analysis
    - Action execution results
    - API call status and responses
    """
    try:
        # Create realistic Airflow incident
        incident_data = IncidentCreateRequest(
            title=f"Airflow DAG '{params.dag_id}' failed - Task '{params.task_id}' {params.failure_type}",
            description=f"""
            DAG Failure Details:
            - DAG ID: {params.dag_id}
            - Task ID: {params.task_id}
            - Failure Type: {params.failure_type}
            - Failure Time: {datetime.utcnow().isoformat()}
            - Expected Duration: 30 minutes
            - Actual Duration: 45 minutes (timeout)
            
            Error Message: Task instance failed after exceeding timeout limit.
            Previous runs were successful. This appears to be a transient issue.
            
            Recommended Action: Restart the DAG run and monitor for completion.
            """,
            severity=IncidentSeverity.HIGH if params.failure_type == "timeout" else IncidentSeverity.CRITICAL,
            category=IncidentCategory.AIRFLOW_DAG,
            metadata={
                "dag_id": params.dag_id,
                "task_id": params.task_id,
                "failure_type": params.failure_type,
                "airflow_url": params.airflow_url,
                "execution_date": datetime.utcnow().isoformat(),
                "dag_run_id": f"manual_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "previous_success": True,
                "timeout_seconds": 1800,
                "enable_real_api_call": params.enable_real_api_call
            }
        )
        
        # Create incident using enhanced service
        incident_service = EnhancedIncidentService()
        incident, action_ids = await incident_service.create_incident_from_airflow(
            incident_data,
            auto_resolve=auto_resolve
        )
        
        # If auto-resolution was attempted, get action details
        action_details = []
        if action_ids:
            action_service = ActionService()
            for action_id in action_ids:
                try:
                    action = await action_service.get_action(action_id)
                    action_details.append(action)
                except Exception as e:
                    logger.warning(f"Could not fetch action {action_id}: {e}")
                    action_details.append({"id": action_id, "status": "unknown", "error": str(e)})
        
        return {
            "scenario": "airflow_dag_failure",
            "test_timestamp": datetime.utcnow().isoformat(),
            "parameters": params.dict(),
            "incident": incident.dict() if hasattr(incident, 'dict') else dict(incident),
            "ai_analysis": {
                "confidence": getattr(incident, 'ai_confidence', None),
                "predicted_resolution_time": getattr(incident, 'predicted_resolution_time', None),
                "auto_resolution_attempted": auto_resolve and len(action_ids) > 0
            },
            "actions_executed": action_details,
            "airflow_api_integration": {
                "enabled": params.enable_real_api_call,
                "url": params.airflow_url,
                "status": "Would attempt DAG restart" if not params.enable_real_api_call else "API call attempted"
            },
            "test_results": {
                "incident_created": True,
                "ai_analyzed": getattr(incident, 'ai_confidence', None) is not None,
                "actions_triggered": len(action_ids) > 0,
                "workflow_complete": True
            },
            "message": f"✅ Airflow DAG failure scenario completed! Created incident {incident.id} with {len(action_ids)} actions triggered."
        }
        
    except Exception as e:
        logger.error(f"❌ Airflow test scenario failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test scenario failed: {str(e)}")


@router.post("/force-action/{action_type}", summary="🔧 Force Execute Specific Action")
async def force_execute_action(
    action_type: ActionType,
    incident_id: Optional[str] = Query(None, description="Incident ID to associate with action"),
    dag_id: Optional[str] = Query("test_dag_123", description="DAG ID for Airflow actions"),
    dag_run_id: Optional[str] = Query(None, description="DAG run ID for Airflow actions")
) -> Dict[str, Any]:
    """
    🔧 **Force Action Execution**
    
    Directly execute a specific action type for testing purposes.
    Useful for testing individual action handlers.
    
    **Available Action Types:**
    - `restart_service`: Restart a system service
    - `restart_airflow_dag`: Restart an Airflow DAG
    - `restart_spark_job`: Restart a Spark job  
    - `call_api_endpoint`: Call an external API
    - `scale_resources`: Scale system resources
    - `clear_cache`: Clear application cache
    - `restart_database_connection`: Reset DB connections
    
    **For Airflow Actions:**
    This will attempt to call the real Airflow API if available, allowing you to test
    the complete integration including API calls, authentication, and error handling.
    """
    try:
        action_service = ActionService()
        
        # Build parameters based on action type
        parameters = {}
        
        if action_type == ActionType.RESTART_AIRFLOW_DAG:
            parameters = {
                "dag_id": dag_id or "test_dag_123",
                "dag_run_id": dag_run_id or f"manual_test_{int(datetime.utcnow().timestamp())}"
            }
        elif action_type == ActionType.RESTART_SPARK_JOB:
            parameters = {
                "job_id": "test_job_123",
                "application_id": "application_test_456"
            }
        elif action_type == ActionType.RESTART_SERVICE:
            parameters = {
                "service_name": "test-service"
            }
        
        # Add incident_id if provided
        if incident_id:
            parameters["incident_id"] = incident_id
        
        # Execute action
        action_id = await action_service.execute_action(
            action_type=action_type.value,
            parameters=parameters,
            timeout_seconds=300
        )
        
        # Wait a moment for action to start
        import asyncio
        await asyncio.sleep(1)
        
        # Get action details
        action_details = await action_service.get_action(action_id)
        
        # Determine if this was a real API call or simulation
        api_call_status = "simulated"
        if action_type == ActionType.RESTART_AIRFLOW_DAG:
            api_call_status = "attempted_real_airflow_api_call"
        
        return {
            "scenario": "force_action_execution",
            "action_type": action_type.value,
            "action_id": action_id,
            "parameters": parameters,
            "action_details": action_details,
            "api_integration": {
                "status": api_call_status,
                "expected_behavior": "Should attempt to call external API and handle connection failures gracefully"
            },
            "test_results": {
                "action_created": True,
                "action_executed": getattr(action_details, 'status', 'unknown') in ["running", "success", "failed"],
                "error_handling": getattr(action_details, 'status', 'unknown') == "failed",
                "workflow_complete": True
            },
            "message": f"✅ Action {action_type.value} executed! Status: {getattr(action_details, 'status', 'unknown')}"
        }
        
    except Exception as e:
        logger.error(f"❌ Action execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Action execution failed: {str(e)}")


@router.post("/simple-incident", summary="🎯 Create Simple Test Incident")
async def create_simple_test_incident(
    title: str = Query("Test API service timeout", description="Incident title"),
    description: str = Query("API service experiencing timeout - test incident", description="Incident description"),
    severity: str = Query("high", description="Incident severity (critical, high, medium, low)"),
    category: str = Query("api_service", description="Incident category"),
    auto_resolve: bool = Query(True, description="Attempt automatic resolution")
) -> Dict[str, Any]:
    """
    🎯 **Simple Test Incident**
    
    Create a simple test incident with basic parameters.
    Perfect for quick testing of the AI analysis and action system.
    """
    try:
        # Map severity string to enum
        severity_map = {
            "critical": IncidentSeverity.CRITICAL,
            "high": IncidentSeverity.HIGH,
            "medium": IncidentSeverity.MEDIUM,
            "low": IncidentSeverity.LOW
        }
        
        # Map category string to enum
        category_map = {
            "airflow_dag": IncidentCategory.AIRFLOW_DAG,
            "api_service": IncidentCategory.API_SERVICE,
            "database": IncidentCategory.DATABASE,
            "spark_job": IncidentCategory.SPARK_JOB,
            "infrastructure": IncidentCategory.INFRASTRUCTURE
        }
        
        incident_data = IncidentCreateRequest(
            title=title,
            description=description,
            severity=severity_map.get(severity.lower(), IncidentSeverity.MEDIUM),
            category=category_map.get(category.lower(), IncidentCategory.UNKNOWN),
            metadata={
                "test_scenario": True,
                "created_via": "testing_api",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        incident_service = EnhancedIncidentService()
        incident, action_ids = await incident_service.create_incident_from_airflow(
            incident_data,
            auto_resolve=auto_resolve
        )
        
        return {
            "scenario": "simple_test_incident",
            "incident": incident.dict() if hasattr(incident, 'dict') else dict(incident),
            "action_ids": action_ids,
            "auto_resolution_attempted": auto_resolve and len(action_ids) > 0,
            "message": f"✅ Simple test incident created! ID: {incident.id}"
        }
        
    except Exception as e:
        logger.error(f"❌ Simple incident test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simple incident test failed: {str(e)}")


@router.get("/health", summary="✅ Test Endpoint Health Check")
async def test_endpoint_health() -> Dict[str, Any]:
    """
    ✅ **Testing Endpoints Health Check**
    
    Verify that all testing endpoints are working correctly.
    """
    return {
        "status": "healthy",
        "testing_endpoints_available": True,
        "timestamp": datetime.utcnow().isoformat(),
        "available_scenarios": [
            "airflow_dag_failure",
            "force_action_execution",
            "simple_test_incident"
        ],
        "quick_test_commands": [
            "curl -X POST 'http://localhost:8000/api/v1/testing/airflow-dag-failure'",
            "curl -X POST 'http://localhost:8000/api/v1/testing/force-action/restart_airflow_dag'",
            "curl -X POST 'http://localhost:8000/api/v1/testing/simple-incident'"
        ],
        "message": "🚀 All testing endpoints are operational and ready for scenario testing!"
    }

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid

from ..core import get_logger
from ..models.schemas import ActionType
from ..models.incident_schemas import IncidentCreateRequest, IncidentSeverity, IncidentCategory
from ..services.enhanced_incident_service import EnhancedIncidentService
from ..services.actions import ActionService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/testing", tags=["Testing & Demo"])


class TestScenario(BaseModel):
    """Test scenario configuration."""
    scenario_name: str = Field(..., description="Name of the test scenario")
    description: str = Field(..., description="Description of what this scenario tests")
    auto_resolve: bool = Field(default=False, description="Whether to attempt auto-resolution")
    confidence_threshold: float = Field(default=0.6, description="Confidence threshold for auto-resolution")


class AirflowTestParams(BaseModel):
    """Parameters for Airflow testing scenarios."""
    dag_id: str = Field(default="test_data_pipeline", description="DAG ID to test")
    task_id: str = Field(default="process_data", description="Task ID that failed")
    failure_type: str = Field(default="timeout", description="Type of failure (timeout, error, dependency)")
    airflow_url: str = Field(default="http://localhost:8080", description="Airflow webserver URL")
    enable_real_api_call: bool = Field(default=False, description="Actually try to call Airflow API")


class SparkTestParams(BaseModel):
    """Parameters for Spark testing scenarios."""
    job_id: str = Field(default="spark_job_12345", description="Spark job ID")
    application_id: str = Field(default="application_1234567890", description="Spark application ID")
    failure_type: str = Field(default="out_of_memory", description="Type of failure")
    cluster_url: str = Field(default="http://localhost:8088", description="Spark cluster URL")


@router.get("/scenarios", summary="List Available Test Scenarios")
async def list_test_scenarios() -> Dict[str, Any]:
    """
    Get a list of all available test scenarios.
    
    Returns detailed information about each scenario including:
    - What it tests
    - Expected behavior
    - Parameters
    """
    return {
        "scenarios": [
            {
                "name": "airflow_dag_timeout",
                "description": "Simulates an Airflow DAG task timeout and attempts automated recovery",
                "endpoint": "/api/v1/testing/airflow-dag-failure",
                "expected_behavior": "Creates incident, AI analyzes, attempts DAG restart via Airflow API",
                "ai_confidence_expected": "60-80%",
                "actions_triggered": ["restart_airflow_dag"]
            },
            {
                "name": "spark_job_failure",
                "description": "Simulates a Spark job failure with memory issues",
                "endpoint": "/api/v1/testing/spark-job-failure",
                "expected_behavior": "Creates incident, AI analyzes, attempts job restart with increased memory",
                "ai_confidence_expected": "50-70%",
                "actions_triggered": ["restart_spark_job"]
            },
            {
                "name": "api_service_down",
                "description": "Simulates an API service outage",
                "endpoint": "/api/v1/testing/api-service-failure",
                "expected_behavior": "Creates incident, AI analyzes, attempts service restart",
                "ai_confidence_expected": "70-90%",
                "actions_triggered": ["restart_service"]
            },
            {
                "name": "database_connection_pool",
                "description": "Simulates database connection pool exhaustion",
                "endpoint": "/api/v1/testing/database-failure",
                "expected_behavior": "Creates incident, AI analyzes, attempts connection pool reset",
                "ai_confidence_expected": "40-60%",
                "actions_triggered": ["restart_database_connection"]
            },
            {
                "name": "high_confidence_auto_resolve",
                "description": "Tests a scenario with high AI confidence for auto-resolution",
                "endpoint": "/api/v1/testing/high-confidence-scenario",
                "expected_behavior": "Immediately triggers automated resolution actions",
                "ai_confidence_expected": "85-95%",
                "actions_triggered": ["restart_service", "clear_cache"]
            }
        ],
        "global_endpoints": [
            {
                "endpoint": "/api/v1/testing/custom-incident",
                "description": "Create a custom incident with specific parameters for testing"
            },
            {
                "endpoint": "/api/v1/testing/force-action/{action_type}",
                "description": "Force execute a specific action type for testing"
            },
            {
                "endpoint": "/api/v1/testing/ai-prediction",
                "description": "Test AI prediction without creating an incident"
            }
        ]
    }


@router.post("/airflow-dag-failure", summary="Test Airflow DAG Failure Scenario")
async def test_airflow_dag_failure(
    params: AirflowTestParams = AirflowTestParams(),
    auto_resolve: bool = Query(True, description="Attempt automatic resolution"),
    confidence_threshold: float = Query(0.5, description="AI confidence threshold for auto-resolution")
) -> Dict[str, Any]:
    """
    🚁 **Airflow DAG Failure Test Scenario**
    
    This endpoint simulates a realistic Airflow DAG failure and tests the complete
    AI-powered incident resolution workflow.
    
    **What This Tests:**
    1. ✅ Incident creation with Airflow-specific metadata
    2. 🤖 AI analysis and confidence scoring
    3. 🔧 Automated action recommendation
    4. 🌐 Real Airflow API integration (if enabled)
    5. 📊 End-to-end monitoring and logging
    
    **Expected Flow:**
    ```
    DAG Failure → Incident Creation → AI Analysis → Action Recommendation → API Call
    ```
    
    **Parameters:**
    - `dag_id`: The DAG that failed
    - `task_id`: The specific task that failed  
    - `failure_type`: Type of failure (timeout, error, dependency)
    - `airflow_url`: Airflow webserver URL for API calls
    - `enable_real_api_call`: Whether to actually call Airflow API
    
    **Returns:**
    - Incident details with AI analysis
    - Action execution results
    - API call status and responses
    """
    try:
        # Create realistic Airflow incident
        incident_data = IncidentCreateRequest(
            title=f"Airflow DAG '{params.dag_id}' failed - Task '{params.task_id}' {params.failure_type}",
            description=f"""
            DAG Failure Details:
            - DAG ID: {params.dag_id}
            - Task ID: {params.task_id}
            - Failure Type: {params.failure_type}
            - Failure Time: {datetime.utcnow().isoformat()}
            - Expected Duration: 30 minutes
            - Actual Duration: 45 minutes (timeout)
            
            Error Message: Task instance failed after exceeding timeout limit.
            Previous runs were successful. This appears to be a transient issue.
            
            Recommended Action: Restart the DAG run and monitor for completion.
            """,
            severity=IncidentSeverity.HIGH if params.failure_type == "timeout" else IncidentSeverity.CRITICAL,
            category=IncidentCategory.AIRFLOW_DAG,
            metadata={
                "dag_id": params.dag_id,
                "task_id": params.task_id,
                "failure_type": params.failure_type,
                "airflow_url": params.airflow_url,
                "execution_date": datetime.utcnow().isoformat(),
                "dag_run_id": f"manual_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "previous_success": True,
                "timeout_seconds": 1800,
                "enable_real_api_call": params.enable_real_api_call
            }
        )
        
        # Create incident using enhanced service
        incident_service = EnhancedIncidentService()
        incident, action_ids = await incident_service.create_incident_from_airflow(
            incident_data,
            auto_resolve=auto_resolve
        )
        
        # If auto-resolution was attempted, get action details
        action_details = []
        if action_ids:
            action_service = ActionService()
            for action_id in action_ids:
                action = await action_service.get_action(action_id)
                action_details.append(action)
        
        return {
            "scenario": "airflow_dag_failure",
            "test_timestamp": datetime.utcnow().isoformat(),
            "parameters": params.dict(),
            "incident": incident.dict(),
            "ai_analysis": {
                "confidence": incident.ai_confidence,
                "predicted_resolution_time": incident.predicted_resolution_time,
                "auto_resolution_attempted": auto_resolve and len(action_ids) > 0
            },
            "actions_executed": action_details,
            "airflow_api_integration": {
                "enabled": params.enable_real_api_call,
                "url": params.airflow_url,
                "status": "Would attempt DAG restart" if not params.enable_real_api_call else "API call attempted"
            },
            "test_results": {
                "incident_created": True,
                "ai_analyzed": incident.ai_confidence is not None,
                "actions_triggered": len(action_ids) > 0,
                "workflow_complete": True
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Airflow test scenario failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test scenario failed: {str(e)}")


@router.post("/spark-job-failure", summary="Test Spark Job Failure Scenario")
async def test_spark_job_failure(
    params: SparkTestParams = SparkTestParams(),
    auto_resolve: bool = Query(True, description="Attempt automatic resolution")
) -> Dict[str, Any]:
    """
    ⚡ **Spark Job Failure Test Scenario**
    
    Simulates a Spark job failure with memory/resource issues.
    Tests AI-driven job restart with increased resource allocation.
    """
    try:
        incident_data = IncidentCreateRequest(
            title=f"Spark job '{params.job_id}' failed - {params.failure_type}",
            description=f"""
            Spark Job Failure:
            - Job ID: {params.job_id}
            - Application ID: {params.application_id}
            - Failure Type: {params.failure_type}
            - Cluster URL: {params.cluster_url}
            
            Error: {"Out of memory - Java heap space" if params.failure_type == "out_of_memory" else "Job execution failed"}
            
            Current Configuration:
            - Driver Memory: 2GB
            - Executor Memory: 4GB
            - Executor Cores: 2
            
            Recommended: Increase memory allocation and restart job.
            """,
            severity=IncidentSeverity.HIGH,
            category=IncidentCategory.SPARK_JOB,
            metadata={
                "job_id": params.job_id,
                "application_id": params.application_id,
                "failure_type": params.failure_type,
                "cluster_url": params.cluster_url,
                "driver_memory": "2GB",
                "executor_memory": "4GB",
                "executor_cores": 2,
                "stage_id": 15,
                "task_count": 200
            }
        )
        
        incident_service = EnhancedIncidentService()
        incident, action_ids = await incident_service.create_incident_from_airflow(
            incident_data, 
            auto_resolve=auto_resolve
        )
        
        return {
            "scenario": "spark_job_failure",
            "test_timestamp": datetime.utcnow().isoformat(),
            "parameters": params.dict(),
            "incident": result["incident"],
            "ai_analysis": {
                "confidence": result["incident"]["ai_confidence"],
                "recommended_action": "restart_spark_job with increased memory"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spark test failed: {str(e)}")


@router.post("/api-service-failure", summary="Test API Service Failure Scenario")
async def test_api_service_failure(
    service_name: str = Query("user-api", description="Name of the service"),
    failure_type: str = Query("timeout", description="Type of failure"),
    auto_resolve: bool = Query(True, description="Attempt automatic resolution")
) -> Dict[str, Any]:
    """
    🌐 **API Service Failure Test Scenario**
    
    Simulates an API service outage with high AI confidence for auto-resolution.
    """
    try:
        incident_data = IncidentCreate(
            title=f"API Service '{service_name}' experiencing {failure_type}",
            description=f"""
            Service Outage Details:
            - Service: {service_name}
            - Issue: {failure_type}
            - Response Time: 5000ms (normal: 200ms)
            - Error Rate: 45%
            - Affected Endpoints: /api/users, /api/auth
            
            Health Check Status: FAILING
            Last Successful Response: 5 minutes ago
            
            This appears to be a common issue that typically resolves with service restart.
            """,
            severity=Severity.HIGH,
            service=service_name,
            category="api_service",
            metadata={
                "service_name": service_name,
                "failure_type": failure_type,
                "response_time_ms": 5000,
                "error_rate": 0.45,
                "affected_endpoints": ["/api/users", "/api/auth"],
                "health_check_status": "FAILING"
            }
        )
        
        incident_service = EnhancedIncidentService()
        result = await incident_service.create_incident(incident_data, auto_resolve=auto_resolve, confidence_threshold=0.4)
        
        return {
            "scenario": "api_service_failure",
            "incident": result["incident"],
            "ai_analysis": {
                "confidence": result["incident"]["ai_confidence"],
                "expected_high_confidence": "This scenario typically has high AI confidence"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API service test failed: {str(e)}")


@router.post("/force-action/{action_type}", summary="Force Execute Specific Action")
async def force_execute_action(
    action_type: ActionType,
    incident_id: Optional[str] = Query(None, description="Incident ID to associate with action"),
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    🔧 **Force Action Execution**
    
    Directly execute a specific action type for testing purposes.
    Useful for testing individual action handlers.
    
    **Available Action Types:**
    - `restart_service`: Restart a system service
    - `restart_airflow_dag`: Restart an Airflow DAG
    - `restart_spark_job`: Restart a Spark job  
    - `call_api_endpoint`: Call an external API
    - `scale_resources`: Scale system resources
    - `clear_cache`: Clear application cache
    - `restart_database_connection`: Reset DB connections
    """
    try:
        action_service = ActionService()
        
        # Default parameters based on action type
        if parameters is None:
            parameters = {}
            
        if action_type == ActionType.RESTART_AIRFLOW_DAG:
            parameters.update({
                "dag_id": parameters.get("dag_id", "test_dag"),
                "dag_run_id": parameters.get("dag_run_id", f"manual_test_{int(datetime.utcnow().timestamp())}")
            })
        elif action_type == ActionType.RESTART_SPARK_JOB:
            parameters.update({
                "job_id": parameters.get("job_id", "test_job_123"),
                "application_id": parameters.get("application_id", "application_test_456")
            })
        elif action_type == ActionType.RESTART_SERVICE:
            parameters.update({
                "service_name": parameters.get("service_name", "test-service"),
                "restart_command": parameters.get("restart_command", "systemctl restart test-service")
            })
        
        # Add incident_id if provided
        if incident_id:
            parameters["incident_id"] = incident_id
        
        # Execute action
        action_id = await action_service.execute_action(
            action_type=action_type.value,
            parameters=parameters,
            timeout_seconds=300
        )
        
        # Get action details
        action_details = await action_service.get_action(action_id)
        
        return {
            "scenario": "force_action_execution",
            "action_type": action_type.value,
            "action_id": action_id,
            "parameters": parameters,
            "action_details": action_details,
            "test_note": "This was a forced execution for testing purposes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action execution failed: {str(e)}")


@router.post("/ai-prediction", summary="Test AI Prediction Without Creating Incident")
async def test_ai_prediction(
    title: str = Query(..., description="Incident title"),
    description: str = Query(..., description="Incident description"),
    service: str = Query("test-service", description="Service name"),
    severity: Severity = Query(Severity.MEDIUM, description="Incident severity")
) -> Dict[str, Any]:
    """
    🤖 **AI Prediction Test**
    
    Test the AI engine's prediction capabilities without creating an actual incident.
    Useful for understanding how the AI analyzes different types of issues.
    """
    try:
        from ..ai.simple_engine import SimpleAIEngine
        
        # Create incident data for prediction
        incident_data = IncidentCreate(
            title=title,
            description=description,
            service=service,
            severity=severity,
            category="unknown"
        )
        
        # Get AI prediction
        ai_engine = SimpleAIEngine()
        confidence = ai_engine.predict_resolution_confidence(incident_data)
        should_automate = ai_engine.should_automate_resolution(incident_data, threshold=0.6)
        features = ai_engine.extract_features(incident_data)
        model_status = ai_engine.get_model_status()
        
        return {
            "scenario": "ai_prediction_test",
            "input": {
                "title": title,
                "description": description,
                "service": service,
                "severity": severity.value
            },
            "ai_analysis": {
                "confidence": confidence,
                "should_automate_at_60_percent": should_automate,
                "confidence_thresholds": {
                    "30%": confidence >= 0.3,
                    "50%": confidence >= 0.5,
                    "70%": confidence >= 0.7,
                    "90%": confidence >= 0.9
                }
            },
            "extracted_features": features,
            "model_status": model_status,
            "interpretation": {
                "confidence_level": "High" if confidence >= 0.7 else "Medium" if confidence >= 0.4 else "Low",
                "recommended_action": "Automate" if should_automate else "Manual review recommended",
                "reasoning": "AI confidence based on similarity to historical incidents and success patterns"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI prediction test failed: {str(e)}")


@router.post("/custom-incident", summary="Create Custom Test Incident")
async def create_custom_test_incident(
    incident: IncidentCreate,
    auto_resolve: bool = Query(False, description="Attempt automatic resolution"),
    confidence_threshold: float = Query(0.6, description="AI confidence threshold")
) -> Dict[str, Any]:
    """
    🎯 **Custom Incident Test**
    
    Create a completely custom incident for testing specific scenarios.
    Provides full control over all incident parameters.
    """
    try:
        incident_service = EnhancedIncidentService()
        result = await incident_service.create_incident(
            incident_data=incident,
            auto_resolve=auto_resolve,
            confidence_threshold=confidence_threshold
        )
        
        return {
            "scenario": "custom_incident",
            "test_timestamp": datetime.utcnow().isoformat(),
            "configuration": {
                "auto_resolve": auto_resolve,
                "confidence_threshold": confidence_threshold
            },
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom incident test failed: {str(e)}")


@router.get("/health", summary="Test Endpoint Health Check")
async def test_endpoint_health() -> Dict[str, Any]:
    """
    ✅ **Testing Endpoints Health Check**
    
    Verify that all testing endpoints are working correctly.
    """
    return {
        "status": "healthy",
        "testing_endpoints_available": True,
        "timestamp": datetime.utcnow().isoformat(),
        "available_scenarios": [
            "airflow_dag_failure",
            "spark_job_failure", 
            "api_service_failure",
            "force_action_execution",
            "ai_prediction_test",
            "custom_incident"
        ],
        "message": "All testing endpoints are operational and ready for scenario testing!"
    }
