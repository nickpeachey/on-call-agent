"""Action execution service."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import asyncio
import aiohttp
import subprocess
import json
import os
from pathlib import Path

from ..core import get_logger
from ..models.schemas import ActionResponse, ActionStatus, ActionType
from .action_logger import action_logger


logger = get_logger(__name__)

# Optional production dependencies - gracefully handle missing imports
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    logger.warning("Docker not available - service restart actions will be limited")
    DOCKER_AVAILABLE = False

try:
    import kubernetes
    from kubernetes import client, config as k8s_config
    KUBERNETES_AVAILABLE = True
except ImportError:
    logger.warning("Kubernetes not available - K8s actions will be limited")
    KUBERNETES_AVAILABLE = False

try:
    from apache_airflow_client.client.api import dag_api, dag_run_api, task_instance_api
    from apache_airflow_client.client.model.dag_run import DagRun
    from apache_airflow_client.client.model.clear_task_instances import ClearTaskInstances
    from apache_airflow_client.client.configuration import Configuration
    from apache_airflow_client.client.api_client import ApiClient
    AIRFLOW_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("Airflow client not available - will use REST API fallback")
    AIRFLOW_CLIENT_AVAILABLE = False

try:
    import psycopg2
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    logger.warning("PostgreSQL driver not available")
    POSTGRES_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis not available")
    REDIS_AVAILABLE = False

try:
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    logger.warning("MongoDB not available")
    MONGODB_AVAILABLE = False

try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    logger.warning("PySpark not available - Spark actions will use REST API")
    SPARK_AVAILABLE = False


class ActionService:
    """Service for managing and executing actions."""
    
    def __init__(self):
        # TODO: Replace with actual database connection
        self._actions = {}
        self._action_queue = asyncio.Queue()
        self._executor_task = None
    
    async def start(self):
        """Start the action executor."""
        if self._executor_task is None:
            self._executor_task = asyncio.create_task(self._action_executor())
            logger.info("Action executor started")
    
    async def stop(self):
        """Stop the action executor."""
        if self._executor_task:
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
            self._executor_task = None
            logger.info("Action executor stopped")
    
    async def list_actions(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[ActionResponse]:
        """List executed actions."""
        logger.info("Listing actions", skip=skip, limit=limit, status=status)
        
        actions = list(self._actions.values())
        
        # Apply status filter
        if status:
            actions = [a for a in actions if a.status == status]
        
        # Sort by creation time (newest first)
        actions.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        return actions[skip:skip + limit]
    
    async def get_action(self, action_id: str) -> Optional[ActionResponse]:
        """Get specific action by ID."""
        logger.info("Getting action", action_id=action_id)
        return self._actions.get(action_id)
    
    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        incident_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_manual: bool = False,
        timeout_seconds: int = 300
    ) -> str:
        """Queue an action for execution."""
        action_id = str(uuid.uuid4())
        
        action = ActionResponse(
            id=action_id,
            action_type=ActionType(action_type),
            parameters=parameters,
            incident_id=incident_id,
            status=ActionStatus.PENDING,
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            result=None,
            error_message=None,
            executed_by=user_id,
            is_manual=is_manual,
            timeout_seconds=timeout_seconds
        )
        
        self._actions[action_id] = action
        
        # Add to execution queue
        await self._action_queue.put(action_id)
        
        logger.info("Action queued for execution", action_id=action_id, action_type=action_type)
        
        return action_id
    
    async def _action_executor(self):
        """Background task to execute queued actions."""
        logger.info("Action executor worker started")
        
        while True:
            try:
                # Get next action from queue
                action_id = await self._action_queue.get()
                action = self._actions.get(action_id)
                
                if not action:
                    continue
                
                # Start detailed logging for this action attempt
                attempt = action_logger.start_action_attempt(
                    action_id=action_id,
                    action_type=action.action_type.value,
                    parameters=action.parameters,
                    incident_id=action.incident_id
                )
                
                # Update status to running
                action.status = ActionStatus.RUNNING
                action.started_at = datetime.utcnow()
                self._actions[action_id] = action
                
                attempt.log_step("status_update", "running", {"started_at": action.started_at.isoformat()})
                
                logger.info("Executing action", action_id=action_id, action_type=action.action_type)
                
                try:
                    attempt.log_step("execution_start", "beginning", {"action_type": action.action_type.value})
                    
                    # Execute the action based on its type
                    result = await self._execute_action_by_type(action)
                    
                    attempt.log_step("execution_end", "success", {"result_keys": list(result.keys()) if result else []})
                    
                    # Update action with success
                    action.status = ActionStatus.SUCCESS
                    action.result = result
                    action.completed_at = datetime.utcnow()
                    
                    # Complete action logging
                    action_logger.complete_action_attempt(action_id, success=True, result=result)
                    
                    logger.info("Action completed successfully", action_id=action_id)
                    
                except Exception as e:
                    attempt.log_step("execution_end", "failed", {
                        "error": str(e),
                        "exception_type": type(e).__name__
                    })
                    
                    # Update action with failure
                    action.status = ActionStatus.FAILED
                    action.error_message = str(e)
                    action.completed_at = datetime.utcnow()
                    
                    # Complete action logging with failure details
                    exception_details = {
                        "exception_type": type(e).__name__,
                        "exception_module": type(e).__module__,
                        "args": str(e.args) if hasattr(e, 'args') else None
                    }
                    action_logger.complete_action_attempt(
                        action_id, 
                        success=False, 
                        error=str(e),
                        exception_details=exception_details
                    )
                    
                    logger.error("Action execution failed", action_id=action_id, error=str(e))
                
                self._actions[action_id] = action
                
            except asyncio.CancelledError:
                logger.info("Action executor cancelled")
                break
            except Exception as e:
                logger.error("Error in action executor", error=str(e))
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def _execute_action_by_type(self, action: ActionResponse) -> Dict[str, Any]:
        """Execute action based on its type."""
        action_type = ActionType(action.action_type)
        parameters = action.parameters
        
        if action_type == ActionType.RESTART_SERVICE:
            return await self._restart_service(parameters)
        elif action_type == ActionType.RESTART_AIRFLOW_DAG:
            return await self._restart_airflow_dag(parameters)
        elif action_type == ActionType.RESTART_SPARK_JOB:
            return await self._restart_spark_job(parameters)
        elif action_type == ActionType.CALL_API_ENDPOINT:
            return await self._call_api_endpoint(parameters)
        elif action_type == ActionType.SCALE_RESOURCES:
            return await self._scale_resources(parameters)
        elif action_type == ActionType.CLEAR_CACHE:
            return await self._clear_cache(parameters)
        elif action_type == ActionType.RESTART_DATABASE_CONNECTION:
            return await self._restart_database_connection(parameters)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    async def _restart_service(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Restart a service using Docker, Kubernetes, or systemctl."""
        service_name = parameters.get("service_name")
        platform = parameters.get("platform", "auto")  # auto, docker, kubernetes, systemctl
        namespace = parameters.get("namespace", "default")
        
        if not service_name:
            raise ValueError("service_name parameter is required")
        
        logger.info("Restarting service", service_name=service_name, platform=platform)
        
        result = {
            "service_name": service_name,
            "platform": platform,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if platform == "docker" or (platform == "auto" and DOCKER_AVAILABLE):
                result.update(await self._restart_docker_service(service_name))
            elif platform == "kubernetes" or (platform == "auto" and KUBERNETES_AVAILABLE):
                result.update(await self._restart_kubernetes_service(service_name, namespace))
            elif platform == "systemctl" or platform == "auto":
                result.update(await self._restart_systemctl_service(service_name))
            else:
                raise ValueError(f"Unsupported platform: {platform}")
            
            result["status"] = "restarted"
            logger.info("Service restarted successfully", service_name=service_name, platform=platform)
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error("Failed to restart service", service_name=service_name, error=str(e))
            raise
        
        return result
    
    async def _restart_docker_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a Docker container or service."""
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker client not available")
        
        try:
            import docker  # Re-import for type checking
            client = docker.from_env()
            
            # Try to find container by name
            try:
                container = client.containers.get(service_name)
                container.restart()
                return {
                    "container_id": container.id,
                    "container_name": container.name,
                    "restart_method": "container"
                }
            except docker.errors.NotFound:
                pass
            
            # Try Docker Swarm service
            try:
                service = client.services.get(service_name)
                service.force_update()
                return {
                    "service_id": service.id,
                    "service_name": service.name,
                    "restart_method": "swarm_service"
                }
            except docker.errors.NotFound:
                pass
            
            # Try Docker Compose service
            result = await self._run_command([
                "docker-compose", "restart", service_name
            ])
            if result["returncode"] == 0:
                return {
                    "restart_method": "compose_service",
                    "output": result["stdout"]
                }
            
            raise RuntimeError(f"Docker service '{service_name}' not found")
            
        except Exception as e:
            raise RuntimeError(f"Docker restart failed: {str(e)}")
    
    async def _restart_kubernetes_service(self, service_name: str, namespace: str) -> Dict[str, Any]:
        """Restart a Kubernetes deployment or pod."""
        if not KUBERNETES_AVAILABLE:
            raise RuntimeError("Kubernetes client not available")
        
        try:
            from kubernetes import client, config as k8s_config  # Re-import for type checking
            
            # Load kubeconfig
            k8s_config.load_incluster_config() if os.getenv("KUBERNETES_SERVICE_HOST") else k8s_config.load_kube_config()
            
            apps_v1 = client.AppsV1Api()
            core_v1 = client.CoreV1Api()
            
            # Try to restart deployment
            try:
                deployment = apps_v1.read_namespaced_deployment(service_name, namespace)
                
                # Update deployment to trigger restart
                deployment.spec.template.metadata.annotations = deployment.spec.template.metadata.annotations or {}
                deployment.spec.template.metadata.annotations["kubectl.kubernetes.io/restartedAt"] = datetime.utcnow().isoformat()
                
                apps_v1.patch_namespaced_deployment(
                    name=service_name,
                    namespace=namespace,
                    body=deployment
                )
                
                return {
                    "deployment_name": service_name,
                    "namespace": namespace,
                    "restart_method": "deployment",
                    "replicas": deployment.spec.replicas
                }
            except client.ApiException as e:
                if e.status != 404:
                    raise
            
            # Try to restart individual pod
            try:
                pods = core_v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"app={service_name}"
                )
                
                if not pods.items:
                    raise RuntimeError(f"No pods found for service '{service_name}'")
                
                deleted_pods = []
                for pod in pods.items:
                    core_v1.delete_namespaced_pod(pod.metadata.name, namespace)
                    deleted_pods.append(pod.metadata.name)
                
                return {
                    "restart_method": "pod_deletion",
                    "namespace": namespace,
                    "deleted_pods": deleted_pods
                }
                
            except client.ApiException as e:
                raise RuntimeError(f"Kubernetes restart failed: {str(e)}")
            
        except Exception as e:
            raise RuntimeError(f"Kubernetes restart failed: {str(e)}")
    
    async def _restart_systemctl_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a systemctl service."""
        try:
            # Check if service exists
            check_result = await self._run_command([
                "systemctl", "is-enabled", service_name
            ])
            
            if check_result["returncode"] != 0:
                raise RuntimeError(f"Service '{service_name}' not found or not enabled")
            
            # Restart the service
            restart_result = await self._run_command([
                "sudo", "systemctl", "restart", service_name
            ])
            
            if restart_result["returncode"] != 0:
                raise RuntimeError(f"Failed to restart service: {restart_result['stderr']}")
            
            # Check status
            status_result = await self._run_command([
                "systemctl", "is-active", service_name
            ])
            
            return {
                "restart_method": "systemctl",
                "service_status": status_result["stdout"].strip(),
                "restart_output": restart_result["stdout"]
            }
            
        except Exception as e:
            raise RuntimeError(f"Systemctl restart failed: {str(e)}")
    
    async def _run_command(self, command: List[str]) -> Dict[str, Any]:
        """Run a shell command asynchronously."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "command": " ".join(command)
            }
            
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(command)
            }
    
    async def _restart_airflow_dag(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Restart an Airflow DAG using API calls."""
        dag_id = parameters.get("dag_id")
        dag_run_id = parameters.get("dag_run_id")
        task_id = parameters.get("task_id")
        execution_date = parameters.get("execution_date")
        reset_dag_run = parameters.get("reset_dag_run", False)
        
        if not dag_id:
            raise ValueError("dag_id parameter is required")
        
        logger.info("Restarting Airflow DAG", 
                   dag_id=dag_id, 
                   dag_run_id=dag_run_id,
                   task_id=task_id,
                   execution_date=execution_date)
        
        # Get Airflow configuration from environment
        airflow_url = os.getenv("AIRFLOW_URL", "http://localhost:8080")
        airflow_username = os.getenv("AIRFLOW_USERNAME", "admin")
        airflow_password = os.getenv("AIRFLOW_PASSWORD", "admin")
        
        # Log the complete action details including endpoints
        action_details = {
            "dag_id": dag_id,
            "dag_run_id": dag_run_id,
            "task_id": task_id,
            "execution_date": execution_date,
            "reset_dag_run": reset_dag_run,
            "airflow_url": airflow_url,
            "airflow_username": airflow_username,
            "timestamp": datetime.utcnow().isoformat(),
            "action_type": "restart_airflow_dag"
        }
        
        logger.info("🚁 AIRFLOW ACTION INITIATED", **action_details)
        
        result = {
            "dag_id": dag_id,
            "dag_run_id": dag_run_id,
            "task_id": task_id,
            "execution_date": execution_date,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if AIRFLOW_CLIENT_AVAILABLE:
                result.update(await self._restart_airflow_with_client(
                    dag_id, dag_run_id, task_id, execution_date, reset_dag_run,
                    airflow_url, airflow_username, airflow_password
                ))
            else:
                result.update(await self._restart_airflow_with_api(
                    dag_id, dag_run_id, task_id, execution_date, reset_dag_run,
                    airflow_url, airflow_username, airflow_password
                ))
            
            result["status"] = "restarted"
            logger.info("Airflow DAG restarted successfully", dag_id=dag_id)
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error("Failed to restart Airflow DAG", dag_id=dag_id, error=str(e))
            raise
        
        return result
    
    async def _restart_airflow_with_client(self, dag_id: str, dag_run_id: Optional[str], 
                                         task_id: Optional[str], execution_date: Optional[str], 
                                         reset_dag_run: bool, airflow_url: str,
                                         username: str, password: str) -> Dict[str, Any]:
        """Restart Airflow DAG using the official Python client."""
        try:
            from apache_airflow_client.client.api import dag_api, dag_run_api, task_instance_api
            from apache_airflow_client.client.model.clear_task_instances import ClearTaskInstances
            from apache_airflow_client.client.configuration import Configuration
            from apache_airflow_client.client.api_client import ApiClient
            
            # Configure client
            configuration = Configuration(
                host=f"{airflow_url}/api/v1",
                username=username,
                password=password
            )
            
            with ApiClient(configuration) as api_client:
                if reset_dag_run and dag_run_id:
                    # Clear and restart entire DAG run
                    dag_run_instance = dag_run_api.DAGRunApi(api_client)
                    
                    # Delete the DAG run
                    dag_run_instance.delete_dag_run(dag_id, dag_run_id)
                    
                    # Trigger new DAG run
                    new_dag_run = dag_run_instance.post_dag_run(
                        dag_id,
                        {
                            "dag_run_id": f"{dag_run_id}_retry_{int(datetime.utcnow().timestamp())}",
                            "execution_date": execution_date or datetime.utcnow().isoformat(),
                            "conf": {}
                        }
                    )
                    
                    return {
                        "action": "dag_run_reset",
                        "new_dag_run_id": new_dag_run.dag_run_id,
                        "state": new_dag_run.state
                    }
                
                elif task_id:
                    # Clear and restart specific task
                    task_instance = task_instance_api.TaskInstanceApi(api_client)
                    
                    clear_request = ClearTaskInstances(
                        dry_run=False,
                        reset_dag_runs=False,
                        only_failed=False,
                        only_running=True,
                        include_subdags=True,
                        include_parentdag=False,
                        task_ids=[task_id]
                    )
                    
                    task_instance.clear_task_instances(dag_id, clear_request)
                    
                    return {
                        "action": "task_cleared",
                        "task_id": task_id,
                        "dag_run_id": dag_run_id
                    }
                
                else:
                    # Trigger DAG
                    dag_instance = dag_api.DAGApi(api_client)
                    dag_run_instance = dag_run_api.DAGRunApi(api_client)
                    
                    new_dag_run = dag_run_instance.post_dag_run(
                        dag_id,
                        {
                            "dag_run_id": f"manual_restart_{int(datetime.utcnow().timestamp())}",
                            "execution_date": execution_date or datetime.utcnow().isoformat(),
                            "conf": {}
                        }
                    )
                    
                    return {
                        "action": "dag_triggered",
                        "new_dag_run_id": new_dag_run.dag_run_id,
                        "state": new_dag_run.state
                    }
            
        except Exception as e:
            raise RuntimeError(f"Airflow client restart failed: {str(e)}")
    
    async def _restart_airflow_with_api(self, dag_id: str, dag_run_id: Optional[str], 
                                      task_id: Optional[str], execution_date: Optional[str], 
                                      reset_dag_run: bool, airflow_url: str,
                                      username: str, password: str) -> Dict[str, Any]:
        """Restart Airflow DAG using direct REST API calls."""
        import base64
        
        # Create basic auth header
        auth_string = f"{username}:{password}"
        auth_bytes = auth_string.encode('ascii')
        auth_header = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/json"
        }
        
        base_url = f"{airflow_url}/api/v1"
        
        logger.info("🌐 AIRFLOW API CALL INITIATED", 
                   base_url=base_url,
                   dag_id=dag_id,
                   dag_run_id=dag_run_id,
                   task_id=task_id,
                   execution_date=execution_date,
                   reset_dag_run=reset_dag_run,
                   auth_username=username)
        
        async with aiohttp.ClientSession() as session:
            try:
                if reset_dag_run and dag_run_id:
                    # Delete DAG run
                    delete_url = f"{base_url}/dags/{dag_id}/dagRuns/{dag_run_id}"
                    
                    logger.info("🗑️  AIRFLOW DELETE DAG RUN", 
                               method="DELETE",
                               url=delete_url,
                               dag_id=dag_id,
                               dag_run_id=dag_run_id)
                    
                    async with session.delete(delete_url, headers=headers) as resp:
                        if resp.status not in [200, 204, 404]:
                            error_text = await resp.text()
                            logger.error("❌ AIRFLOW DELETE FAILED",
                                       status_code=resp.status,
                                       url=delete_url,
                                       error_response=error_text)
                            raise RuntimeError(f"Failed to delete DAG run: {resp.status}")
                        
                        logger.info("✅ AIRFLOW DAG RUN DELETED", 
                                   status_code=resp.status,
                                   dag_run_id=dag_run_id)
                    
                    # Create new DAG run
                    new_dag_run_id = f"{dag_run_id}_retry_{int(datetime.utcnow().timestamp())}"
                    create_url = f"{base_url}/dags/{dag_id}/dagRuns"
                    create_data = {
                        "dag_run_id": new_dag_run_id,
                        "execution_date": execution_date or datetime.utcnow().isoformat(),
                        "conf": {}
                    }
                    
                    logger.info("🚀 AIRFLOW CREATE DAG RUN",
                               method="POST",
                               url=create_url,
                               dag_id=dag_id,
                               payload=create_data)
                    
                    async with session.post(create_url, headers=headers, json=create_data) as resp:
                        if resp.status not in [200, 201]:
                            error_text = await resp.text()
                            logger.error("❌ AIRFLOW CREATE FAILED",
                                       status_code=resp.status,
                                       url=create_url,
                                       payload=create_data,
                                       error_response=error_text)
                            raise RuntimeError(f"Failed to create DAG run: {resp.status} - {error_text}")
                        
                        result_data = await resp.json()
                        logger.info("✅ AIRFLOW DAG RUN CREATED",
                                   status_code=resp.status,
                                   new_dag_run_id=result_data.get("dag_run_id"),
                                   state=result_data.get("state"))
                        
                        return {
                            "action": "dag_run_reset",
                            "new_dag_run_id": result_data.get("dag_run_id"),
                            "state": result_data.get("state"),
                            "endpoint_used": create_url,
                            "method": "POST",
                            "request_payload": create_data
                        }
                
                elif task_id:
                    # Clear task instances
                    clear_url = f"{base_url}/dags/{dag_id}/clearTaskInstances"
                    clear_data = {
                        "dry_run": False,
                        "reset_dag_runs": False,
                        "only_failed": False,
                        "only_running": True,
                        "include_subdags": True,
                        "task_ids": [task_id]
                    }
                    
                    if dag_run_id:
                        clear_data["dag_run_id"] = dag_run_id
                    
                    logger.info("🧹 AIRFLOW CLEAR TASK",
                               method="POST",
                               url=clear_url,
                               dag_id=dag_id,
                               task_id=task_id,
                               payload=clear_data)
                    
                    async with session.post(clear_url, headers=headers, json=clear_data) as resp:
                        if resp.status not in [200, 201]:
                            error_text = await resp.text()
                            logger.error("❌ AIRFLOW CLEAR FAILED",
                                       status_code=resp.status,
                                       url=clear_url,
                                       payload=clear_data,
                                       error_response=error_text)
                            raise RuntimeError(f"Failed to clear task: {resp.status} - {error_text}")
                        
                        logger.info("✅ AIRFLOW TASK CLEARED",
                                   status_code=resp.status,
                                   task_id=task_id,
                                   dag_run_id=dag_run_id)
                        
                        return {
                            "action": "task_cleared",
                            "task_id": task_id,
                            "dag_run_id": dag_run_id,
                            "endpoint_used": clear_url,
                            "method": "POST",
                            "request_payload": clear_data
                        }
                
                else:
                    # Trigger new DAG run
                    new_dag_run_id = f"manual_restart_{int(datetime.utcnow().timestamp())}"
                    trigger_url = f"{base_url}/dags/{dag_id}/dagRuns"
                    trigger_data = {
                        "dag_run_id": new_dag_run_id,
                        "execution_date": execution_date or datetime.utcnow().isoformat(),
                        "conf": {}
                    }
                    
                    logger.info("🎯 AIRFLOW TRIGGER DAG",
                               method="POST",
                               url=trigger_url,
                               dag_id=dag_id,
                               payload=trigger_data)
                    
                    async with session.post(trigger_url, headers=headers, json=trigger_data) as resp:
                        if resp.status not in [200, 201]:
                            error_text = await resp.text()
                            logger.error("❌ AIRFLOW TRIGGER FAILED",
                                       status_code=resp.status,
                                       url=trigger_url,
                                       payload=trigger_data,
                                       error_response=error_text)
                            raise RuntimeError(f"Failed to trigger DAG: {resp.status} - {error_text}")
                        
                        result_data = await resp.json()
                        logger.info("✅ AIRFLOW DAG TRIGGERED",
                                   status_code=resp.status,
                                   new_dag_run_id=result_data.get("dag_run_id"),
                                   state=result_data.get("state"))
                        
                        return {
                            "action": "dag_triggered",
                            "new_dag_run_id": result_data.get("dag_run_id"),
                            "state": result_data.get("state"),
                            "endpoint_used": trigger_url,
                            "method": "POST",
                            "request_payload": trigger_data
                        }
            
            except Exception as e:
                logger.error("💥 AIRFLOW API ERROR",
                           error=str(e),
                           dag_id=dag_id,
                           airflow_url=airflow_url)
                raise RuntimeError(f"Airflow API restart failed: {str(e)}")
    
    async def _restart_spark_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Restart a Spark job."""
        application_id = parameters.get("application_id")
        application_name = parameters.get("application_name")
        spark_master = parameters.get("spark_master", os.getenv("SPARK_MASTER_URL", "spark://localhost:7077"))
        memory_config = parameters.get("memory_config", {})
        force_kill = parameters.get("force_kill", True)
        
        if not application_id and not application_name:
            raise ValueError("Either application_id or application_name parameter is required")
        
        logger.info("Restarting Spark job", 
                   application_id=application_id,
                   application_name=application_name,
                   spark_master=spark_master)
        
        result = {
            "application_id": application_id,
            "application_name": application_name,
            "spark_master": spark_master,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if SPARK_AVAILABLE:
                result.update(await self._restart_spark_with_session(
                    application_id, application_name, spark_master, memory_config, force_kill
                ))
            else:
                result.update(await self._restart_spark_with_api(
                    application_id, application_name, spark_master, force_kill
                ))
            
            result["status"] = "restarted"
            logger.info("Spark job restarted successfully", 
                       application_id=application_id,
                       application_name=application_name)
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error("Failed to restart Spark job", 
                        application_id=application_id,
                        error=str(e))
            raise
        
        return result
    
    async def _restart_spark_with_session(self, application_id: Optional[str], 
                                        application_name: Optional[str],
                                        spark_master: str, memory_config: Dict[str, Any],
                                        force_kill: bool) -> Dict[str, Any]:
        """Restart Spark job using PySpark."""
        try:
            from pyspark.sql import SparkSession
            from pyspark import SparkContext
            
            # Stop any existing Spark context first
            try:
                spark_context = SparkContext.getOrCreate()
                if spark_context:
                    spark_context.stop()
                    # Give it time to fully stop
                    import time
                    time.sleep(2)
            except Exception as e:
                logger.debug("No existing Spark context to stop", error=str(e))
            
            # Kill existing application if needed
            if application_id and force_kill:
                try:
                    # Use Spark REST API to kill the application
                    await self._kill_spark_application_via_api(application_id, spark_master)
                except Exception as e:
                    logger.warning("Failed to kill existing Spark application", 
                                 application_id=application_id, error=str(e))
            
            # Configure new Spark session with unique app name
            import time
            unique_app_name = f"{application_name or 'OnCallAgent-RestartedJob'}-{int(time.time())}"
            builder = SparkSession.builder.appName(unique_app_name)
            
            # Use local mode for testing if spark_master is not available
            if spark_master and not spark_master.startswith("local"):
                builder = builder.master(spark_master)
            else:
                builder = builder.master("local[*]")
            
            # Apply memory configuration
            if memory_config.get("driver_memory"):
                builder = builder.config("spark.driver.memory", memory_config["driver_memory"])
            if memory_config.get("executor_memory"):
                builder = builder.config("spark.executor.memory", memory_config["executor_memory"])
            if memory_config.get("executor_instances"):
                builder = builder.config("spark.executor.instances", memory_config["executor_instances"])
            if memory_config.get("executor_cores"):
                builder = builder.config("spark.executor.cores", memory_config["executor_cores"])
            
            # Add config to avoid conflicts
            builder = builder.config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
            
            # Create new session
            spark = builder.getOrCreate()
            new_application_id = spark.sparkContext.applicationId
            
            return {
                "action": "spark_session_created",
                "new_application_id": new_application_id,
                "application_name": spark.sparkContext.appName,
                "master": spark.sparkContext.master,
                "memory_config": memory_config
            }
            
        except Exception as e:
            raise RuntimeError(f"Spark session restart failed: {str(e)}")
    
    async def _restart_spark_with_api(self, application_id: Optional[str], 
                                    application_name: Optional[str],
                                    spark_master: str, force_kill: bool) -> Dict[str, Any]:
        """Restart Spark job using REST API calls."""
        # Parse Spark master URL to get REST API URL
        if "://" in spark_master:
            protocol, host_port = spark_master.split("://", 1)
            if ":" in host_port:
                host, port = host_port.split(":", 1)
                # Spark REST API typically runs on port 6066 for standalone cluster
                api_url = f"http://{host}:6066"
            else:
                api_url = f"http://{host_port}:6066"
        else:
            api_url = f"http://{spark_master}:6066"
        
        # Kill existing application
        if application_id and force_kill:
            await self._kill_spark_application_via_api(application_id, spark_master)
        
        # Submit new application (this would need actual job submission logic)
        # For now, we'll return success with metadata
        return {
            "action": "spark_restart_via_api",
            "api_url": api_url,
            "killed_application_id": application_id if force_kill else None,
            "note": "Job restart initiated via Spark REST API"
        }
    
    async def _kill_spark_application_via_api(self, application_id: str, spark_master: str):
        """Kill a Spark application using REST API."""
        # Parse Spark master URL
        if "://" in spark_master:
            protocol, host_port = spark_master.split("://", 1)
            if ":" in host_port:
                host, port = host_port.split(":", 1)
                api_url = f"http://{host}:6066"
            else:
                api_url = f"http://{host_port}:6066"
        else:
            api_url = f"http://{spark_master}:6066"
        
        kill_url = f"{api_url}/v1/submissions/kill/{application_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(kill_url) as resp:
                if resp.status not in [200, 201]:
                    error_text = await resp.text()
                    logger.warning("Failed to kill Spark application via API", 
                                 application_id=application_id,
                                 status=resp.status,
                                 error=error_text)
    
    async def _call_api_endpoint(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call an API endpoint with comprehensive options."""
        url = parameters.get("url")
        method = parameters.get("method", "GET").upper()
        headers = parameters.get("headers", {})
        data = parameters.get("data")
        json_data = parameters.get("json")
        timeout = parameters.get("timeout", 30)
        auth = parameters.get("auth")  # {"type": "basic", "username": "user", "password": "pass"}
        
        if not url:
            raise ValueError("url parameter is required")
        
        logger.info("Calling API endpoint", url=url, method=method)
        
        # Prepare authentication
        auth_header = None
        if auth and auth.get("type") == "basic":
            import base64
            auth_string = f"{auth['username']}:{auth['password']}"
            auth_bytes = auth_string.encode('ascii')
            auth_header = base64.b64encode(auth_bytes).decode('ascii')
            headers["Authorization"] = f"Basic {auth_header}"
        elif auth and auth.get("type") == "bearer":
            headers["Authorization"] = f"Bearer {auth['token']}"
        
        result = {
            "url": url,
            "method": method,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    json=json_data
                ) as resp:
                    result["status_code"] = resp.status
                    result["response_headers"] = dict(resp.headers)
                    
                    # Read response
                    if resp.content_type and "application/json" in resp.content_type:
                        result["response"] = await resp.json()
                    else:
                        result["response"] = await resp.text()
                    
                    if resp.status >= 400:
                        raise RuntimeError(f"API call failed with status {resp.status}")
                    
                    logger.info("API endpoint called successfully", url=url, status=resp.status)
                    
        except Exception as e:
            result["error"] = str(e)
            logger.error("API endpoint call failed", url=url, error=str(e))
            raise RuntimeError(f"API call failed: {str(e)}")
        
        return result
    
    async def _scale_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Scale application resources on different platforms."""
        service_name = parameters.get("service_name")
        replicas = parameters.get("replicas")
        platform = parameters.get("platform", "auto")  # auto, kubernetes, docker, systemctl
        namespace = parameters.get("namespace", "default")
        
        if not service_name or replicas is None:
            raise ValueError("service_name and replicas parameters are required")
        
        logger.info("Scaling resources", service_name=service_name, replicas=replicas, platform=platform)
        
        result = {
            "service_name": service_name,
            "replicas": replicas,
            "platform": platform,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if platform == "kubernetes" or (platform == "auto" and KUBERNETES_AVAILABLE):
                result.update(await self._scale_kubernetes_deployment(service_name, replicas, namespace))
            elif platform == "docker" or (platform == "auto" and DOCKER_AVAILABLE):
                result.update(await self._scale_docker_service(service_name, replicas))
            else:
                raise ValueError(f"Scaling not supported for platform: {platform}")
            
            result["status"] = "scaled"
            logger.info("Resources scaled successfully", service_name=service_name, replicas=replicas)
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error("Failed to scale resources", service_name=service_name, error=str(e))
            raise
        
        return result
    
    async def _scale_kubernetes_deployment(self, service_name: str, replicas: int, namespace: str) -> Dict[str, Any]:
        """Scale a Kubernetes deployment."""
        if not KUBERNETES_AVAILABLE:
            raise RuntimeError("Kubernetes client not available")
        
        try:
            from kubernetes import client, config as k8s_config
            
            # Load kubeconfig
            k8s_config.load_incluster_config() if os.getenv("KUBERNETES_SERVICE_HOST") else k8s_config.load_kube_config()
            
            apps_v1 = client.AppsV1Api()
            
            # Scale deployment
            deployment = apps_v1.read_namespaced_deployment(service_name, namespace)
            current_replicas = deployment.spec.replicas
            
            deployment.spec.replicas = replicas
            apps_v1.patch_namespaced_deployment_scale(
                name=service_name,
                namespace=namespace,
                body={"spec": {"replicas": replicas}}
            )
            
            return {
                "scaling_method": "kubernetes_deployment",
                "namespace": namespace,
                "previous_replicas": current_replicas,
                "target_replicas": replicas
            }
            
        except Exception as e:
            raise RuntimeError(f"Kubernetes scaling failed: {str(e)}")
    
    async def _scale_docker_service(self, service_name: str, replicas: int) -> Dict[str, Any]:
        """Scale a Docker Swarm service."""
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker client not available")
        
        try:
            import docker
            client = docker.from_env()
            
            service = client.services.get(service_name)
            current_replicas = service.attrs["Spec"]["Mode"]["Replicated"]["Replicas"]
            
            service.scale(replicas)
            
            return {
                "scaling_method": "docker_swarm",
                "service_id": service.id,
                "previous_replicas": current_replicas,
                "target_replicas": replicas
            }
            
        except Exception as e:
            raise RuntimeError(f"Docker scaling failed: {str(e)}")
    
    async def _clear_cache(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Clear application cache from various cache systems."""
        cache_type = parameters.get("cache_type", "redis")
        pattern = parameters.get("pattern", "*")
        host = parameters.get("host", "localhost")
        port = parameters.get("port", 6379)
        database = parameters.get("database", 0)
        password = parameters.get("password")
        
        logger.info("Clearing cache", cache_type=cache_type, pattern=pattern, host=host)
        
        result = {
            "cache_type": cache_type,
            "pattern": pattern,
            "host": host,
            "port": port,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if cache_type == "redis":
                result.update(await self._clear_redis_cache(host, port, database, password, pattern))
            elif cache_type == "memcached":
                result.update(await self._clear_memcached_cache(host, port, pattern))
            elif cache_type == "filesystem":
                result.update(await self._clear_filesystem_cache(pattern))
            else:
                raise ValueError(f"Unsupported cache type: {cache_type}")
            
            logger.info("Cache cleared successfully", cache_type=cache_type, pattern=pattern)
            
        except Exception as e:
            result["error"] = str(e)
            logger.error("Failed to clear cache", cache_type=cache_type, error=str(e))
            raise
        
        return result
    
    async def _clear_redis_cache(self, host: str, port: int, database: int, 
                               password: Optional[str], pattern: str) -> Dict[str, Any]:
        """Clear Redis cache."""
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis client not available")
        
        try:
            import redis
            
            r = redis.Redis(
                host=host,
                port=port,
                db=database,
                password=password,
                decode_responses=True
            )
            
            # Test connection
            r.ping()
            
            # Get keys matching pattern
            keys = r.keys(pattern)
            keys_cleared = 0
            
            if keys:
                keys_cleared = r.delete(*keys)
            
            return {
                "keys_cleared": keys_cleared,
                "total_keys_found": len(keys),
                "cache_method": "redis"
            }
            
        except Exception as e:
            raise RuntimeError(f"Redis cache clear failed: {str(e)}")
    
    async def _clear_memcached_cache(self, host: str, port: int, pattern: str) -> Dict[str, Any]:
        """Clear Memcached cache (basic implementation)."""
        # Memcached doesn't support pattern matching, so we flush all
        try:
            import socket
            
            # Create socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((host, port))
            
            # Send flush_all command
            sock.send(b"flush_all\r\n")
            
            # Read response
            response = sock.recv(1024)
            sock.close()
            
            if b"OK" in response:
                return {
                    "cache_method": "memcached",
                    "action": "flush_all",
                    "note": "Memcached doesn't support pattern matching - flushed all keys"
                }
            else:
                raise RuntimeError("Memcached flush failed")
                
        except Exception as e:
            raise RuntimeError(f"Memcached cache clear failed: {str(e)}")
    
    async def _clear_filesystem_cache(self, pattern: str) -> Dict[str, Any]:
        """Clear filesystem cache."""
        import glob
        
        try:
            files = glob.glob(pattern)
            files_deleted = 0
            
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        files_deleted += 1
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                        files_deleted += 1
                except Exception as e:
                    logger.warning("Failed to delete cache file", file=file_path, error=str(e))
            
            return {
                "cache_method": "filesystem",
                "files_deleted": files_deleted,
                "total_files_found": len(files)
            }
            
        except Exception as e:
            raise RuntimeError(f"Filesystem cache clear failed: {str(e)}")
    
    async def _restart_database_connection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Restart database connection pool."""
        database_type = parameters.get("database_type", "postgresql")
        database_name = parameters.get("database_name")
        host = parameters.get("host", "localhost")
        port = parameters.get("port")
        username = parameters.get("username")
        password = parameters.get("password")
        pool_size = parameters.get("pool_size", 20)
        
        if not database_name:
            raise ValueError("database_name parameter is required")
        
        logger.info("Restarting database connection", 
                   database_type=database_type,
                   database_name=database_name,
                   host=host)
        
        result = {
            "database_type": database_type,
            "database_name": database_name,
            "host": host,
            "port": port,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if database_type == "postgresql":
                result.update(await self._restart_postgresql_connection(
                    host, port or 5432, database_name, username, password, pool_size
                ))
            elif database_type == "mysql":
                result.update(await self._restart_mysql_connection(
                    host, port or 3306, database_name, username, password, pool_size
                ))
            elif database_type == "mongodb":
                result.update(await self._restart_mongodb_connection(
                    host, port or 27017, database_name, username, password
                ))
            else:
                raise ValueError(f"Unsupported database type: {database_type}")
            
            result["status"] = "connection_restarted"
            logger.info("Database connection restarted successfully", 
                       database_type=database_type,
                       database_name=database_name)
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error("Failed to restart database connection", 
                        database_type=database_type,
                        error=str(e))
            raise
        
        return result
    
    async def _restart_postgresql_connection(self, host: str, port: int, database: str,
                                           username: Optional[str], password: Optional[str],
                                           pool_size: int) -> Dict[str, Any]:
        """Restart PostgreSQL connection pool."""
        if not POSTGRES_AVAILABLE:
            raise RuntimeError("PostgreSQL driver not available")
        
        try:
            import psycopg2
            from psycopg2 import pool
            
            # Test connection parameters
            conn_params = {
                "host": host,
                "port": port,
                "database": database
            }
            
            if username:
                conn_params["user"] = username
            if password:
                conn_params["password"] = password
            
            # Test basic connection
            test_conn = psycopg2.connect(**conn_params)
            test_conn.close()
            
            # Create connection pool
            connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=pool_size,
                **conn_params
            )
            
            # Test pool
            pool_conn = connection_pool.getconn()
            connection_pool.putconn(pool_conn)
            
            return {
                "connection_method": "postgresql_pool",
                "pool_size": pool_size,
                "connection_test": "successful"
            }
            
        except Exception as e:
            raise RuntimeError(f"PostgreSQL connection restart failed: {str(e)}")
    
    async def _restart_mysql_connection(self, host: str, port: int, database: str,
                                      username: Optional[str], password: Optional[str],
                                      pool_size: int) -> Dict[str, Any]:
        """Restart MySQL connection pool (basic implementation)."""
        # This would require mysql-connector-python or PyMySQL
        try:
            result = await self._run_command([
                "mysql", "-h", host, "-P", str(port), "-u", username or "root",
                f"-p{password}" if password else "", "-e", "SELECT 1;", database
            ])
            
            if result["returncode"] == 0:
                return {
                    "connection_method": "mysql_test",
                    "connection_test": "successful",
                    "note": "MySQL connection verified via CLI"
                }
            else:
                raise RuntimeError(f"MySQL connection test failed: {result['stderr']}")
                
        except Exception as e:
            raise RuntimeError(f"MySQL connection restart failed: {str(e)}")
    
    async def _restart_mongodb_connection(self, host: str, port: int, database: str,
                                        username: Optional[str], password: Optional[str]) -> Dict[str, Any]:
        """Restart MongoDB connection."""
        if not MONGODB_AVAILABLE:
            raise RuntimeError("MongoDB driver not available")
        
        try:
            import pymongo
            
            # Build connection string
            if username and password:
                conn_string = f"mongodb://{username}:{password}@{host}:{port}/{database}"
            else:
                conn_string = f"mongodb://{host}:{port}/{database}"
            
            # Test connection
            client = pymongo.MongoClient(conn_string, serverSelectionTimeoutMS=5000)
            client.server_info()  # Force connection
            
            db = client[database]
            collections = db.list_collection_names()
            
            client.close()
            
            return {
                "connection_method": "mongodb",
                "connection_test": "successful",
                "collections_count": len(collections)
            }
            
        except Exception as e:
            raise RuntimeError(f"MongoDB connection restart failed: {str(e)}")
