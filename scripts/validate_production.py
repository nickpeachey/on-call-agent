#!/usr/bin/env python3
"""
Production validation script for AI On-Call Agent.
Tests all action implementations and integrations.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.actions import ActionService
from src.models.schemas import ActionType, Severity


class ProductionValidator:
    """Comprehensive production validation suite."""
    
    def __init__(self):
        self.action_service = ActionService()
        self.test_results = []
        self.failed_tests = []
        
    async def run_all_tests(self):
        """Run comprehensive production validation tests."""
        print("🧪 AI On-Call Agent Production Validation")
        print("=" * 50)
        
        await self.action_service.start()
        
        try:
            # Test each action type
            await self.test_service_restart()
            await self.test_airflow_integration()
            await self.test_spark_integration()
            await self.test_api_endpoint_calls()
            await self.test_resource_scaling()
            await self.test_cache_clearing()
            await self.test_database_connections()
            
            # Integration tests
            await self.test_error_handling()
            await self.test_concurrent_actions()
            
            # Generate report
            self.generate_report()
            
        finally:
            await self.action_service.stop()
    
    async def test_service_restart(self):
        """Test service restart functionality."""
        print("\\n🔄 Testing Service Restart")
        print("-" * 30)
        
        test_cases = [
            {
                "name": "Docker Container Restart",
                "parameters": {
                    "service_name": "test-container",
                    "platform": "docker"
                }
            },
            {
                "name": "Kubernetes Deployment Restart", 
                "parameters": {
                    "service_name": "test-deployment",
                    "platform": "kubernetes",
                    "namespace": "default"
                }
            },
            {
                "name": "Systemctl Service Restart",
                "parameters": {
                    "service_name": "cron",
                    "platform": "systemctl"
                }
            }
        ]
        
        for test_case in test_cases:
            await self._run_action_test(
                "restart_service",
                test_case["name"],
                test_case["parameters"]
            )
    
    async def test_airflow_integration(self):
        """Test Airflow DAG restart functionality."""
        print("\\n✈️ Testing Airflow Integration")
        print("-" * 30)
        
        test_cases = [
            {
                "name": "DAG Task Restart",
                "parameters": {
                    "dag_id": "test_pipeline",
                    "dag_run_id": "dag_run_20240730_120000",
                    "task_id": "transform_data",
                    "reset_dag_run": False
                }
            },
            {
                "name": "Full DAG Reset",
                "parameters": {
                    "dag_id": "test_pipeline",
                    "dag_run_id": "dag_run_20240730_120000",
                    "reset_dag_run": True
                }
            },
            {
                "name": "DAG Trigger",
                "parameters": {
                    "dag_id": "test_pipeline",
                    "execution_date": "2024-07-30T12:00:00Z"
                }
            }
        ]
        
        for test_case in test_cases:
            await self._run_action_test(
                "restart_airflow_dag",
                test_case["name"],
                test_case["parameters"]
            )
    
    async def test_spark_integration(self):
        """Test Spark job restart functionality."""
        print("\\n⚡ Testing Spark Integration")
        print("-" * 30)
        
        test_cases = [
            {
                "name": "Spark Application Restart",
                "parameters": {
                    "application_id": "app-20240730120000-0001",
                    "application_name": "test-etl-job",
                    "force_kill": True,
                    "memory_config": {
                        "driver_memory": "4g",
                        "executor_memory": "8g",
                        "executor_instances": "4"
                    }
                }
            },
            {
                "name": "Spark Session Creation",
                "parameters": {
                    "application_name": "oncall-test-job",
                    "memory_config": {
                        "driver_memory": "2g",
                        "executor_memory": "4g"
                    }
                }
            }
        ]
        
        for test_case in test_cases:
            await self._run_action_test(
                "restart_spark_job",
                test_case["name"],
                test_case["parameters"]
            )
    
    async def test_api_endpoint_calls(self):
        """Test API endpoint calling functionality."""
        print("\\n🌐 Testing API Endpoint Calls")
        print("-" * 30)
        
        test_cases = [
            {
                "name": "GET Request",
                "parameters": {
                    "url": "https://httpbin.org/get",
                    "method": "GET",
                    "timeout": 10
                }
            },
            {
                "name": "POST Request with JSON",
                "parameters": {
                    "url": "https://httpbin.org/post",
                    "method": "POST",
                    "json": {"test": "data"},
                    "timeout": 10
                }
            },
            {
                "name": "Authenticated Request",
                "parameters": {
                    "url": "https://httpbin.org/basic-auth/user/pass",
                    "method": "GET",
                    "auth": {
                        "type": "basic",
                        "username": "user",
                        "password": "pass"
                    }
                }
            }
        ]
        
        for test_case in test_cases:
            await self._run_action_test(
                "call_api_endpoint",
                test_case["name"],
                test_case["parameters"]
            )
    
    async def test_resource_scaling(self):
        """Test resource scaling functionality."""
        print("\\n📈 Testing Resource Scaling")
        print("-" * 30)
        
        test_cases = [
            {
                "name": "Kubernetes Deployment Scaling",
                "parameters": {
                    "service_name": "test-deployment",
                    "replicas": 5,
                    "platform": "kubernetes",
                    "namespace": "default"
                }
            },
            {
                "name": "Docker Swarm Service Scaling",
                "parameters": {
                    "service_name": "test-service",
                    "replicas": 3,
                    "platform": "docker"
                }
            }
        ]
        
        for test_case in test_cases:
            await self._run_action_test(
                "scale_resources",
                test_case["name"],
                test_case["parameters"]
            )
    
    async def test_cache_clearing(self):
        """Test cache clearing functionality."""
        print("\\n🗑️ Testing Cache Clearing")
        print("-" * 30)
        
        test_cases = [
            {
                "name": "Redis Cache Clear",
                "parameters": {
                    "cache_type": "redis",
                    "host": "localhost",
                    "port": 6379,
                    "pattern": "test:*"
                }
            },
            {
                "name": "Filesystem Cache Clear",
                "parameters": {
                    "cache_type": "filesystem",
                    "pattern": "/tmp/test_cache_*"
                }
            }
        ]
        
        for test_case in test_cases:
            await self._run_action_test(
                "clear_cache",
                test_case["name"],
                test_case["parameters"]
            )
    
    async def test_database_connections(self):
        """Test database connection restart functionality."""
        print("\\n🗄️ Testing Database Connections")
        print("-" * 30)
        
        test_cases = [
            {
                "name": "PostgreSQL Connection Test",
                "parameters": {
                    "database_type": "postgresql",
                    "database_name": "test_db",
                    "host": "localhost",
                    "port": 5432,
                    "username": "test_user",
                    "pool_size": 10
                }
            },
            {
                "name": "MongoDB Connection Test",
                "parameters": {
                    "database_type": "mongodb",
                    "database_name": "test_db",
                    "host": "localhost",
                    "port": 27017
                }
            }
        ]
        
        for test_case in test_cases:
            await self._run_action_test(
                "restart_database_connection",
                test_case["name"],
                test_case["parameters"]
            )
    
    async def test_error_handling(self):
        """Test error handling and resilience."""
        print("\\n⚠️ Testing Error Handling")
        print("-" * 30)
        
        test_cases = [
            {
                "name": "Invalid Service Name",
                "action_type": "restart_service",
                "parameters": {
                    "service_name": "non-existent-service",
                    "platform": "docker"
                },
                "expect_failure": True
            },
            {
                "name": "Invalid DAG ID",
                "action_type": "restart_airflow_dag",
                "parameters": {
                    "dag_id": "non-existent-dag"
                },
                "expect_failure": True
            },
            {
                "name": "Invalid URL",
                "action_type": "call_api_endpoint",
                "parameters": {
                    "url": "https://invalid-domain-12345.com",
                    "timeout": 5
                },
                "expect_failure": True
            }
        ]
        
        for test_case in test_cases:
            await self._run_action_test(
                test_case["action_type"],
                test_case["name"],
                test_case["parameters"],
                expect_failure=test_case.get("expect_failure", False)
            )
    
    async def test_concurrent_actions(self):
        """Test concurrent action execution."""
        print("\\n🔀 Testing Concurrent Actions")
        print("-" * 30)
        
        # Create multiple actions to run concurrently
        concurrent_actions = []
        for i in range(5):
            action_id = await self.action_service.execute_action(
                action_type="call_api_endpoint",
                parameters={
                    "url": f"https://httpbin.org/delay/1",
                    "method": "GET"
                },
                timeout_seconds=30
            )
            concurrent_actions.append(action_id)
        
        # Wait for all actions to complete
        start_time = asyncio.get_event_loop().time()
        completed_actions = []
        
        while len(completed_actions) < len(concurrent_actions) and (asyncio.get_event_loop().time() - start_time) < 60:
            for action_id in concurrent_actions:
                if action_id not in completed_actions:
                    action = await self.action_service.get_action(action_id)
                    if action and action.status in ["success", "failed"]:
                        completed_actions.append(action_id)
            
            await asyncio.sleep(0.5)
        
        success = len(completed_actions) == len(concurrent_actions)
        print(f"   {'✅' if success else '❌'} Concurrent Actions: {len(completed_actions)}/{len(concurrent_actions)} completed")
        
        self.test_results.append({
            "test": "Concurrent Actions",
            "success": success,
            "details": f"{len(completed_actions)}/{len(concurrent_actions)} actions completed"
        })
    
    async def _run_action_test(self, action_type: str, test_name: str, 
                             parameters: Dict[str, Any], expect_failure: bool = False):
        """Run a single action test."""
        try:
            action_id = await self.action_service.execute_action(
                action_type=action_type,
                parameters=parameters,
                timeout_seconds=30
            )
            
            # Wait for action to complete
            max_wait = 30  # seconds
            start_time = asyncio.get_event_loop().time()
            
            while (asyncio.get_event_loop().time() - start_time) < max_wait:
                action = await self.action_service.get_action(action_id)
                if action and action.status in ["success", "failed"]:
                    break
                await asyncio.sleep(0.5)
            
            action = await self.action_service.get_action(action_id)
            
            if not action:
                success = False
                details = "Action not found"
            elif expect_failure:
                success = action.status == "failed"
                details = f"Expected failure: {action.status}"
            else:
                success = action.status == "success"
                details = f"Status: {action.status}"
                if action.error_message:
                    details += f", Error: {action.error_message}"
            
            status_emoji = "✅" if success else "❌"
            print(f"   {status_emoji} {test_name}: {details}")
            
            self.test_results.append({
                "test": test_name,
                "action_type": action_type,
                "success": success,
                "status": action.status if action else "unknown",
                "details": details
            })
            
            if not success and not expect_failure:
                self.failed_tests.append(test_name)
            
        except Exception as e:
            success = expect_failure  # If we expected failure, an exception is OK
            details = f"Exception: {str(e)}"
            status_emoji = "✅" if success else "❌"
            print(f"   {status_emoji} {test_name}: {details}")
            
            self.test_results.append({
                "test": test_name,
                "action_type": action_type,
                "success": success,
                "details": details
            })
            
            if not success:
                self.failed_tests.append(test_name)
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\\n📊 Production Validation Report")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r["success"]])
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests} ✅")
        print(f"Failed: {failed_tests} ❌") 
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests:
            print(f"\\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   - {test}")
        
        # Save detailed report
        report_file = Path("validation_report.json")
        with open(report_file, "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "success_rate": success_rate
                },
                "test_results": self.test_results,
                "failed_tests": self.failed_tests
            }, f, indent=2)
        
        print(f"\\n📄 Detailed report saved to: {report_file}")
        
        # Overall result
        if success_rate >= 80:
            print("\\n🎉 Production validation PASSED! System is ready for deployment.")
            return True
        else:
            print("\\n⚠️ Production validation FAILED! Please review failed tests before deployment.")
            return False


async def main():
    """Run production validation."""
    validator = ProductionValidator()
    success = await validator.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
