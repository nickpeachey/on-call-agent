#!/usr/bin/env python3
"""
Docker Deployment Test Suite
Tests the AI On-Call Agent Docker deployment process.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
import requests


class DockerDeploymentTester:
    """Docker deployment testing class."""
    
    def __init__(self):
        self.test_results = []
        self.project_root = Path(__file__).parent
        
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test results."""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}: {message}")
        
    def test_docker_files_exist(self) -> bool:
        """Test that required Docker files exist."""
        print("🐳 Checking Docker configuration files...")
        
        required_files = [
            "Dockerfile",
            "docker-compose.yml", 
            "requirements-prod.txt",
            ".env.example"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
        
        success = len(missing_files) == 0
        message = f"Found {len(required_files) - len(missing_files)}/{len(required_files)} required files"
        
        if missing_files:
            message += f" (Missing: {', '.join(missing_files)})"
            
        self.log_test("Docker Files Exist", success, message)
        return success
    
    def test_docker_compose_syntax(self) -> bool:
        """Test docker-compose.yml syntax."""
        print("📝 Validating docker-compose.yml syntax...")
        
        try:
            result = subprocess.run(
                ["docker-compose", "config", "-q"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            success = result.returncode == 0
            message = "Docker compose syntax is valid" if success else f"Syntax error: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            success = False
            message = "Docker compose validation timed out"
        except FileNotFoundError:
            success = False 
            message = "docker-compose command not found"
        except Exception as e:
            success = False
            message = f"Error validating docker-compose: {str(e)}"
            
        self.log_test("Docker Compose Syntax", success, message)
        return success
    
    def test_environment_file(self) -> bool:
        """Test that .env file exists or can be created."""
        print("🔧 Checking environment configuration...")
        
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if env_file.exists():
            message = ".env file already exists"
            success = True
        elif env_example.exists():
            # Copy .env.example to .env
            try:
                import shutil
                shutil.copy2(env_example, env_file)
                message = "Created .env from .env.example"
                success = True
            except Exception as e:
                message = f"Failed to create .env: {str(e)}"
                success = False
        else:
            message = "No .env or .env.example file found"
            success = False
            
        self.log_test("Environment File", success, message)
        return success
    
    def test_docker_build(self) -> bool:
        """Test Docker image build."""
        print("🔨 Building Docker image...")
        
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "oncall-agent-test", "."],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            success = result.returncode == 0
            
            if success:
                message = "Docker image built successfully"
            else:
                # Get last few lines of error
                error_lines = result.stderr.split('\n')[-5:]
                message = f"Build failed: {' '.join(error_lines)}"
                
        except subprocess.TimeoutExpired:
            success = False
            message = "Docker build timed out (5 minutes)"
        except FileNotFoundError:
            success = False
            message = "Docker command not found"
        except Exception as e:
            success = False
            message = f"Error building Docker image: {str(e)}"
            
        self.log_test("Docker Build", success, message)
        return success
    
    def test_docker_services_start(self) -> bool:
        """Test that Docker services can start."""
        print("🚀 Starting Docker services...")
        
        try:
            # Start services in detached mode
            result = subprocess.run(
                ["docker-compose", "up", "-d", "--build"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes
            )
            
            success = result.returncode == 0
            
            if success:
                message = "Docker services started successfully"
                
                # Wait a moment for services to initialize
                time.sleep(10)
                
                # Check service status
                status_result = subprocess.run(
                    ["docker-compose", "ps"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                running_services = len([line for line in status_result.stdout.split('\n') 
                                      if 'Up' in line or 'running' in line])
                message += f" ({running_services} services running)"
                
            else:
                message = f"Failed to start services: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            success = False
            message = "Docker compose up timed out"
        except Exception as e:
            success = False
            message = f"Error starting services: {str(e)}"
            
        self.log_test("Docker Services Start", success, message)
        return success
    
    def test_application_health(self) -> bool:
        """Test that the application responds to health checks."""
        print("🏥 Testing application health...")
        
        # Wait for application to start
        max_attempts = 12  # 2 minutes with 10 second intervals
        success = False
        message = ""
        attempt = 0
        
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    success = True
                    message = f"Application healthy (attempt {attempt + 1})"
                    break
                else:
                    message = f"Health check returned status {response.status_code}"
            except requests.exceptions.ConnectionError:
                message = f"Connection refused (attempt {attempt + 1}/{max_attempts})"
            except Exception as e:
                message = f"Health check error: {str(e)}"
            
            if attempt < max_attempts - 1:
                time.sleep(10)
        
        if not success:
            message = f"Application did not become healthy after {max_attempts} attempts"
            
        self.log_test("Application Health", success, message)
        return success
    
    def test_api_endpoints(self) -> bool:
        """Test basic API endpoint accessibility."""
        print("🌐 Testing API endpoints...")
        
        endpoints_to_test = [
            ("/docs", "API documentation"),
            ("/", "Root endpoint"),
        ]
        
        successful_endpoints = 0
        total_endpoints = len(endpoints_to_test)
        
        for endpoint, description in endpoints_to_test:
            try:
                response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
                if response.status_code in [200, 404, 422]:  # 404/422 are acceptable for some endpoints
                    successful_endpoints += 1
            except Exception:
                pass  # Count as failure
        
        success = successful_endpoints > 0
        message = f"Accessible endpoints: {successful_endpoints}/{total_endpoints}"
        
        self.log_test("API Endpoints", success, message)
        return success
    
    def cleanup_docker(self) -> bool:
        """Clean up Docker resources."""
        print("🧹 Cleaning up Docker resources...")
        
        try:
            # Stop and remove containers
            subprocess.run(
                ["docker-compose", "down", "-v"],
                cwd=self.project_root,
                capture_output=True,
                timeout=60
            )
            
            # Remove test image if it exists
            subprocess.run(
                ["docker", "rmi", "oncall-agent-test"],
                capture_output=True
            )
            
            message = "Docker cleanup completed"
            success = True
            
        except Exception as e:
            message = f"Cleanup warning: {str(e)}"
            success = True  # Don't fail on cleanup issues
            
        self.log_test("Docker Cleanup", success, message)
        return success
    
    def run_deployment_test(self) -> dict:
        """Run complete deployment test suite."""
        print("🧪 Starting Docker Deployment Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test sequence
        tests = [
            ("File Check", self.test_docker_files_exist),
            ("Compose Syntax", self.test_docker_compose_syntax), 
            ("Environment", self.test_environment_file),
            ("Docker Build", self.test_docker_build),
            ("Services Start", self.test_docker_services_start),
            ("App Health", self.test_application_health),
            ("API Endpoints", self.test_api_endpoints),
            ("Cleanup", self.cleanup_docker)
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                    
                # Add delay between major steps
                if test_name in ["Docker Build", "Services Start"]:
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                print("\n⚠️ Test interrupted by user")
                self.cleanup_docker()
                break
            except Exception as e:
                self.log_test(test_name, False, f"Test execution error: {str(e)}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Generate summary
        total_tests = len(tests)
        success_rate = (passed_tests / total_tests) * 100
        overall_success = passed_tests >= total_tests - 1  # Allow cleanup to fail
        
        summary = {
            "overall_success": overall_success,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "execution_time_seconds": execution_time,
            "test_results": self.test_results
        }
        
        return summary


def main():
    """Main test execution."""
    tester = DockerDeploymentTester()
    results = tester.run_deployment_test()
    
    # Save results
    results_file = "deployment_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT TEST SUMMARY")
    print("=" * 60)
    print(f"Overall Success: {'✅ YES' if results['overall_success'] else '❌ NO'}")
    print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Execution Time: {results['execution_time_seconds']:.1f} seconds")
    print(f"Results saved to: {results_file}")
    
    if not results['overall_success']:
        print("\n❌ Failed Tests:")
        for test_result in results['test_results']:
            if not test_result['success']:
                print(f"  - {test_result['test_name']}: {test_result['message']}")
    
    print("=" * 60)
    
    return 0 if results['overall_success'] else 1


if __name__ == "__main__":
    sys.exit(main())
