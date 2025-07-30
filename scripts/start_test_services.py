#!/usr/bin/env python3
"""
Start minimal test services for validation.
"""

import asyncio
import subprocess
import time
import os
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def start_redis_server():
    """Start a Redis server for testing."""
    try:
        # Check if Redis is already running
        result = subprocess.run(
            ["redis-cli", "ping"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Redis already running")
            return None
        
        # Start Redis server
        print("🚀 Starting Redis server...")
        process = subprocess.Popen(
            ["redis-server", "--port", "6379", "--daemonize", "yes"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for Redis to start
        for i in range(10):
            try:
                result = subprocess.run(
                    ["redis-cli", "ping"], 
                    capture_output=True, 
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    print("✅ Redis server started successfully")
                    return process
            except:
                pass
            time.sleep(1)
        
        print("❌ Failed to start Redis server")
        return None
        
    except FileNotFoundError:
        print("⚠️  Redis not installed - cache tests will fail")
        return None
    except Exception as e:
        print(f"⚠️  Redis start failed: {e}")
        return None

async def create_test_files():
    """Create test cache files."""
    import tempfile
    test_files = []
    
    for i in range(3):
        temp_file = f"/tmp/test_cache_{i}.txt"
        with open(temp_file, "w") as f:
            f.write(f"Test cache file {i}")
        test_files.append(temp_file)
    
    print(f"✅ Created {len(test_files)} test cache files")
    return test_files

async def setup_mock_services():
    """Set up environment variables for mock services."""
    # Set up environment for testing
    os.environ["AIRFLOW_URL"] = "http://localhost:8080"
    os.environ["AIRFLOW_USERNAME"] = "admin" 
    os.environ["AIRFLOW_PASSWORD"] = "admin"
    os.environ["SPARK_MASTER_URL"] = "local[*]"
    
    print("✅ Environment variables configured")

async def main():
    """Start test services and run validation."""
    print("🧪 Setting up test environment...")
    
    # Setup mock services
    await setup_mock_services()
    
    # Create test files
    test_files = await create_test_files()
    
    # Start Redis
    redis_process = await start_redis_server()
    
    print("\n🎯 Test environment ready!")
    print("Now run: python scripts/validate_production.py")
    
    # Keep services running
    try:
        input("\nPress Enter to stop test services...")
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    
    # Remove test files
    for file_path in test_files:
        try:
            os.remove(file_path)
        except:
            pass
    
    print("✅ Test environment cleaned up")

if __name__ == "__main__":
    asyncio.run(main())
