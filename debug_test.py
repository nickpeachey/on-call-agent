#!/usr/bin/env python3
"""
Simple debug test to isolate the validation error
"""
import asyncio
import aiohttp
import json


async def test_incident_creation():
    incident_payload = {
        "title": "ETL Pipeline Database Connection Failure",
        "description": "Customer data extraction pipeline failed due to database connection timeout. This is affecting real-time analytics dashboard.",
        "severity": "critical",
        "category": "database",
        "source": "airflow",
        "external_id": "data_pipeline_dag_extract_customer_data_2025-07-30T10:00:00Z",
        "metadata": {
            "dag_id": "data_pipeline_dag",
            "task_id": "extract_customer_data",
            "database_host": "db.example.com",
            "timeout_seconds": 30,
            "retry_attempts": 3
        }
    }
    
    base_url = "http://localhost:8000/api/v1"
    
    async with aiohttp.ClientSession() as session:
        print("🔍 Testing direct incident creation endpoint...")
        
        try:
            # Test WITHOUT auto_resolve parameter
            print("1. Testing without auto_resolve parameter...")
            async with session.post(
                f"{base_url}/enhanced-incidents/",
                json=incident_payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                status = resp.status
                response_text = await resp.text()
                print(f"   Status: {status}")
                if status != 200:
                    print(f"   Error: {response_text}")
                else:
                    print("   ✅ Success!")
                    
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            
        try:
            # Test WITH auto_resolve parameter as string
            print("2. Testing with auto_resolve='true' (string)...")
            async with session.post(
                f"{base_url}/enhanced-incidents/?auto_resolve=true",
                json=incident_payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                status = resp.status
                response_text = await resp.text()
                print(f"   Status: {status}")
                if status != 200:
                    print(f"   Error: {response_text}")
                else:
                    print("   ✅ Success!")
                    
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            
        try:
            # Test WITH auto_resolve parameter as boolean (this might be the issue)
            print("3. Testing with auto_resolve=True (boolean param)...")
            async with session.post(
                f"{base_url}/enhanced-incidents/",
                json=incident_payload,
                headers={"Content-Type": "application/json"},
                params={"auto_resolve": True}
            ) as resp:
                status = resp.status
                response_text = await resp.text()
                print(f"   Status: {status}")
                if status != 200:
                    print(f"   Error: {response_text}")
                else:
                    print("   ✅ Success!")
                    
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            print(f"   This is likely the issue in the original test!")


if __name__ == "__main__":
    asyncio.run(test_incident_creation())
