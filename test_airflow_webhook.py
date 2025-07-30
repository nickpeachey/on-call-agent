#!/usr/bin/env python3
"""
Test script for the Airflow webhook incident creation system.
"""

import asyncio
import aiohttp
import json
from datetime import datetime


async def test_airflow_webhook():
    """Test the Airflow webhook endpoint."""
    
    # Sample Airflow webhook payload
    webhook_payload = {
        "dag_id": "data_pipeline_dag",
        "task_id": "extract_customer_data",
        "execution_date": "2025-07-30T10:00:00Z",
        "state": "failed",
        "log_url": "http://airflow.local/admin/airflow/log?dag_id=data_pipeline_dag&task_id=extract_customer_data",
        "error_message": "Database connection timeout after 30 seconds. Unable to connect to customer_db on host db.example.com:5432",
        "context": {
            "environment": "production",
            "cluster": "spark-cluster-prod",
            "database": "customer_db",
            "retry_count": 3
        }
    }
    
    # Also test the main incident creation endpoint
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
        
        print("🚨 Testing Airflow Webhook Endpoint...")
        print(f"📤 Payload: {json.dumps(webhook_payload, indent=2)}")
        
        try:
            # Test webhook endpoint
            async with session.post(
                f"{base_url}/enhanced-incidents/webhook/airflow",
                json=webhook_payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                status = resp.status
                response_text = await resp.text()
                
                print(f"📥 Webhook Response: {status}")
                print(f"📋 Response Body: {response_text}")
                
                if status == 200:
                    response_data = await resp.json() if resp.content_type == 'application/json' else None
                    if response_data:
                        print(f"✅ Incident ID: {response_data.get('incident_id')}")
                        print(f"🤖 Actions Triggered: {response_data.get('action_ids', [])}")
                        print(f"🔄 Auto Resolution: {response_data.get('auto_resolution_attempted', False)}")
                else:
                    print(f"❌ Webhook failed with status {status}")
        
        except Exception as e:
            print(f"❌ Webhook test failed: {str(e)}")
        
        print("\n" + "="*60 + "\n")
        
        print("📝 Testing Direct Incident Creation...")
        print(f"📤 Payload: {json.dumps(incident_payload, indent=2)}")
        
        try:
            # Test direct incident creation
            async with session.post(
                f"{base_url}/enhanced-incidents/",
                json=incident_payload,
                headers={"Content-Type": "application/json"},
                params={"auto_resolve": "true"}  # aiohttp requires string, not boolean
            ) as resp:
                status = resp.status
                response_text = await resp.text()
                
                print(f"📥 Direct Creation Response: {status}")
                print(f"📋 Response Body: {response_text}")
                
                if status == 200:
                    response_data = await resp.json() if resp.content_type == 'application/json' else None
                    if response_data:
                        incident_data = response_data.get('incident', {})
                        print(f"✅ Incident ID: {incident_data.get('id')}")
                        print(f"🎯 AI Confidence: {incident_data.get('ai_confidence')}")
                        print(f"⏱️  Predicted Time: {incident_data.get('predicted_resolution_time')} minutes")
                        print(f"🤖 Actions Triggered: {response_data.get('action_ids', [])}")
                else:
                    print(f"❌ Direct creation failed with status {status}")
        
        except Exception as e:
            print(f"❌ Direct creation test failed: {str(e)}")
        
        print("\n" + "="*60 + "\n")
        
        # Test getting incidents list
        print("📋 Testing Incident List...")
        try:
            async with session.get(f"{base_url}/enhanced-incidents/") as resp:
                status = resp.status
                response_text = await resp.text()
                
                print(f"📥 List Response: {status}")
                if status == 200:
                    print("✅ Incident list retrieved successfully")
                else:
                    print(f"📋 Response: {response_text}")
        
        except Exception as e:
            print(f"❌ List test failed: {str(e)}")


if __name__ == "__main__":
    print("🔧 Airflow Incident Management Test")
    print("="*60)
    asyncio.run(test_airflow_webhook())
