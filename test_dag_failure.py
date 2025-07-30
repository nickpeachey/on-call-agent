#!/usr/bin/env python3
"""
Test script to trigger an Airflow DAG failure incident.
This will test our enhanced action logging with real HTTP endpoints.
"""

import asyncio
import aiohttp
import json
from datetime import datetime


async def trigger_dag_failure():
    """Trigger an Airflow DAG restart action to test the enhanced logging."""
    
    # Execute an Airflow DAG restart action directly
    action_data = {
        "action_type": "restart_airflow_dag",
        "parameters": {
            "dag_id": "etl_data_pipeline",
            "dag_run_id": f"scheduled__{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')}",
            "task_id": "extract_customer_data",
            "execution_date": datetime.utcnow().isoformat(),
            "reset_dag_run": True
        }
    }
    
    print("🔥 Triggering Airflow DAG restart action...")
    print(f"🎯 DAG ID: {action_data['parameters']['dag_id']}")
    print(f"� DAG Run ID: {action_data['parameters']['dag_run_id']}")
    print(f"🔗 Task ID: {action_data['parameters']['task_id']}")
    print(f"📅 Execution Date: {action_data['parameters']['execution_date']}")
    print(f"🔄 Reset DAG Run: {action_data['parameters']['reset_dag_run']}")
    
    try:
        # Execute the action via the on-call agent API
        async with aiohttp.ClientSession() as session:
            # Prepare the URL with action_type as query parameter
            url = f"http://localhost:8000/api/v1/actions/execute?action_type={action_data['action_type']}"
            
            async with session.post(
                url,
                json=action_data['parameters'],
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status in [200, 201]:
                    result = await response.json()
                    print(f"✅ Action execution started successfully!")
                    print(f"🆔 Action ID: {result.get('action_id')}")
                    print("\n🎬 This should trigger detailed logging:")
                    print("  🌐 Airflow API calls with HTTP method/endpoint logging")
                    print("  🗑️ DELETE /api/v1/dags/{dag_id}/dagRuns/{dag_run_id}")
                    print("  🚀 POST /api/v1/dags/{dag_id}/dagRuns (new run)")
                    print("  🧹 POST /api/v1/dags/{dag_id}/clearTaskInstances (if task_id)")
                    print("  📝 Complete payload and response logging")
                    print("  🎯 Endpoint URLs, status codes, and error handling")
                    
                    return result.get('action_id')
                    
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to execute action: {response.status}")
                    print(f"📄 Error: {error_text}")
                    
    except Exception as e:
        print(f"💥 Error executing action: {str(e)}")
        print("ℹ️  Make sure the on-call agent is running on http://localhost:8000")
        
    return None


async def monitor_action_logs(action_id):
    """Monitor the action execution and show the logs."""
    print(f"\n🔍 Monitoring action {action_id}...")
    
    for i in range(30):  # Monitor for 30 seconds
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:8000/api/v1/actions/{action_id}") as response:
                    if response.status == 200:
                        action_data = await response.json()
                        status = action_data.get('status')
                        
                        print(f"📊 Action Status: {status}")
                        
                        if status in ['success', 'failed']:
                            print("🏁 Action completed!")
                            if action_data.get('result'):
                                result = action_data['result']
                                print(f"📋 Result: {json.dumps(result, indent=2)}")
                            if action_data.get('error_message'):
                                print(f"❌ Error: {action_data['error_message']}")
                            break
                            
                        elif status == 'running':
                            print("🔄 Action is running...")
                            
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"⚠️  Error monitoring action: {str(e)}")
            await asyncio.sleep(1)
    
    print("\n📋 Check the Docker logs for detailed HTTP logging:")
    print("docker-compose logs --tail=50 oncall-agent | grep -E '(🌐|🗑️|🚀|🧹|🎯|✅|❌|AIRFLOW)'")


if __name__ == "__main__":
    print("🚁 Airflow DAG Failure Test Script")
    print("=" * 50)
    
    async def main():
        action_id = await trigger_dag_failure()
        if action_id:
            await monitor_action_logs(action_id)
        else:
            print("❌ Failed to start action")
    
    asyncio.run(main())
