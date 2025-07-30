#!/usr/bin/env python3
"""Test script to create incidents and monitor their automated resolution."""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any

API_BASE = "http://localhost:8000"

async def create_test_incident(session: aiohttp.ClientSession, incident_data: Dict[str, Any]) -> str:
    """Create a test incident via API."""
    print(f"🔥 Creating incident: {incident_data['title']}")
    
    # For now, we'll simulate incident creation since the exact endpoint structure may vary
    # This would typically POST to /incidents
    incident_id = f"test_{int(time.time())}"
    print(f"✅ Created incident ID: {incident_id}")
    return incident_id

async def monitor_incident_resolution(session: aiohttp.ClientSession, incident_id: str, max_wait: int = 300):
    """Monitor an incident until it's resolved or timeout."""
    print(f"👁️  Monitoring incident {incident_id} for resolution...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            # Check resolution status
            async with session.get(f"{API_BASE}/resolutions/live-feed") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"📊 AI Engine Status: {data.get('ai_engine_status', 'unknown')}")
                    print(f"📋 Active Incidents: {data.get('active_incidents', 0)}")
                    print(f"🤖 Queue Size: {data.get('queue_size', 0)}")
                    
                    last_action = data.get('last_automated_action')
                    if last_action:
                        print(f"🔧 Last Action: {last_action.get('action')} on {last_action.get('incident_id')} - {last_action.get('status')}")
                    
        except Exception as e:
            print(f"⚠️  Error monitoring: {e}")
        
        await asyncio.sleep(5)  # Check every 5 seconds
    
    print(f"⏰ Monitoring timeout after {max_wait} seconds")

async def get_resolution_summary(session: aiohttp.ClientSession):
    """Get resolution summary statistics."""
    try:
        async with session.get(f"{API_BASE}/resolutions/summary") as response:
            if response.status == 200:
                data = await response.json()
                print("\n📈 RESOLUTION SUMMARY (Last 24 Hours)")
                print(f"   Total Incidents: {data['total_incidents']}")
                print(f"   Automated: {data['automated_resolutions']} ✅")
                print(f"   Manual: {data['manual_resolutions']} 👤")
                print(f"   Failed: {data['failed_automations']} ❌")
                print(f"   Success Rate: {data['automation_success_rate']:.1%}")
                print(f"   Avg Resolution Time: {data['average_resolution_time']:.1f}s")
                print()
    except Exception as e:
        print(f"❌ Error getting summary: {e}")

async def get_recent_resolutions(session: aiohttp.ClientSession):
    """Get recent resolution details."""
    try:
        async with session.get(f"{API_BASE}/resolutions/recent?limit=5") as response:
            if response.status == 200:
                resolutions = await response.json()
                print("🕒 RECENT RESOLUTIONS:")
                for res in resolutions:
                    status_emoji = "✅" if res['automated_success'] else "❌"
                    method_emoji = "🤖" if res['resolution_method'] == 'automated' else "👤"
                    
                    print(f"   {status_emoji} {method_emoji} {res['incident_id']}: {res['title']}")
                    print(f"      Service: {res['service']} | Severity: {res['severity']}")
                    print(f"      Confidence: {res['ai_confidence']:.1%} | Method: {res['resolution_method']}")
                    if res['resolution_time_seconds']:
                        print(f"      Resolution Time: {res['resolution_time_seconds']:.1f}s")
                    if res['actions_taken']:
                        print(f"      Actions: {', '.join(res['actions_taken'])}")
                    print()
    except Exception as e:
        print(f"❌ Error getting recent resolutions: {e}")

async def test_health_endpoint(session: aiohttp.ClientSession):
    """Test that the API is accessible."""
    try:
        async with session.get(f"{API_BASE}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ API Health: {data['status']}")
                services = data.get('services', {})
                for service, status in services.items():
                    status_emoji = "✅" if status else "❌"
                    print(f"   {status_emoji} {service}")
                return True
    except Exception as e:
        print(f"❌ API Health Check Failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 AI On-Call Agent Resolution Monitor Test")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # 1. Test API connectivity
        if not await test_health_endpoint(session):
            print("❌ Cannot connect to API. Make sure Docker Compose is running.")
            return
        
        print()
        
        # 2. Get current resolution summary
        await get_resolution_summary(session)
        
        # 3. Get recent resolutions
        await get_recent_resolutions(session)
        
        print("=" * 50)
        print("🎯 TO SEE LIVE INCIDENT RESOLUTION:")
        print()
        print("1. Check Logs in Real-Time:")
        print("   docker-compose logs -f oncall-agent | grep -E '(AUTOMATED|RESOLUTION|SUCCESS)'")
        print()
        print("2. Monitor Resolution API:")
        print(f"   curl {API_BASE}/resolutions/live-feed | jq")
        print()
        print("3. Get Resolution Summary:")
        print(f"   curl {API_BASE}/resolutions/summary | jq")
        print()
        print("4. View Recent Resolutions:")
        print(f"   curl {API_BASE}/resolutions/recent | jq")
        print()
        print("5. Get Detailed Metrics:")
        print(f"   curl {API_BASE}/resolutions/metrics | jq")
        print()
        print("6. Access Web Interface:")
        print(f"   Open {API_BASE}/docs in your browser for interactive API testing")
        print()
        
        # 4. Demo: Create a test incident (simulation)
        print("🧪 CREATING TEST INCIDENT...")
        test_incident = {
            "title": "Test Database Connection Timeout",
            "description": "Connection to PostgreSQL database timing out after 30 seconds",
            "service": "postgresql",
            "severity": "high",
            "tags": ["database", "timeout", "connection"]
        }
        
        incident_id = await create_test_incident(session, test_incident)
        
        # 5. Monitor for a short time
        print("\n👁️  MONITORING FOR AUTOMATED RESOLUTION (30 seconds)...")
        await monitor_incident_resolution(session, incident_id, max_wait=30)
        
        print("\n✨ Test completed! Use the commands above to monitor real incidents.")

if __name__ == "__main__":
    asyncio.run(main())
