#!/usr/bin/env python3
"""
Test script to demonstrate continuous learning capabilities.
This script simulates incident resolution and learning feedback.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate, Severity


async def test_continuous_learning():
    """Test the continuous learning system with simulated incidents."""
    
    print("🤖 Testing AI On-Call Agent Continuous Learning")
    print("=" * 50)
    
    # Initialize AI engine
    ai_engine = AIDecisionEngine()
    
    # Simulate incident resolution scenarios
    test_scenarios = [
        {
            "incident": IncidentCreate(
                title="Airflow DAG 'data_pipeline' timeout",
                description="DAG data_pipeline (dag_id: data_pipeline) failed to complete within 4 hours. Task 'transform_data' in DAG run dag_run_20241201_083000 is stuck in 'running' state. Error: Task instance timeout after 3600 seconds.",
                service="airflow",
                severity=Severity.HIGH,
                tags=["dag_timeout", "task_stuck", "data_pipeline"]
            ),
            "actions_taken": [
                {
                    "action": "restart_dag_task",
                    "target": "data_pipeline.transform_data",
                    "dag_id": "data_pipeline",
                    "dag_run_id": "dag_run_20241201_083000",
                    "task_id": "transform_data",
                    "result": "success",
                    "timestamp": "2024-12-01T08:32:15Z",
                    "duration": 45
                }
            ],
            "success": True,
            "resolution_time": 90,
            "confidence": 0.85
        },
        {
            "incident": IncidentCreate(
                title="PostgreSQL connection pool exhausted",
                description="Database connection pool exhausted: max_connections=20, current_connections=20. Host: prod-db.company.com:5432. Error: FATAL: remaining connection slots are reserved. Connection timeout after 30 seconds.",
                service="postgresql",
                severity=Severity.CRITICAL,
                tags=["database", "connection_pool", "timeout"]
            ),
            "actions_taken": [
                {
                    "action": "increase_connection_pool",
                    "target": "prod-db.company.com:5432",
                    "host": "prod-db.company.com",
                    "port": 5432,
                    "pool_size_before": 20,
                    "pool_size_after": 40,
                    "result": "failed",
                    "timestamp": "2024-12-01T09:15:30Z",
                    "duration": 120
                },
                {
                    "action": "restart_application_pool",
                    "target": "web-app",
                    "result": "success",
                    "timestamp": "2024-12-01T09:17:45Z",
                    "duration": 180
                }
            ],
            "success": True,
            "resolution_time": 450,
            "confidence": 0.72
        },
        {
            "incident": IncidentCreate(
                title="Spark application out of memory",
                description="Spark application spark-etl-job-20241201 (application_id: app-20241201083045-0001) failed with OutOfMemoryError. Executor spark-exec-001 on host worker-node-03 crashed. Driver memory: 2GB, Executor memory: 4GB. Stage 3, Task 45 failed with java.lang.OutOfMemoryError: Java heap space.",
                service="spark",
                severity=Severity.HIGH,
                tags=["spark", "oom", "memory", "executor_failure"]
            ),
            "actions_taken": [
                {
                    "action": "increase_executor_memory",
                    "target": "app-20241201083045-0001",
                    "application_id": "app-20241201083045-0001",
                    "memory_before": "4g",
                    "memory_after": "8g",
                    "result": "failed",
                    "timestamp": "2024-12-01T10:20:15Z",
                    "duration": 300
                },
                {
                    "action": "restart_with_more_memory",
                    "target": "spark-etl-job-20241201",
                    "application_id": "app-20241201103045-0002",
                    "executor_memory": "8g",
                    "driver_memory": "4g",
                    "result": "success",
                    "timestamp": "2024-12-01T10:25:30Z",
                    "duration": 600
                }
            ],
            "success": True,
            "resolution_time": 1200,
            "confidence": 0.68
        }
    ]
    
    # Process each scenario and record learning
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['incident'].title}")
        print(f"   Service: {scenario['incident'].service}")
        print(f"   Severity: {scenario['incident'].severity}")
        
        # Simulate incident analysis
        incident_id = f"test_incident_{i}_{int(asyncio.get_event_loop().time())}"
        
        # Record resolution outcome for learning
        learning_record = await ai_engine.record_resolution_outcome(
            incident_id=incident_id,
            actions_taken=scenario["actions_taken"],
            success=scenario["success"],
            resolution_time=scenario["resolution_time"],
            confidence_score=scenario["confidence"]
        )
        
        print(f"   ✅ Resolution: {'Success' if scenario['success'] else 'Failed'}")
        print(f"   ⏱️  Time: {scenario['resolution_time']}s")
        print(f"   🎯 Confidence: {scenario['confidence']:.2f}")
        
        # Show learning feedback
        feedback = learning_record["learning_feedback"]
        print(f"   📚 Pattern Strength: {feedback.get('pattern_match_strength', 0):.2f}")
        
        if feedback.get("new_patterns_learned"):
            print(f"   🧠 New Patterns: {', '.join(feedback['new_patterns_learned'])}")
        
        # Small delay between scenarios
        await asyncio.sleep(0.5)
    
    # Show overall learning statistics
    print(f"\n📊 Learning Statistics")
    print("=" * 30)
    
    stats = await ai_engine.get_learning_statistics()
    
    print(f"Total Incidents Processed: {stats['total_incidents']}")
    print(f"Overall Success Rate: {stats['success_rate']:.2f}")
    print(f"Average Resolution Time: {stats['average_resolution_time']:.0f}s")
    print(f"Recent Success Rate: {stats.get('recent_success_rate', 0):.2f}")
    print(f"Current Confidence Threshold: {stats.get('confidence_threshold', 0.6):.2f}")
    
    if stats.get('learning_trends'):
        trend = stats['learning_trends']
        trend_emoji = "📈" if trend.get('improving') else "📉"
        print(f"Learning Trend: {trend_emoji} {trend.get('trend_direction', 'stable').upper()}")
    
    print(f"\n🎉 Continuous learning test completed!")
    print(f"   Learning data saved to: data/continuous_learning.json")


async def demonstrate_metadata_extraction():
    """Demonstrate service-specific metadata extraction."""
    
    print("\n🔍 Testing Metadata Extraction")
    print("=" * 40)
    
    ai_engine = AIDecisionEngine()
    
    # Test different log formats
    test_logs = [
        {
            "type": "Airflow DAG",
            "log": "DAG data_pipeline (dag_id: data_pipeline) failed to complete within 4 hours. Task 'transform_data' in DAG run dag_run_20241201_083000 is stuck.",
            "extract_method": "_extract_airflow_metadata"
        },
        {
            "type": "Database Connection",
            "log": "Database connection pool exhausted: max_connections=20. Host: prod-db.company.com:5432. Connection timeout after 30 seconds.",
            "extract_method": "_extract_database_metadata"
        },
        {
            "type": "Spark Application",
            "log": "Spark application spark-etl-job-20241201 (application_id: app-20241201083045-0001) failed with OutOfMemoryError. Executor spark-exec-001 crashed.",
            "extract_method": "_extract_spark_metadata"
        }
    ]
    
    for test in test_logs:
        print(f"\n📝 {test['type']} Log:")
        print(f"   {test['log'][:100]}...")
        
        # Extract metadata using the appropriate method
        # Create a mock incident for testing
        mock_incident = IncidentCreate(
            title="Test Incident",
            description=test['log'],
            service="test",
            severity=Severity.MEDIUM,
            tags=[]
        )
        
        if test['extract_method'] == '_extract_airflow_metadata':
            metadata = ai_engine._extract_airflow_metadata(mock_incident)
        elif test['extract_method'] == '_extract_database_metadata':
            metadata = ai_engine._extract_database_metadata(mock_incident)
        elif test['extract_method'] == '_extract_spark_metadata':
            metadata = ai_engine._extract_spark_metadata(mock_incident)
        else:
            metadata = {}
        
        print(f"   🔧 Extracted Metadata:")
        for key, value in metadata.items():
            print(f"      {key}: {value}")


if __name__ == "__main__":
    async def main():
        await test_continuous_learning()
        await demonstrate_metadata_extraction()
    
    asyncio.run(main())
