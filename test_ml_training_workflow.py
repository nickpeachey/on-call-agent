#!/usr/bin/env python3
"""
Test script to demonstrate the complete ML training workflow.

This script shows:
1. How training data is collected from incident resolutions
2. When models are trained (startup, incremental, manual)
3. How models are saved and loaded
4. How the trained models are used for predictions
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate, Severity
from src.core import setup_logging


async def create_sample_incident(title: str, service: str, severity: Severity, description: str) -> IncidentCreate:
    """Create a sample incident for testing."""
    return IncidentCreate(
        title=title,
        service=service,
        severity=severity,
        description=description,
        tags=["automated-test"]
    )


async def simulate_incident_resolutions(ai_engine: AIDecisionEngine, num_incidents: int = 20):
    """Simulate a series of incident resolutions to generate training data."""
    print(f"\n🎭 SIMULATING {num_incidents} INCIDENT RESOLUTIONS")
    print("=" * 60)
    
    # Sample incident scenarios
    incident_scenarios = [
        {
            "title": "Database connection timeout",
            "service": "payment-db",
            "severity": Severity.HIGH,
            "description": "Database connection timeout after 30 seconds. Connection pool exhausted.",
            "outcome": "resolved",
            "resolution_time": 180,
            "success": True,
            "actions": ["restart_database_connection", "scale_resources"]
        },
        {
            "title": "Airflow DAG failure",
            "service": "etl-pipeline",
            "severity": Severity.MEDIUM,
            "description": "ETL DAG failed due to missing input file in /data/staging/",
            "outcome": "resolved",
            "resolution_time": 240,
            "success": True,
            "actions": ["restart_airflow_dag"]
        },
        {
            "title": "Memory leak in microservice",
            "service": "user-api",
            "severity": Severity.HIGH,
            "description": "OutOfMemoryError: Java heap space. Service consuming 8GB RAM.",
            "outcome": "resolved",
            "resolution_time": 300,
            "success": True,
            "actions": ["restart_service", "increase_memory"]
        },
        {
            "title": "Spark job execution timeout",
            "service": "analytics",
            "severity": Severity.MEDIUM,
            "description": "Spark job timeout after 2 hours. Executor lost due to network issues.",
            "outcome": "resolved", 
            "resolution_time": 150,
            "success": True,
            "actions": ["restart_spark_job"]
        },
        {
            "title": "Redis cache connection refused",
            "service": "session-cache",
            "severity": Severity.CRITICAL,
            "description": "Connection to Redis refused. Service unavailable error.",
            "outcome": "failed",
            "resolution_time": 600,
            "success": False,
            "actions": ["restart_service", "clear_cache"]
        },
        {
            "title": "Disk space critical on data node",
            "service": "storage",
            "severity": Severity.CRITICAL,
            "description": "Disk usage 95% on /data partition. Only 2GB remaining.",
            "outcome": "resolved",
            "resolution_time": 120,
            "success": True,
            "actions": ["cleanup_logs", "scale_resources"]
        }
    ]
    
    for i in range(num_incidents):
        scenario = incident_scenarios[i % len(incident_scenarios)]
        
        # Create incident with some variation
        incident = await create_sample_incident(
            title=f"{scenario['title']} #{i+1}",
            service=scenario['service'],
            severity=scenario['severity'],
            description=scenario['description']
        )
        
        # Add some randomness to outcomes
        success = scenario['success']
        if i % 7 == 0:  # Make some resolutions fail randomly
            success = not success
            
        resolution_time = scenario['resolution_time'] + (i * 10)  # Vary resolution times
        confidence_score = 0.8 if success else 0.3
        
        # Add training data
        await ai_engine.add_training_data_async(
            incident=incident,
            outcome=scenario['outcome'],
            resolution_time=resolution_time,
            success=success,
            confidence_score=confidence_score,
            actions_executed=scenario['actions']
        )
        
        print(f"  📊 Added incident {i+1:2d}: {incident.title[:50]:<50} | Success: {success}")
    
    print(f"\n✅ Generated {num_incidents} training samples")


async def test_model_predictions(ai_engine: AIDecisionEngine):
    """Test model predictions on new incidents."""
    print(f"\n🔮 TESTING MODEL PREDICTIONS")
    print("=" * 60)
    
    # Test incidents
    test_incidents = [
        await create_sample_incident(
            "Database deadlock detected",
            "payment-db", 
            Severity.HIGH,
            "Database deadlock timeout exceeded. Multiple transactions waiting."
        ),
        await create_sample_incident(
            "Airflow task stuck in queued state", 
            "etl-pipeline",
            Severity.MEDIUM,
            "DAG task has been queued for 30 minutes without execution."
        ),
        await create_sample_incident(
            "Memory usage spike in API service",
            "user-api",
            Severity.HIGH, 
            "Memory usage increased from 2GB to 7GB in 10 minutes. GC overhead limit exceeded."
        )
    ]
    
    for i, incident in enumerate(test_incidents, 1):
        print(f"\n  🎯 Test Incident {i}: {incident.title}")
        
        # Predict category
        category, cat_confidence = ai_engine.predict_incident_category(incident)
        print(f"     📂 Predicted Category: {category} (confidence: {cat_confidence:.2f})")
        
        # Predict resolution confidence
        res_confidence = ai_engine.predict_resolution_confidence(incident)
        print(f"     🎯 Resolution Confidence: {res_confidence:.2f}")
        
        # Detect anomalies
        anomaly_result = ai_engine.detect_anomalies(incident)
        print(f"     🚨 Anomaly Detection: {anomaly_result['is_anomaly']} (score: {anomaly_result['anomaly_score']:.2f})")


async def demonstrate_training_workflow():
    """Demonstrate the complete ML training workflow."""
    setup_logging()
    
    print("🧠 AI DECISION ENGINE - ML TRAINING WORKFLOW DEMONSTRATION")
    print("=" * 80)
    
    # Initialize AI engine
    ai_engine = AIDecisionEngine()
    
    # Show initial state (no models)
    print(f"\n📊 INITIAL STATE")
    print("-" * 40)
    model_info = ai_engine.get_model_info()
    print(f"  Has Classifier: {model_info['has_classifier']}")
    print(f"  Has Confidence Model: {model_info['has_confidence_model']}")
    print(f"  Has Clustering: {model_info['has_clustering']}")
    print(f"  Training Samples: {model_info['training_samples']}")
    
    # Step 1: Simulate incident resolutions to generate training data
    await simulate_incident_resolutions(ai_engine, num_incidents=25)
    
    # Step 2: Show training data accumulated
    print(f"\n📈 TRAINING DATA ACCUMULATED")
    print("-" * 40)
    samples = len(ai_engine.training_data["incidents"])
    print(f"  Total Training Samples: {samples}")
    
    # Step 3: Train models manually
    print(f"\n🧠 TRAINING ML MODELS")
    print("-" * 40)
    print("  Starting model training...")
    
    results = ai_engine.train_models(min_samples=10)
    
    if results["success"]:
        print(f"  ✅ Training successful!")
        print(f"     📊 Samples used: {results['training_samples']}")
        print(f"     🎯 Classification accuracy: {results.get('evaluation', {}).get('classification_accuracy', 0):.3f}")
        print(f"     📝 Model version: {results['metadata'].get('version', 'unknown')}")
        
        # Step 4: Save models
        print(f"\n💾 SAVING TRAINED MODELS")
        print("-" * 40)
        model_path = "models/test_ai_model.pkl"
        save_success = ai_engine.save_model(model_path)
        print(f"  Model saved: {save_success}")
        print(f"  Model path: {model_path}")
        
        if save_success and os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"  File size: {file_size:,} bytes")
        
        # Step 5: Test model loading
        print(f"\n📂 TESTING MODEL LOADING")
        print("-" * 40)
        
        # Create new AI engine instance
        ai_engine_2 = AIDecisionEngine()
        load_success = ai_engine_2.load_model(model_path)
        print(f"  Model loaded: {load_success}")
        
        if load_success:
            loaded_info = ai_engine_2.get_model_info()
            print(f"  Loaded samples: {loaded_info['training_samples']}")
            print(f"  Version: {loaded_info['metadata'].get('version', 'unknown')}")
            print(f"  Accuracy: {loaded_info['metadata'].get('accuracy', 0):.3f}")
        
        # Step 6: Test predictions
        await test_model_predictions(ai_engine_2 if load_success else ai_engine)
        
        # Step 7: Simulate adding more data and incremental training
        print(f"\n🔄 TESTING INCREMENTAL TRAINING")
        print("-" * 40)
        
        # Add a few more incidents
        await simulate_incident_resolutions(ai_engine, num_incidents=5)
        
        # Manually trigger retraining
        retrain_results = await ai_engine.retrain_models(min_samples=10)
        if retrain_results["success"]:
            print(f"  ✅ Incremental training successful!")
            print(f"  📊 New accuracy: {retrain_results.get('evaluation', {}).get('classification_accuracy', 0):.3f}")
            print(f"  💾 Model saved: {retrain_results.get('model_saved', False)}")
        
    else:
        print(f"  ❌ Training failed: {results.get('error', 'Unknown error')}")
    
    # Final state
    print(f"\n📊 FINAL STATE")
    print("-" * 40)
    final_info = ai_engine.get_model_info()
    print(f"  Has Classifier: {final_info['has_classifier']}")
    print(f"  Has Confidence Model: {final_info['has_confidence_model']}")
    print(f"  Has Clustering: {final_info['has_clustering']}")
    print(f"  Training Samples: {final_info['training_samples']}")
    print(f"  Feature Count: {final_info['feature_count']}")
    
    print(f"\n🎉 ML TRAINING WORKFLOW DEMONSTRATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demonstrate_training_workflow())
