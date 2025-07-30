#!/usr/bin/env python3
"""
AI Model Training Script for On-Call Agent

This script trains machine learning models for incident classification,
resolution confidence prediction, and anomaly detection.

Usage:
    python scripts/train_models.py --data data/training.json --output models/trained/ai_model.pkl
    python scripts/train_models.py --help

Features:
    - Supports JSON training data format
    - Automatic model validation
    - Cross-validation evaluation
    - Model versioning with metadata
    - Export detailed training reports

Example training data format:
    [
        {
            "incident": {
                "title": "Database Connection Timeout",
                "description": "PostgreSQL connection pool exhausted",
                "service": "postgres",
                "severity": "high",
                "tags": ["database", "timeout"]
            },
            "outcome": "restart_database_connection",
            "resolution_time": 120,
            "success": true
        }
    ]
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai import AIDecisionEngine
from src.models.schemas import IncidentCreate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(file_path: str) -> AIDecisionEngine:
    """Load training data from JSON file into AI engine."""
    logger.info(f"Loading training data from {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Training data must be a list of samples")
    
    ai_engine = AIDecisionEngine()
    
    valid_samples = 0
    for i, sample in enumerate(data):
        try:
            # Validate sample structure
            if not all(key in sample for key in ["incident", "outcome", "success"]):
                logger.warning(f"Skipping invalid sample {i}: missing required fields")
                continue
            
            incident = IncidentCreate(**sample["incident"])
            ai_engine.add_training_data(
                incident=incident,
                outcome=sample["outcome"],
                resolution_time=sample.get("resolution_time", 300),
                success=sample.get("success", True)
            )
            valid_samples += 1
            
        except Exception as e:
            logger.warning(f"Skipping invalid sample {i}: {e}")
            continue
    
    logger.info(f"Loaded {valid_samples} valid training samples from {len(data)} total")
    return ai_engine


def validate_training_data(ai_engine: AIDecisionEngine, min_samples: int) -> bool:
    """Validate that training data meets requirements."""
    sample_count = len(ai_engine.training_data["incidents"])
    
    if sample_count < min_samples:
        logger.error(f"Insufficient training data: {sample_count} samples (minimum: {min_samples})")
        return False
    
    # Check data diversity
    outcomes = [sample["outcome"] for sample in ai_engine.training_data["incidents"]]
    unique_outcomes = set(outcomes)
    
    if len(unique_outcomes) < 2:
        logger.warning("Training data has low diversity (only 1 outcome type)")
    
    success_rate = sum(sample["success"] for sample in ai_engine.training_data["incidents"]) / sample_count
    logger.info(f"Training data success rate: {success_rate:.3f}")
    
    if success_rate < 0.1 or success_rate > 0.95:
        logger.warning(f"Unusual success rate in training data: {success_rate:.3f}")
    
    return True


def train_and_evaluate(ai_engine: AIDecisionEngine, min_samples: int) -> Dict[str, Any]:
    """Train models and perform evaluation."""
    logger.info("Starting model training...")
    
    # Train models
    training_results = ai_engine.train_models(min_samples=min_samples)
    
    if not training_results["success"]:
        logger.error(f"Training failed: {training_results['error']}")
        return training_results
    
    # Log training results
    evaluation = training_results.get("evaluation", {})
    logger.info("Training completed successfully!")
    logger.info(f"  📊 Training samples: {training_results['training_samples']}")
    logger.info(f"  🎯 Classification accuracy: {evaluation.get('classification_accuracy', 0):.3f}")
    logger.info(f"  🔄 Confidence accuracy: {evaluation.get('confidence_accuracy', 0):.3f}")
    
    # Display feature importance
    feature_importance = ai_engine._get_feature_importance()
    if feature_importance:
        logger.info("Top 5 important features:")
        for i, feat in enumerate(feature_importance[:5]):
            logger.info(f"  {i+1}. {feat['feature']}: {feat['importance']:.3f}")
    
    return training_results


def save_model_with_metadata(ai_engine: AIDecisionEngine, output_path: str, 
                            training_results: Dict[str, Any]) -> bool:
    """Save model with comprehensive metadata."""
    logger.info(f"Saving model to {output_path}")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Add additional metadata
    ai_engine.model_metadata.update({
        "training_script_version": "1.0.0",
        "training_timestamp": datetime.utcnow().isoformat(),
        "training_results": training_results
    })
    
    # Save model
    if ai_engine.save_model(output_path):
        logger.info("✅ Model saved successfully!")
        
        # Create symlink to latest
        latest_path = Path(output_path).parent / "latest.pkl"
        try:
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(Path(output_path).name)
            logger.info(f"Created symlink: {latest_path}")
        except Exception as e:
            logger.warning(f"Could not create latest symlink: {e}")
        
        return True
    else:
        logger.error("❌ Failed to save model")
        return False


def export_training_report(ai_engine: AIDecisionEngine, report_path: str, 
                          training_results: Dict[str, Any]) -> bool:
    """Export comprehensive training report."""
    logger.info(f"Exporting training report to {report_path}")
    
    try:
        # Create comprehensive report
        report = {
            "training_summary": {
                "timestamp": datetime.utcnow().isoformat(),
                "samples_used": training_results["training_samples"],
                "training_success": training_results["success"],
                "evaluation_metrics": training_results.get("evaluation", {})
            },
            "model_info": ai_engine.get_model_info(),
            "feature_importance": ai_engine._get_feature_importance(),
            "training_data_stats": {
                "total_samples": len(ai_engine.training_data["incidents"]),
                "outcomes": {},
                "success_rate": 0.0,
                "avg_resolution_time": 0.0
            }
        }
        
        # Calculate training data statistics
        if ai_engine.training_data["incidents"]:
            outcomes = [sample["outcome"] for sample in ai_engine.training_data["incidents"]]
            successes = [sample["success"] for sample in ai_engine.training_data["incidents"]]
            times = [sample.get("resolution_time", 300) for sample in ai_engine.training_data["incidents"]]
            
            from collections import Counter
            report["training_data_stats"].update({
                "outcomes": dict(Counter(outcomes)),
                "success_rate": sum(successes) / len(successes),
                "avg_resolution_time": sum(times) / len(times)
            })
        
        # Save report
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("✅ Training report exported successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to export training report: {e}")
        return False


def generate_sample_data(output_path: str, num_samples: int = 100):
    """Generate sample training data for testing."""
    logger.info(f"Generating {num_samples} sample training records...")
    
    sample_templates = [
        {
            "incident": {
                "title": "Database Connection Timeout",
                "description": "PostgreSQL connection pool exhausted after 30 seconds",
                "service": "postgres",
                "severity": "high",
                "tags": ["database", "timeout", "connection"]
            },
            "outcome": "restart_database_connection",
            "resolution_time": 120,
            "success": True
        },
        {
            "incident": {
                "title": "Spark OutOfMemoryError",
                "description": "Java heap space exceeded in executor",
                "service": "spark",
                "severity": "high",
                "tags": ["spark", "memory", "oom"]
            },
            "outcome": "restart_spark_job",
            "resolution_time": 180,
            "success": True
        },
        {
            "incident": {
                "title": "Airflow DAG Timeout",
                "description": "data_pipeline DAG stuck for 45 minutes",
                "service": "airflow",
                "severity": "medium",
                "tags": ["airflow", "dag", "timeout"]
            },
            "outcome": "restart_airflow_dag",
            "resolution_time": 90,
            "success": True
        },
        {
            "incident": {
                "title": "API Rate Limit Exceeded",
                "description": "Too many requests to external service",
                "service": "api-gateway",
                "severity": "medium",
                "tags": ["api", "rate-limit", "external"]
            },
            "outcome": "enable_circuit_breaker",
            "resolution_time": 60,
            "success": True
        },
        {
            "incident": {
                "title": "Disk Space Critical",
                "description": "Root filesystem 95% full",
                "service": "server",
                "severity": "critical",
                "tags": ["disk", "storage", "filesystem"]
            },
            "outcome": "cleanup_logs",
            "resolution_time": 300,
            "success": False  # Some failures for realistic data
        }
    ]
    
    import random
    
    samples = []
    for i in range(num_samples):
        template = random.choice(sample_templates)
        sample = json.loads(json.dumps(template))  # Deep copy
        
        # Add some variation
        sample["resolution_time"] = random.randint(60, 600)
        sample["success"] = random.random() > 0.15  # 85% success rate
        
        # Vary severity occasionally
        if random.random() < 0.1:
            sample["incident"]["severity"] = random.choice(["low", "medium", "high", "critical"])
        
        samples.append(sample)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    logger.info(f"✅ Generated sample data: {output_path}")


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="Train AI models for incident classification and resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--data", 
        required=True, 
        help="Training data JSON file path"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output model file path (.pkl)"
    )
    parser.add_argument(
        "--min-samples", 
        type=int, 
        default=50, 
        help="Minimum training samples required (default: 50)"
    )
    parser.add_argument(
        "--report", 
        help="Export training report to JSON file"
    )
    parser.add_argument(
        "--generate-sample-data", 
        metavar="PATH",
        help="Generate sample training data and save to specified path"
    )
    parser.add_argument(
        "--sample-count", 
        type=int, 
        default=100,
        help="Number of sample records to generate (default: 100)"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only validate training data without training"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate sample data if requested
    if args.generate_sample_data:
        generate_sample_data(args.generate_sample_data, args.sample_count)
        if not Path(args.data).exists():
            logger.info("No training data file specified, using generated sample data")
            args.data = args.generate_sample_data
    
    try:
        logger.info("🚀 Starting AI model training pipeline...")
        
        # Load training data
        ai_engine = load_training_data(args.data)
        
        # Validate training data
        if not validate_training_data(ai_engine, args.min_samples):
            return 1
        
        if args.validate_only:
            logger.info("✅ Training data validation completed")
            return 0
        
        # Train models
        training_results = train_and_evaluate(ai_engine, args.min_samples)
        
        if not training_results["success"]:
            return 1
        
        # Save model
        if not save_model_with_metadata(ai_engine, args.output, training_results):
            return 1
        
        # Export training report if requested
        if args.report:
            export_training_report(ai_engine, args.report, training_results)
        
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info(f"📁 Model saved: {args.output}")
        if args.report:
            logger.info(f"📄 Report saved: {args.report}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
