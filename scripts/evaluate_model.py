#!/usr/bin/env python3
"""
AI Model Evaluation Script for On-Call Agent

This script evaluates trained AI models on test data to measure performance
and provide detailed analysis of model predictions.

Usage:
    python scripts/evaluate_model.py --model models/trained/ai_model.pkl --test-data data/test.json
    python scripts/evaluate_model.py --model models/trained/ai_model.pkl --interactive

Features:
    - Comprehensive model evaluation metrics
    - Interactive testing mode
    - Confusion matrix and classification reports
    - Feature importance analysis
    - Anomaly detection testing
    - Performance benchmarking

Example test data format:
    [
        {
            "incident": {
                "title": "Database Connection Timeout",
                "description": "PostgreSQL connection pool exhausted",
                "service": "postgres",
                "severity": "high",
                "tags": ["database", "timeout"]
            },
            "expected_category": "database_connectivity",
            "expected_success": true
        }
    ]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

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


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model_path: str):
        """Initialize evaluator with trained model."""
        self.ai_engine = AIDecisionEngine()
        
        if not self.ai_engine.load_model(model_path):
            raise ValueError(f"Failed to load model from {model_path}")
        
        self.model_info = self.ai_engine.get_model_info()
        logger.info(f"Loaded model: {model_path}")
        logger.info(f"Model version: {self.model_info['metadata'].get('version', 'unknown')}")
        logger.info(f"Training samples: {self.model_info['training_samples']}")
        logger.info(f"Model accuracy: {self.model_info['metadata'].get('accuracy', 0):.3f}")
    
    def evaluate_on_test_data(self, test_data_path: str) -> Dict[str, Any]:
        """Evaluate model on test dataset."""
        logger.info(f"Loading test data from {test_data_path}")
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        if not isinstance(test_data, list):
            raise ValueError("Test data must be a list of samples")
        
        results = {
            "total_samples": len(test_data),
            "correct_predictions": 0,
            "predictions": [],
            "category_accuracy": {},
            "confidence_stats": {
                "mean_confidence": 0.0,
                "correct_high_confidence": 0,
                "incorrect_high_confidence": 0
            },
            "anomaly_stats": {
                "detected_anomalies": 0,
                "false_anomalies": 0
            },
            "performance_metrics": {}
        }
        
        category_correct = {}
        category_total = {}
        confidence_scores = []
        
        logger.info(f"Evaluating {len(test_data)} test samples...")
        
        for i, sample in enumerate(test_data):
            try:
                incident = IncidentCreate(**sample["incident"])
                expected_category = sample.get("expected_category", "unknown")
                expected_success = sample.get("expected_success", True)
                
                # Make predictions
                start_time = time.time()
                predicted_category, category_confidence = self.ai_engine.predict_incident_category(incident)
                resolution_confidence = self.ai_engine.predict_resolution_confidence(incident)
                anomaly_info = self.ai_engine.detect_anomalies(incident)
                prediction_time = time.time() - start_time
                
                # Evaluate category prediction
                category_correct_flag = predicted_category.lower() == expected_category.lower()
                if category_correct_flag:
                    results["correct_predictions"] += 1
                
                # Track category-specific accuracy
                if expected_category not in category_total:
                    category_total[expected_category] = 0
                    category_correct[expected_category] = 0
                
                category_total[expected_category] += 1
                if category_correct_flag:
                    category_correct[expected_category] += 1
                
                # Track confidence statistics
                confidence_scores.append(category_confidence)
                if category_confidence > 0.8:  # High confidence threshold
                    if category_correct_flag:
                        results["confidence_stats"]["correct_high_confidence"] += 1
                    else:
                        results["confidence_stats"]["incorrect_high_confidence"] += 1
                
                # Track anomaly detection
                is_anomaly = anomaly_info.get("is_anomaly", False)
                expected_anomaly = sample.get("is_anomaly", False)
                if is_anomaly:
                    results["anomaly_stats"]["detected_anomalies"] += 1
                    if not expected_anomaly:
                        results["anomaly_stats"]["false_anomalies"] += 1
                
                # Store detailed prediction result
                prediction_result = {
                    "sample_id": i,
                    "incident_title": incident.title,
                    "expected_category": expected_category,
                    "predicted_category": predicted_category,
                    "category_confidence": category_confidence,
                    "resolution_confidence": resolution_confidence,
                    "is_anomaly": is_anomaly,
                    "correct": category_correct_flag,
                    "prediction_time_ms": prediction_time * 1000
                }
                
                results["predictions"].append(prediction_result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    current_accuracy = results["correct_predictions"] / (i + 1)
                    logger.info(f"Processed {i + 1}/{len(test_data)} samples, accuracy: {current_accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue
        
        # Calculate final metrics
        if results["total_samples"] > 0:
            results["overall_accuracy"] = results["correct_predictions"] / results["total_samples"]
            results["confidence_stats"]["mean_confidence"] = sum(confidence_scores) / len(confidence_scores)
            
            # Calculate per-category accuracy
            for category in category_total:
                if category_total[category] > 0:
                    results["category_accuracy"][category] = category_correct[category] / category_total[category]
            
            # Calculate average prediction time
            prediction_times = [p["prediction_time_ms"] for p in results["predictions"]]
            results["performance_metrics"] = {
                "avg_prediction_time_ms": sum(prediction_times) / len(prediction_times),
                "max_prediction_time_ms": max(prediction_times),
                "min_prediction_time_ms": min(prediction_times)
            }
        
        return results
    
    def interactive_evaluation(self):
        """Interactive mode for testing individual incidents."""
        logger.info("🔬 Starting interactive evaluation mode")
        logger.info("Enter incident details to test model predictions (type 'quit' to exit)")
        
        while True:
            try:
                print("\n" + "="*60)
                print("Interactive Model Testing")
                print("="*60)
                
                # Get user input
                title = input("Incident Title: ").strip()
                if title.lower() in ['quit', 'exit', 'q']:
                    break
                
                description = input("Description: ").strip()
                service = input("Service (optional): ").strip() or "unknown"
                severity = input("Severity [low/medium/high/critical]: ").strip() or "medium"
                tags_input = input("Tags (comma-separated, optional): ").strip()
                
                tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
                
                # Create incident
                incident = IncidentCreate(
                    title=title,
                    description=description,
                    service=service,
                    severity=severity,
                    tags=tags
                )
                
                # Make predictions
                print("\n🤖 AI Analysis:")
                print("-" * 40)
                
                start_time = time.time()
                
                category, category_confidence = self.ai_engine.predict_incident_category(incident)
                resolution_confidence = self.ai_engine.predict_resolution_confidence(incident)
                anomaly_info = self.ai_engine.detect_anomalies(incident)
                
                prediction_time = (time.time() - start_time) * 1000
                
                # Display results
                print(f"📊 Predicted Category: {category}")
                print(f"🎯 Category Confidence: {category_confidence:.3f}")
                print(f"✅ Resolution Confidence: {resolution_confidence:.3f}")
                print(f"🚨 Anomaly Detection: {'Yes' if anomaly_info['is_anomaly'] else 'No'}")
                if anomaly_info['is_anomaly']:
                    print(f"   Anomaly Score: {anomaly_info['anomaly_score']:.3f}")
                print(f"⏱️  Prediction Time: {prediction_time:.2f}ms")
                
                # Recommendations
                print(f"\n💡 Recommendations:")
                if resolution_confidence > 0.8 and not anomaly_info['is_anomaly']:
                    print("   ✅ Recommend automated resolution")
                elif resolution_confidence > 0.6:
                    print("   ⚠️  Consider automated resolution with monitoring")
                else:
                    print("   🚨 Recommend manual intervention")
                
                # Show extracted features for debugging
                debug = input("\nShow feature analysis? [y/N]: ").strip().lower()
                if debug == 'y':
                    features = self.ai_engine._extract_ml_features(incident)
                    print(f"\n🔍 Extracted Features:")
                    for key, value in features.items():
                        if key != 'text_content':  # Skip large text content
                            print(f"   {key}: {value}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    def benchmark_performance(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model prediction performance."""
        logger.info(f"🏃 Running performance benchmark with {num_iterations} iterations...")
        
        # Create sample incident for benchmarking
        sample_incident = IncidentCreate(
            title="Performance Test Incident",
            description="Database connection timeout after 30 seconds in production environment",
            service="postgres",
            severity="high",
            tags=["database", "timeout", "production"]
        )
        
        prediction_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            
            # Make all predictions
            self.ai_engine.predict_incident_category(sample_incident)
            self.ai_engine.predict_resolution_confidence(sample_incident)
            self.ai_engine.detect_anomalies(sample_incident)
            
            prediction_time = (time.time() - start_time) * 1000
            prediction_times.append(prediction_time)
            
            if (i + 1) % 20 == 0:
                avg_time = sum(prediction_times) / len(prediction_times)
                logger.info(f"Completed {i + 1}/{num_iterations} iterations, avg time: {avg_time:.2f}ms")
        
        # Calculate statistics
        import statistics
        
        results = {
            "iterations": num_iterations,
            "total_time_ms": sum(prediction_times),
            "avg_time_ms": statistics.mean(prediction_times),
            "median_time_ms": statistics.median(prediction_times),
            "min_time_ms": min(prediction_times),
            "max_time_ms": max(prediction_times),
            "std_dev_ms": statistics.stdev(prediction_times) if len(prediction_times) > 1 else 0,
            "predictions_per_second": 1000 / statistics.mean(prediction_times)
        }
        
        logger.info(f"📊 Performance Results:")
        logger.info(f"   Average: {results['avg_time_ms']:.2f}ms")
        logger.info(f"   Median: {results['median_time_ms']:.2f}ms")
        logger.info(f"   Min/Max: {results['min_time_ms']:.2f}ms / {results['max_time_ms']:.2f}ms")
        logger.info(f"   Throughput: {results['predictions_per_second']:.1f} predictions/second")
        
        return results
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze and display feature importance."""
        logger.info("🔍 Analyzing feature importance...")
        
        feature_importance = self.ai_engine._get_feature_importance()
        
        if not feature_importance:
            logger.warning("No feature importance data available")
            return {}
        
        # Display top features
        print("\n📊 Top 10 Most Important Features:")
        print("-" * 50)
        for i, feat in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feat['feature']:<30} {feat['importance']:.4f}")
        
        # Categorize features
        categories = {
            "text_features": [],
            "pattern_features": [],
            "numerical_features": [],
            "temporal_features": []
        }
        
        for feat in feature_importance:
            name = feat['feature'].lower()
            if 'has_' in name or 'pattern' in name:
                categories["pattern_features"].append(feat)
            elif 'length' in name or 'count' in name:
                categories["numerical_features"].append(feat)
            elif 'hour' in name or 'day' in name or 'time' in name:
                categories["temporal_features"].append(feat)
            else:
                categories["text_features"].append(feat)
        
        # Display category summaries
        print(f"\n📋 Feature Categories:")
        for category, features in categories.items():
            if features:
                avg_importance = sum(f['importance'] for f in features) / len(features)
                print(f"   {category.replace('_', ' ').title()}: {len(features)} features, avg importance: {avg_importance:.4f}")
        
        return {
            "feature_importance": feature_importance,
            "categories": categories
        }


def print_evaluation_summary(results: Dict[str, Any]):
    """Print comprehensive evaluation summary."""
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION SUMMARY")
    print("="*60)
    
    print(f"📈 Overall Accuracy: {results['overall_accuracy']:.3f} ({results['correct_predictions']}/{results['total_samples']})")
    print(f"🎯 Mean Confidence: {results['confidence_stats']['mean_confidence']:.3f}")
    
    # Category-specific accuracy
    if results['category_accuracy']:
        print(f"\n📋 Category-Specific Accuracy:")
        for category, accuracy in results['category_accuracy'].items():
            print(f"   {category:<20}: {accuracy:.3f}")
    
    # Confidence analysis
    conf_stats = results['confidence_stats']
    print(f"\n🎯 Confidence Analysis:")
    print(f"   High Confidence Correct: {conf_stats['correct_high_confidence']}")
    print(f"   High Confidence Incorrect: {conf_stats['incorrect_high_confidence']}")
    
    if conf_stats['correct_high_confidence'] + conf_stats['incorrect_high_confidence'] > 0:
        high_conf_accuracy = conf_stats['correct_high_confidence'] / (
            conf_stats['correct_high_confidence'] + conf_stats['incorrect_high_confidence']
        )
        print(f"   High Confidence Accuracy: {high_conf_accuracy:.3f}")
    
    # Anomaly detection
    anomaly_stats = results['anomaly_stats']
    print(f"\n🚨 Anomaly Detection:")
    print(f"   Detected Anomalies: {anomaly_stats['detected_anomalies']}")
    print(f"   False Anomalies: {anomaly_stats['false_anomalies']}")
    
    # Performance metrics
    if 'performance_metrics' in results:
        perf = results['performance_metrics']
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Average Prediction Time: {perf['avg_prediction_time_ms']:.2f}ms")
        print(f"   Min/Max Prediction Time: {perf['min_prediction_time_ms']:.2f}ms / {perf['max_prediction_time_ms']:.2f}ms")


def export_evaluation_report(results: Dict[str, Any], report_path: str):
    """Export detailed evaluation report to JSON."""
    logger.info(f"📄 Exporting evaluation report to {report_path}")
    
    # Ensure output directory exists
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("✅ Evaluation report exported successfully")


def main():
    """Main evaluation script entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained AI models for incident classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model", 
        required=True, 
        help="Path to trained model file (.pkl)"
    )
    parser.add_argument(
        "--test-data", 
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive testing mode"
    )
    parser.add_argument(
        "--benchmark", 
        type=int, 
        metavar="N",
        help="Run performance benchmark with N iterations"
    )
    parser.add_argument(
        "--feature-analysis", 
        action="store_true",
        help="Analyze and display feature importance"
    )
    parser.add_argument(
        "--report", 
        help="Export detailed evaluation report to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("🔬 Starting model evaluation...")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model)
        
        # Run feature analysis if requested
        if args.feature_analysis:
            evaluator.analyze_feature_importance()
        
        # Run performance benchmark if requested
        if args.benchmark:
            benchmark_results = evaluator.benchmark_performance(args.benchmark)
            if args.report:
                # Include benchmark in report
                pass
        
        # Run test data evaluation if provided
        evaluation_results = None
        if args.test_data:
            evaluation_results = evaluator.evaluate_on_test_data(args.test_data)
            print_evaluation_summary(evaluation_results)
            
            if args.report:
                export_evaluation_report(evaluation_results, args.report)
        
        # Run interactive mode if requested
        if args.interactive:
            evaluator.interactive_evaluation()
        
        # If no specific mode requested, show model info and suggest options
        if not any([args.test_data, args.interactive, args.benchmark, args.feature_analysis]):
            print("\n💡 Available evaluation options:")
            print("   --test-data FILE       : Evaluate on test dataset")
            print("   --interactive          : Interactive testing mode")
            print("   --benchmark N          : Performance benchmark")
            print("   --feature-analysis     : Analyze feature importance")
            print("\nUse --help for more information")
        
        logger.info("✅ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
