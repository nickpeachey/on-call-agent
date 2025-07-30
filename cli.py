#!/usr/bin/env python3
"""Development CLI for AI On-Call Agent."""

import asyncio
import typer
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from datetime import datetime, timedelta

from src.core import setup_logging, get_logger, settings
from src.services.incidents import IncidentService
from src.services.knowledge_base import KnowledgeBaseService
from src.services.actions import ActionService
from src.ai import AIDecisionEngine
from src.monitoring import LogMonitorService
from src.models.schemas import IncidentCreate


app = typer.Typer(help="AI On-Call Agent Development CLI")

# Add ML subcommand group
ml_app = typer.Typer(help="Machine Learning model operations")
app.add_typer(ml_app, name="ml")

console = Console()
logger = get_logger(__name__)


@app.command()
def status():
    """Show system status."""
    console.print("🤖 AI On-Call Agent System Status", style="bold blue")
    
    table = Table(show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    table.add_row("Configuration", "✅ Loaded", f"Environment: {'Development' if settings.debug else 'Production'}")
    table.add_row("Database", "🔄 Checking", settings.database_url.split('@')[-1])  # Hide credentials
    table.add_row("Redis", "🔄 Checking", settings.redis_url.split('@')[-1])
    table.add_row("OpenAI", "✅ Configured" if settings.openai_api_key else "❌ Missing", "API Key present" if settings.openai_api_key else "Set OPENAI_API_KEY")
    table.add_row("Log Sources", "✅ Configured", f"{len(settings.elasticsearch_urls)} sources")
    
    console.print(table)


@app.command()
def incidents():
    """List recent incidents."""
    console.print("📋 Recent Incidents", style="bold blue")
    
    async def _list_incidents():
        service = IncidentService()
        incidents = await service.list_incidents(limit=10)
        
        if not incidents:
            console.print("No incidents found.", style="yellow")
            return
        
        table = Table(show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Service", style="magenta")
        table.add_column("Severity", style="red")
        table.add_column("Status", style="green")
        table.add_column("Created")
        
        for incident in incidents:
            created_ago = datetime.utcnow() - incident.created_at
            created_str = f"{created_ago.total_seconds() / 3600:.1f}h ago"
            
            table.add_row(
                incident.id[:8],
                incident.title[:50] + "..." if len(incident.title) > 50 else incident.title,
                incident.service,
                incident.severity,
                incident.status,
                created_str
            )
        
        console.print(table)
    
    asyncio.run(_list_incidents())


@app.command()
def knowledge():
    """List knowledge base entries."""
    console.print("📚 Knowledge Base Entries", style="bold blue")
    
    async def _list_knowledge():
        service = KnowledgeBaseService()
        entries = await service.search_entries(limit=10)
        
        if not entries:
            console.print("No knowledge base entries found.", style="yellow")
            return
        
        table = Table(show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Category", style="magenta")
        table.add_column("Success Rate", style="green")
        table.add_column("Last Used")
        
        for entry in entries:
            last_used = "Never" if not entry.last_used else f"{(datetime.utcnow() - entry.last_used).days}d ago"
            
            table.add_row(
                entry.id[:8],
                entry.title[:40] + "..." if len(entry.title) > 40 else entry.title,
                entry.category,
                f"{entry.success_rate:.0%}",
                last_used
            )
        
        console.print(table)
    
    asyncio.run(_list_knowledge())


@app.command()
def actions(limit: int = 10):
    """List recent actions."""
    console.print("⚡ Recent Actions", style="bold blue")
    
    async def _list_actions():
        service = ActionService()
        actions = await service.list_actions(limit=limit)
        
        if not actions:
            console.print("No actions found.", style="yellow")
            return
        
        table = Table(show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Created")
        table.add_column("Duration")
        table.add_column("Manual", style="yellow")
        
        for action in actions:
            created_ago = datetime.utcnow() - action.created_at
            created_str = f"{created_ago.total_seconds() / 60:.0f}m ago"
            
            duration = "N/A"
            if action.started_at and action.completed_at:
                duration_seconds = (action.completed_at - action.started_at).total_seconds()
                duration = f"{duration_seconds:.1f}s"
            
            table.add_row(
                action.id[:8],
                action.action_type,
                action.status,
                created_str,
                duration,
                "Yes" if action.is_manual else "No"
            )
        
        console.print(table)
    
    asyncio.run(_list_actions())


@app.command()
def simulate_incident(
    title: str = "Test Incident",
    service: str = "test-service",
    severity: str = "medium"
):
    """Simulate an incident for testing."""
    console.print(f"🚨 Simulating incident: {title}", style="bold red")
    
    async def _simulate():
        incident_service = IncidentService()
        
        incident = await incident_service.create_incident(
            title=title,
            description=f"Simulated incident for testing purposes. Service: {service}",
            severity=severity,
            service=service,
            tags=["simulated", "test"]
        )
        
        console.print(f"✅ Created incident: {incident.id}", style="green")
        console.print(f"   Title: {incident.title}")
        console.print(f"   Service: {incident.service}")
        console.print(f"   Severity: {incident.severity}")
        
        # Simulate AI processing
        ai_engine = AIDecisionEngine()
        await ai_engine.queue_incident(incident)
        console.print("🤖 Queued for AI analysis", style="blue")
    
    asyncio.run(_simulate())


@app.command()
def test_action(
    action_type: str = "restart_service",
    service_name: str = "test-service"
):
    """Test an automated action."""
    console.print(f"⚡ Testing action: {action_type}", style="bold yellow")
    
    async def _test_action():
        action_service = ActionService()
        await action_service.start()
        
        action_id = await action_service.execute_action(
            action_type=action_type,
            parameters={"service_name": service_name},
            is_manual=True
        )
        
        console.print(f"✅ Action queued: {action_id}", style="green")
        console.print("Monitor with: python cli.py actions", style="blue")
        
        # Wait a moment for action to complete
        await asyncio.sleep(3)
        
        action = await action_service.get_action(action_id)
        if action:
            console.print(f"Status: {action.status}", style="cyan")
            if action.result:
                console.print(f"Result: {action.result}")
        
        await action_service.stop()
    
    asyncio.run(_test_action())


@app.command()
def monitor_logs(duration: int = 60):
    """Monitor logs for the specified duration (seconds)."""
    console.print(f"👀 Monitoring logs for {duration} seconds...", style="bold blue")
    
    async def _monitor():
        log_monitor = LogMonitorService()
        await log_monitor.start()
        
        console.print("Log monitoring started. Press Ctrl+C to stop.", style="green")
        
        try:
            await asyncio.sleep(duration)
        except KeyboardInterrupt:
            console.print("\nStopping log monitor...", style="yellow")
        finally:
            await log_monitor.stop()
    
    asyncio.run(_monitor())


@app.command()
def start_services():
    """Start all core services for development."""
    console.print("🚀 Starting AI On-Call Agent services...", style="bold blue")
    
    async def _start_services():
        # Initialize services
        log_monitor = LogMonitorService()
        ai_engine = AIDecisionEngine()
        action_service = ActionService()
        
        # Start services
        await log_monitor.start()
        await ai_engine.start()
        await action_service.start()
        
        console.print("✅ All services started successfully!", style="green")
        console.print("Services running:", style="blue")
        console.print("  • Log Monitor")
        console.print("  • AI Decision Engine")
        console.print("  • Action Engine")
        console.print("\nPress Ctrl+C to stop all services...")
        
        try:
            # Keep services running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("\n🛑 Stopping services...", style="yellow")
        finally:
            await action_service.stop()
            await ai_engine.stop()
            await log_monitor.stop()
            console.print("✅ All services stopped.", style="green")
    
    asyncio.run(_start_services())


@app.command()
def test_log_polling():
    """Test the log polling system with sample data."""
    console.print("[bold green]Testing Log Polling System[/bold green]")
    
    try:
        # Create test log directory and file
        import os
        os.makedirs("/tmp/test_logs", exist_ok=True)
        
        # Write some test log entries
        with open("/tmp/test_logs/application.log", "w") as f:
            f.write("2024-07-28 10:30:15 [INFO] web-server: Application started successfully\n")
            f.write("2024-07-28 10:31:22 [ERROR] database: Connection timeout after 30 seconds\n")
            f.write("2024-07-28 10:32:05 [CRITICAL] redis: OutOfMemoryError: heap space exhausted\n")
            f.write("2024-07-28 10:32:30 [ERROR] api-gateway: Service unavailable - health check failed\n")
            f.write("2024-07-28 10:33:10 [INFO] web-server: Processing request completed\n")
        
        console.print("✅ Created test log file with sample entries")
        console.print("📁 File: /tmp/test_logs/application.log")
        console.print("📊 5 log entries created (2 errors, 1 critical)")
        
        # Test pattern matching
        console.print("\n[bold yellow]Testing Pattern Matching:[/bold yellow]")
        test_messages = [
            "Database connection timeout after 30 seconds",
            "OutOfMemoryError: Java heap space exhausted", 
            "Service health check failed for redis-cluster",
            "Application started successfully",
            "CRITICAL: System disk 95% full"
        ]
        
        patterns = [
            ("connection_failed", r"connection.*timeout|connection.*refused|connection.*failed"),
            ("out_of_memory", r"OutOfMemoryError|heap.*space|memory.*exhausted"),
            ("service_unavailable", r"service.*unavailable|server.*not.*responding|health.*check.*failed"),
            ("critical_error", r"CRITICAL|FATAL|ERROR.*failed.*start"),
            ("disk_full", r"disk.*full|no.*space.*left|filesystem.*full")
        ]
        
        import re
        for message in test_messages:
            matches = []
            for pattern_name, pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    matches.append(pattern_name)
            
            if matches:
                console.print(f"🔍 '{message[:40]}...' → [red]{matches}[/red]")
            else:
                console.print(f"🔍 '{message[:40]}...' → [green]No alerts[/green]")
        
        console.print("\n✅ Log polling test completed successfully")
        console.print("💡 Run the application to see real-time log monitoring")
        
    except Exception as e:
        console.print(f"[red]❌ Error testing log polling: {e}[/red]")


@app.command()
def create_test_logs():
    """Create sample log files for testing."""
    import os
    
    console.print("[bold green]Creating test log files...[/bold green]")
    
    try:
        # Create test directory
        os.makedirs("/tmp/test_logs", exist_ok=True)
        
        # Sample log entries with various patterns
        log_entries = [
            "2024-07-28 10:30:15 [INFO] web-server: Application started successfully",
            "2024-07-28 10:31:22 [ERROR] database: Connection timeout after 30 seconds",
            "2024-07-28 10:32:05 [CRITICAL] redis: OutOfMemoryError: heap space exhausted", 
            "2024-07-28 10:32:30 [ERROR] api-gateway: Service unavailable - health check failed",
            "2024-07-28 10:33:10 [INFO] web-server: Processing batch job completed",
            "2024-07-28 10:34:45 [FATAL] spark: Application failed to start - configuration error",
            "2024-07-28 10:35:12 [ERROR] airflow: Task instance failed with exit code 1",
            "2024-07-28 10:36:08 [WARNING] system: Disk space usage at 85%",
            "2024-07-28 10:37:22 [ERROR] network: Connection refused to downstream service",
            "2024-07-28 10:38:01 [INFO] scheduler: Job queue processed successfully"
        ]
        
        # Write to application log
        with open("/tmp/test_logs/application.log", "w") as f:
            for entry in log_entries:
                f.write(entry + "\n")
        
        # Write to error log (only errors)
        with open("/tmp/test_logs/error.log", "w") as f:
            for entry in log_entries:
                if any(level in entry for level in ["ERROR", "CRITICAL", "FATAL"]):
                    f.write(entry + "\n")
        
        console.print("✅ Created test log files:")
        console.print("   📁 /tmp/test_logs/application.log")
        console.print("   📁 /tmp/test_logs/error.log")
        console.print(f"   📊 {len(log_entries)} total entries")
        console.print(f"   ⚠️  {sum(1 for e in log_entries if any(l in e for l in ['ERROR', 'CRITICAL', 'FATAL']))} error entries")
        console.print("\n💡 Now run 'python -m src.main --dev' to see log polling in action!")
        
    except Exception as e:
        console.print(f"[red]❌ Error creating test logs: {e}[/red]")


@app.command()
def test_log_pattern(pattern: str, test_message: str):
    """Test a regex pattern against a log message."""
    import re
    
    console.print(f"[bold blue]Testing Pattern[/bold blue]")
    console.print(f"🔍 Pattern: {pattern}")
    console.print(f"📝 Message: {test_message}")
    
    try:
        match = re.search(pattern, test_message, re.IGNORECASE)
        if match:
            console.print(f"[green]✅ MATCH FOUND[/green]")
            console.print(f"📍 Matched text: '{match.group()}'")
            if match.groups():
                console.print(f"🎯 Captured groups: {match.groups()}")
        else:
            console.print(f"[red]❌ NO MATCH[/red]")
            
    except re.error as e:
        console.print(f"[red]❌ Invalid regex pattern: {e}[/red]")


if __name__ == "__main__":
    setup_logging()
    app()


# =============================================
# ML COMMANDS
# =============================================

@ml_app.command("train")
def train_models(
    data_file: str = typer.Option(..., "--data", "-d", help="Training data JSON file"),
    output_file: str = typer.Option(..., "--output", "-o", help="Output model file"),
    min_samples: int = typer.Option(50, "--min-samples", help="Minimum training samples"),
    export_report: str = typer.Option(None, "--report", help="Export training report"),
):
    """Train ML models for incident classification."""
    console.print("🎓 Training AI Models", style="bold blue")
    
    async def _train_models():
        try:
            # Load training data
            console.print(f"📊 Loading training data from {data_file}")
            
            if not Path(data_file).exists():
                console.print(f"❌ Training data file not found: {data_file}", style="red")
                return
            
            with open(data_file, 'r') as f:
                training_data = json.load(f)
            
            ai_engine = AIDecisionEngine()
            
            # Add training samples
            valid_samples = 0
            for sample in training_data:
                try:
                    incident = IncidentCreate(**sample["incident"])
                    ai_engine.add_training_data(
                        incident=incident,
                        outcome=sample["outcome"],
                        resolution_time=sample.get("resolution_time", 300),
                        success=sample.get("success", True)
                    )
                    valid_samples += 1
                except Exception as e:
                    console.print(f"⚠️  Skipping invalid sample: {e}", style="yellow")
                    continue
            
            console.print(f"📈 Loaded {valid_samples} training samples")
            
            if valid_samples < min_samples:
                console.print(f"❌ Need at least {min_samples} samples, got {valid_samples}", style="red")
                return
            
            # Train models
            console.print("🔄 Training models...")
            results = ai_engine.train_models(min_samples=min_samples)
            
            if results["success"]:
                # Display results
                console.print("✅ Training completed successfully!", style="green")
                console.print(f"📊 Training samples: {results['training_samples']}")
                
                eval_metrics = results.get("evaluation", {})
                if "classification_accuracy" in eval_metrics:
                    console.print(f"🎯 Classification accuracy: {eval_metrics['classification_accuracy']:.3f}")
                if "confidence_accuracy" in eval_metrics:
                    console.print(f"🔮 Confidence accuracy: {eval_metrics['confidence_accuracy']:.3f}")
                
                # Save model
                console.print(f"💾 Saving model to {output_file}")
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                
                if ai_engine.save_model(output_file):
                    console.print("✅ Model saved successfully!", style="green")
                    
                    # Create latest symlink
                    latest_path = Path(output_file).parent / "latest.pkl"
                    try:
                        if latest_path.exists():
                            latest_path.unlink()
                        latest_path.symlink_to(Path(output_file).name)
                        console.print(f"🔗 Created symlink: {latest_path}")
                    except Exception as e:
                        console.print(f"⚠️  Could not create symlink: {e}", style="yellow")
                else:
                    console.print("❌ Failed to save model", style="red")
                    return
                
                # Export report if requested
                if export_report:
                    console.print(f"📄 Exporting report to {export_report}")
                    Path(export_report).parent.mkdir(parents=True, exist_ok=True)
                    
                    if ai_engine.export_model_summary(export_report):
                        console.print("✅ Report exported successfully!", style="green")
                    else:
                        console.print("⚠️  Failed to export report", style="yellow")
                
                # Display feature importance
                feature_importance = ai_engine._get_feature_importance()
                if feature_importance:
                    console.print("\n🔍 Top 5 Important Features:")
                    table = Table(show_header=True)
                    table.add_column("Rank", style="cyan")
                    table.add_column("Feature", style="white")
                    table.add_column("Importance", style="green")
                    
                    for i, feat in enumerate(feature_importance[:5]):
                        table.add_row(
                            str(i + 1),
                            feat["feature"],
                            f"{feat['importance']:.4f}"
                        )
                    console.print(table)
                
            else:
                console.print(f"❌ Training failed: {results['error']}", style="red")
                
        except Exception as e:
            console.print(f"❌ Error during training: {e}", style="red")
            import traceback
            traceback.print_exc()
    
    asyncio.run(_train_models())


@ml_app.command("evaluate")
def evaluate_model(
    model_file: str = typer.Option(..., "--model", "-m", help="Trained model file"),
    test_data: str = typer.Option(None, "--test-data", "-t", help="Test data JSON file"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive testing mode"),
    benchmark: int = typer.Option(None, "--benchmark", "-b", help="Performance benchmark iterations"),
):
    """Evaluate trained ML models."""
    console.print("🔬 Evaluating AI Model", style="bold blue")
    
    try:
        # Load model
        console.print(f"📂 Loading model from {model_file}")
        ai_engine = AIDecisionEngine()
        
        if not ai_engine.load_model(model_file):
            console.print(f"❌ Failed to load model: {model_file}", style="red")
            return
        
        model_info = ai_engine.get_model_info()
        console.print(f"✅ Model loaded successfully", style="green")
        console.print(f"📊 Training samples: {model_info['training_samples']}")
        console.print(f"🎯 Model accuracy: {model_info['metadata'].get('accuracy', 0):.3f}")
        
        # Test data evaluation
        if test_data:
            console.print(f"\n📋 Evaluating on test data: {test_data}")
            
            if not Path(test_data).exists():
                console.print(f"❌ Test data file not found: {test_data}", style="red")
                return
            
            with open(test_data, 'r') as f:
                test_samples = json.load(f)
            
            correct_predictions = 0
            total_predictions = len(test_samples)
            
            console.print(f"🧪 Testing on {total_predictions} samples...")
            
            results_table = Table(show_header=True)
            results_table.add_column("Title", style="white", max_width=30)
            results_table.add_column("Expected", style="cyan")
            results_table.add_column("Predicted", style="yellow")
            results_table.add_column("Confidence", style="green")
            results_table.add_column("Correct", style="white")
            
            for i, sample in enumerate(test_samples):
                try:
                    incident = IncidentCreate(**sample["incident"])
                    expected = sample.get("expected_category", "unknown")
                    
                    predicted_category, confidence = ai_engine.predict_incident_category(incident)
                    resolution_conf = ai_engine.predict_resolution_confidence(incident)
                    
                    is_correct = predicted_category.lower() == expected.lower()
                    if is_correct:
                        correct_predictions += 1
                    
                    # Add to results table (show first 10)
                    if i < 10:
                        results_table.add_row(
                            incident.title[:30] + "..." if len(incident.title) > 30 else incident.title,
                            expected,
                            predicted_category,
                            f"{confidence:.3f}",
                            "✅" if is_correct else "❌"
                        )
                        
                except Exception as e:
                    console.print(f"⚠️  Error testing sample {i}: {e}", style="yellow")
                    continue
            
            console.print(results_table)
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            console.print(f"\n📊 Overall Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        
        # Performance benchmark
        if benchmark:
            console.print(f"\n🏃 Running performance benchmark ({benchmark} iterations)")
            
            # Create sample incident
            sample_incident = IncidentCreate(
                title="Performance Test",
                description="Database connection timeout after 30 seconds",
                service="postgres",
                severity="high",
                tags=["database", "timeout"]
            )
            
            import time
            times = []
            
            for i in range(benchmark):
                start = time.time()
                ai_engine.predict_incident_category(sample_incident)
                ai_engine.predict_resolution_confidence(sample_incident)
                ai_engine.detect_anomalies(sample_incident)
                times.append((time.time() - start) * 1000)
                
                if (i + 1) % 20 == 0:
                    avg_time = sum(times) / len(times)
                    console.print(f"  Progress: {i + 1}/{benchmark}, avg: {avg_time:.2f}ms")
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            console.print(f"⏱️  Average prediction time: {avg_time:.2f}ms")
            console.print(f"📈 Min/Max: {min_time:.2f}ms / {max_time:.2f}ms")
            console.print(f"🚀 Throughput: {1000/avg_time:.1f} predictions/second")
        
        # Interactive mode
        if interactive:
            console.print("\n🎮 Interactive Testing Mode")
            console.print("Enter incident details to test predictions (Ctrl+C to exit)")
            
            while True:
                try:
                    title = typer.prompt("Incident Title")
                    description = typer.prompt("Description")
                    service = typer.prompt("Service", default="unknown")
                    severity = typer.prompt("Severity", default="medium")
                    tags_input = typer.prompt("Tags (comma-separated)", default="")
                    
                    tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
                    
                    incident = IncidentCreate(
                        title=title,
                        description=description,
                        service=service,
                        severity=severity,
                        tags=tags
                    )
                    
                    # Predictions
                    category, cat_conf = ai_engine.predict_incident_category(incident)
                    res_conf = ai_engine.predict_resolution_confidence(incident)
                    anomaly = ai_engine.detect_anomalies(incident)
                    
                    console.print(f"\n🤖 AI Analysis:")
                    console.print(f"📊 Category: {category} (confidence: {cat_conf:.3f})")
                    console.print(f"✅ Resolution confidence: {res_conf:.3f}")
                    console.print(f"🚨 Anomaly: {'Yes' if anomaly['is_anomaly'] else 'No'}")
                    
                    if res_conf > 0.8:
                        console.print("💡 Recommendation: Automated resolution", style="green")
                    else:
                        console.print("💡 Recommendation: Manual intervention", style="yellow")
                    
                    console.print()
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"❌ Error: {e}", style="red")
        
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}", style="red")


@ml_app.command("info")
def model_info(
    model_file: str = typer.Option(..., "--model", "-m", help="Model file to inspect")
):
    """Show information about a trained model."""
    console.print("ℹ️  Model Information", style="bold blue")
    
    try:
        ai_engine = AIDecisionEngine()
        
        if not ai_engine.load_model(model_file):
            console.print(f"❌ Failed to load model: {model_file}", style="red")
            return
        
        info = ai_engine.get_model_info()
        metadata = info.get("metadata", {})
        
        # Basic info table
        info_table = Table(show_header=True)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Model File", model_file)
        info_table.add_row("Version", metadata.get("version", "unknown"))
        info_table.add_row("Trained At", metadata.get("trained_at", "unknown"))
        info_table.add_row("Training Samples", str(info.get("training_samples", 0)))
        info_table.add_row("Accuracy", f"{metadata.get('accuracy', 0):.3f}")
        info_table.add_row("Has Classifier", "✅" if info.get("has_classifier") else "❌")
        info_table.add_row("Has Confidence Model", "✅" if info.get("has_confidence_model") else "❌")
        info_table.add_row("Has Clustering", "✅" if info.get("has_clustering") else "❌")
        info_table.add_row("Feature Count", str(info.get("feature_count", 0)))
        
        console.print(info_table)
        
        # Feature importance
        feature_importance = ai_engine._get_feature_importance()
        if feature_importance:
            console.print(f"\n🔍 Top 10 Important Features:")
            feat_table = Table(show_header=True)
            feat_table.add_column("Rank", style="cyan")
            feat_table.add_column("Feature", style="white")
            feat_table.add_column("Importance", style="green")
            
            for i, feat in enumerate(feature_importance[:10]):
                feat_table.add_row(
                    str(i + 1),
                    feat["feature"],
                    f"{feat['importance']:.4f}"
                )
            
            console.print(feat_table)
        
        # Label classes
        label_classes = info.get("label_classes", [])
        if label_classes:
            console.print(f"\n📋 Label Classes: {', '.join(label_classes)}")
        
    except Exception as e:
        console.print(f"❌ Error reading model info: {e}", style="red")


@ml_app.command("predict")
def predict_incident(
    model_file: str = typer.Option(..., "--model", "-m", help="Trained model file"),
    title: str = typer.Option(..., "--title", "-t", help="Incident title"),
    description: str = typer.Option(..., "--description", "-d", help="Incident description"),
    service: str = typer.Option("unknown", "--service", "-s", help="Service name"),
    severity: str = typer.Option("medium", "--severity", help="Incident severity"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
):
    """Make predictions for a single incident."""
    console.print("🔮 Making Prediction", style="bold blue")
    
    try:
        # Load model
        ai_engine = AIDecisionEngine()
        if not ai_engine.load_model(model_file):
            console.print(f"❌ Failed to load model: {model_file}", style="red")
            return
        
        # Create incident
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        incident = IncidentCreate(
            title=title,
            description=description,
            service=service,
            severity=severity,
            tags=tag_list
        )
        
        # Make predictions
        category, cat_confidence = ai_engine.predict_incident_category(incident)
        resolution_confidence = ai_engine.predict_resolution_confidence(incident)
        anomaly_info = ai_engine.detect_anomalies(incident)
        
        # Display results
        result_table = Table(show_header=True)
        result_table.add_column("Prediction", style="cyan")
        result_table.add_column("Value", style="white")
        result_table.add_column("Confidence", style="green")
        
        result_table.add_row("Category", category, f"{cat_confidence:.3f}")
        result_table.add_row("Resolution Success", "High" if resolution_confidence > 0.8 else "Medium" if resolution_confidence > 0.6 else "Low", f"{resolution_confidence:.3f}")
        result_table.add_row("Anomaly Detection", "Yes" if anomaly_info["is_anomaly"] else "No", f"{anomaly_info.get('anomaly_score', 0):.3f}")
        
        console.print(result_table)
        
        # Recommendations
        console.print(f"\n💡 Recommendations:")
        if resolution_confidence > 0.8 and not anomaly_info["is_anomaly"]:
            console.print("✅ Recommend automated resolution", style="green")
        elif resolution_confidence > 0.6:
            console.print("⚠️  Consider automated resolution with monitoring", style="yellow")
        else:
            console.print("🚨 Recommend manual intervention", style="red")
        
    except Exception as e:
        console.print(f"❌ Prediction failed: {e}", style="red")


@ml_app.command("generate-sample-data")
def generate_sample_data(
    output_file: str = typer.Option(..., "--output", "-o", help="Output file path"),
    count: int = typer.Option(100, "--count", "-c", help="Number of samples to generate"),
):
    """Generate sample training data for testing."""
    console.print("🎲 Generating Sample Data", style="bold blue")
    
    import random
    
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
        }
    ]
    
    samples = []
    for i in range(count):
        template = random.choice(sample_templates)
        sample = json.loads(json.dumps(template))  # Deep copy
        
        # Add variation
        sample["resolution_time"] = random.randint(60, 600)
        sample["success"] = random.random() > 0.15  # 85% success rate
        
        samples.append(sample)
    
    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    console.print(f"✅ Generated {count} samples: {output_file}", style="green")
