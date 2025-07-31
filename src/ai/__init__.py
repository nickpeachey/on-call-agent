"""AI decision engine for automated issue resolution."""

import asyncio
import pickle
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.sparse
import random

from ..core import get_logger, settings
from ..models.schemas import LogEntry, IncidentCreate, ActionCreate
from ..services.knowledge_base import KnowledgeBaseEntry
from ..services.action_logger import action_logger
from ..services.ml_service import MLService
from ..database import get_db_session, Base
from sqlalchemy.orm import Session
from sqlalchemy import text, Column, String, Float, Boolean, DateTime, JSON
from sqlalchemy.ext.asyncio import AsyncSession


logger = get_logger(__name__)


class TrainingData(Base):
    """Database model for storing ML training data from resolved incidents."""
    __tablename__ = "training_data"
    
    id = Column(String, primary_key=True)
    incident_id = Column(String, nullable=True)
    incident_title = Column(String, nullable=False)
    incident_service = Column(String, nullable=False)
    incident_severity = Column(String, nullable=False)
    incident_description = Column(String, nullable=False)
    features = Column(JSON, nullable=False)  # Extracted ML features
    outcome = Column(String, nullable=False)  # "resolved", "escalated", etc.
    resolution_time = Column(Float, nullable=False)  # Time in seconds
    success = Column(Boolean, nullable=False)  # Was resolution successful
    confidence_score = Column(Float, nullable=True)  # AI confidence at time of resolution
    actions_executed = Column(JSON, nullable=True)  # List of actions that were executed
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)


class AIDecisionEngine:
    """AI-powered decision engine for automated issue resolution with ML capabilities."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.is_running = False
        self.decision_task = None
        self.incident_queue = asyncio.Queue()
        self.knowledge_base_service = None  # Will be injected
        self.action_service = None  # Will be injected
        
        # ML Service integration
        self.ml_service = MLService()
        
        # ML Models
        self.incident_classifier = None
        self.confidence_model = None
        self.pattern_clustering = None
        self.feature_vectorizer = None
        self.label_encoders = {}
        
        # Model metadata
        self.model_metadata = {
            "version": "1.0.0",
            "trained_at": None,
            "training_samples": 0,
            "accuracy": 0.0,
            "feature_names": [],
            "label_classes": []
        }
        
        # Training data storage
        self.training_data = {
            "incidents": [],
            "outcomes": [],
            "features": [],
            "labels": []
        }
        
        # Load existing model if provided
        if model_path:
            self.load_model(model_path)
    
    async def start(self):
        """Start the AI decision engine."""
        if not self.is_running:
            self.is_running = True
            
            # Initialize ML service
            await self.ml_service.initialize()
            logger.info("🤖 ML Service initialized")
            
            # Try to load existing trained models first
            model_path = "models/ai_decision_engine.pkl"
            model_loaded = self.load_model(model_path)
            
            if model_loaded:
                logger.info("🎯 Loaded existing trained models", 
                           model_path=model_path,
                           version=self.model_metadata.get("version", "unknown"),
                           accuracy=self.model_metadata.get("accuracy", 0))
            else:
                logger.info("🔍 No existing models found, will train from scratch")
            
            # Load training data from database
            await self._load_training_data_from_db()
            
            # Train models if we have enough data and either no models exist or models are outdated
            should_train = False
            reason = "no training needed"
            samples_available = len(self.training_data["incidents"])
            
            if not model_loaded and samples_available >= 10:
                should_train = True
                reason = "no existing models found"
            elif model_loaded and samples_available >= 50:
                # Check if models are outdated (more than 100 new samples since last training)
                last_training_samples = self.model_metadata.get("training_samples", 0)
                new_samples = samples_available - last_training_samples
                if new_samples >= 100:
                    should_train = True
                    reason = f"models outdated - {new_samples} new samples available"
            
            if should_train:
                logger.info("🧠 Training models on startup", 
                           samples=samples_available, reason=reason)
                results = self.train_models(min_samples=10)
                
                if results["success"]:
                    # Save the newly trained models
                    save_success = self.save_model(model_path)
                    logger.info("✅ Models trained and saved successfully", 
                               accuracy=results.get("evaluation", {}).get("classification_accuracy", 0),
                               model_saved=save_success,
                               model_path=model_path)
                else:
                    logger.warning("⚠️ Model training failed on startup", 
                                 error=results.get("error", "Unknown"))
            else:
                logger.info("🤖 Using existing models or insufficient data for training",
                           samples=samples_available, 
                           has_models=model_loaded,
                           required_for_training=10)
            
            self.decision_task = asyncio.create_task(self._decision_loop())
            logger.info("🤖 AI Decision Engine started - loop task created", task_id=id(self.decision_task))
            # Give the task a moment to start
            await asyncio.sleep(0.1)
            if self.decision_task.done():
                exception = self.decision_task.exception()
                logger.error("❌ AI Decision Loop task failed immediately!", exception=exception)
                raise exception if exception else RuntimeError("Decision loop task completed unexpectedly")
            logger.info("✅ AI Decision Engine confirmed running", is_running=self.is_running)
    
    async def stop(self):
        """Stop the AI decision engine."""
        self.is_running = False
        if self.decision_task:
            self.decision_task.cancel()
            try:
                await self.decision_task
            except asyncio.CancelledError:
                pass
            self.decision_task = None
            logger.info("AI Decision Engine stopped")
    
    async def _decision_loop(self):
        """Main decision loop for processing incidents."""
        logger.info("🚀 AI Decision Engine loop started - entering main processing loop")
        
        while self.is_running:
            try:
                # Wait for incident to process (with timeout)
                try:
                    logger.debug("🔍 About to call incident_queue.get() with 2s timeout... queue size: {}".format(self.incident_queue.qsize()))
                    incident = await asyncio.wait_for(
                        self.incident_queue.get(),
                        timeout=2.0
                    )
                    logger.info("🔥 GOT INCIDENT FROM QUEUE!", title=getattr(incident, 'title', 'NO_TITLE'), 
                               incident_type=type(incident).__name__, queue_size_after=self.incident_queue.qsize())
                    try:
                        await self._process_incident(incident)
                        logger.info("✅ INCIDENT PROCESSING COMPLETED", title=getattr(incident, 'title', 'NO_TITLE'))
                    except Exception as process_error:
                        logger.error("💥 ERROR PROCESSING INCIDENT", title=getattr(incident, 'title', 'NO_TITLE'), 
                                   error=str(process_error), exc_info=True)
                except asyncio.TimeoutError:
                    # No incident to process, continue loop - only log this in debug mode
                    logger.debug("⏰ No incident received in 2s timeout, queue size: {}".format(self.incident_queue.qsize()))
                    continue
                    
            except asyncio.CancelledError:
                logger.info("🛑 AI Decision Engine loop cancelled")
                break
            except Exception as e:
                logger.error("💥 ERROR in AI decision loop", error=str(e), exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info("🏁 AI Decision Engine loop exited - is_running: {}".format(self.is_running))
    
    async def queue_incident(self, incident: IncidentCreate):
        """Queue an incident for AI analysis and resolution."""
        logger.info("📥 Queuing incident for AI analysis", title=incident.title)
        await self.incident_queue.put(incident)
        logger.debug("✅ Incident queued successfully", queue_size=self.incident_queue.qsize(), loop_running=self.is_running)
    
    async def _process_incident(self, incident: IncidentCreate):
        """Process an incident and determine appropriate actions."""
        logger.info("🚀 STARTING INCIDENT PROCESSING", title=incident.title, severity=incident.severity)
        
        try:
            # Step 1: Analyze the incident using AI
            if not settings.quiet_mode:
                logger.info("📊 Step 1: Starting AI analysis", title=incident.title)
            analysis = await self._analyze_incident(incident)
            if not settings.quiet_mode:
                logger.info("✅ Step 1: AI analysis complete", title=incident.title)
            
            # Step 2: Find matching knowledge base entries
            kb_matches = await self._find_knowledge_base_matches(incident, analysis)
            
            # Step 3: Determine if automated action should be taken
            should_automate, confidence, recommended_actions = await self._should_automate_resolution(
                incident, analysis, kb_matches
            )
            
            logger.info(
                "AI analysis complete",
                incident_title=incident.title,
                should_automate=should_automate,
                confidence=confidence,
                num_kb_matches=len(kb_matches),
                num_actions=len(recommended_actions)
            )
            
            # Step 4: Execute actions if confidence is high enough
            if should_automate and confidence > 0.6:  # 60% confidence threshold
                logger.info(
                    "🤖 AUTOMATED RESOLUTION TRIGGERED",
                    incident_title=incident.title,
                    confidence=f"{confidence:.2f}",
                    action_count=len(recommended_actions),
                    threshold="0.60"
                )
                success = await self._execute_automated_actions(incident, recommended_actions)
                
                if success:
                    logger.info(
                        "✅ AUTOMATED RESOLUTION SUCCESSFUL",
                        incident_title=incident.title,
                        resolution_time="<2min",
                        actions_executed=len(recommended_actions)
                    )
                else:
                    logger.error(
                        "❌ AUTOMATED RESOLUTION FAILED - Escalating to manual intervention",
                        incident_title=incident.title,
                        actions_attempted=len(recommended_actions)
                    )
                    await self._create_manual_intervention_alert(incident, analysis, "Automated resolution failed")
            else:
                # Log why automation was skipped
                if not should_automate:
                    if confidence <= 0.6:
                        reason = f"Confidence {confidence:.2f} below threshold 0.60"
                    elif len(recommended_actions) == 0:
                        reason = "No automated actions available"
                    elif incident.severity == "critical":
                        reason = "Critical severity - manual review required"
                    else:
                        reason = "Automation decision: not recommended"
                else:
                    reason = f"Confidence {confidence:.2f} below execution threshold 0.60"
                    
                logger.warning(
                    "⚠️  MANUAL INTERVENTION REQUIRED",
                    incident_title=incident.title,
                    reason=reason,
                    confidence=f"{confidence:.2f}",
                    should_automate=should_automate,
                    action_count=len(recommended_actions),
                    severity=incident.severity
                )
                
                await self._create_manual_intervention_alert(incident, analysis, reason)
        
        except Exception as e:
            logger.error("Error processing incident with AI", incident_title=incident.title, error=str(e))
    
    async def _analyze_incident(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Analyze incident using AI/ML to extract key information."""
        if not settings.quiet_mode:
            logger.info("🔬 Starting incident analysis", title=incident.title)
        
        try:
            # Use ML Service for predictions (uses saved models from disk)
            root_cause_category, category_confidence = await self.predict_incident_category(incident)
            resolution_confidence = self.predict_resolution_confidence(incident)
            anomaly_info = self.detect_anomalies(incident)
            
            ml_analysis = {
                "root_cause_category": root_cause_category,
                "category_confidence": category_confidence,
                "affected_components": self._extract_affected_components(incident),
                "error_patterns": self._extract_error_patterns(incident),
                "recommended_action_types": await self._recommend_action_types(incident),
                "risk_assessment": self._assess_risk(incident),
                "confidence_score": resolution_confidence,
                "anomaly_detection": anomaly_info,
                "analysis_method": "ml_service_models",
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("ML Service incident analysis completed",
                       category=root_cause_category,
                       confidence=resolution_confidence,
                       is_anomaly=anomaly_info.get("is_anomaly", False))
            
            return ml_analysis
                
        except Exception as e:
            logger.error("Error in ML analysis, falling back to rule-based", error=str(e))
        
        # Fallback to rule-based analysis
        analysis_prompt = f"""
        Analyze this incident and extract key information:
        
        Title: {incident.title}
        Description: {incident.description}
        Service: {incident.service}
        Severity: {incident.severity}
        Tags: {', '.join(incident.tags)}
        
        Extract:
        1. Root cause category
        2. Affected components
        3. Error patterns
        4. Recommended action types
        5. Risk assessment
        """
        
        # Rule-based analysis
        rule_based_analysis = {
            "root_cause_category": self._determine_root_cause_category(incident),
            "affected_components": self._extract_affected_components(incident),
            "error_patterns": self._extract_error_patterns(incident),
            "recommended_action_types": self._recommend_action_types(incident),
            "risk_assessment": self._assess_risk(incident),
            "confidence_score": 0.75,  # Increased confidence for rule-based matching
            "analysis_method": "rule_based",
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        return rule_based_analysis
    
    def _determine_root_cause_category(self, incident: IncidentCreate) -> str:
        """Determine the root cause category based on incident details."""
        description_lower = incident.description.lower()
        
        if any(term in description_lower for term in ["connection", "timeout", "database"]):
            return "database_connectivity"
        elif any(term in description_lower for term in ["memory", "oom", "heap"]):
            return "memory_issues"
        elif any(term in description_lower for term in ["airflow", "dag", "task"]):
            return "workflow_failure"
        elif any(term in description_lower for term in ["spark", "executor", "yarn"]):
            return "compute_failure"
        elif any(term in description_lower for term in ["file", "not found", "missing"]):
            return "data_availability"
        else:
            return "unknown"
    
    def _extract_affected_components(self, incident: IncidentCreate) -> List[str]:
        """Extract affected components from incident."""
        components = [incident.service]
        
        description_lower = incident.description.lower()
        
        # Add components based on description
        if "database" in description_lower:
            components.append("database")
        if "airflow" in description_lower:
            components.append("airflow")
        if "spark" in description_lower:
            components.append("spark")
        if "api" in description_lower:
            components.append("api")
        
        return list(set(components))  # Remove duplicates
    
    async def _extract_enhanced_metadata(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Extract enhanced service-specific metadata for detailed resolution."""
        metadata = {"extraction_timestamp": datetime.utcnow().isoformat()}
        
        # Extract DAG information from Airflow incidents
        if incident.service == "airflow":
            metadata.update(self._extract_airflow_metadata(incident))
        
        # Extract database connection details
        elif incident.service in ["postgres", "mysql", "mongodb", "redis"]:
            metadata.update(self._extract_database_metadata(incident))
        
        # Extract Spark application details
        elif incident.service == "spark":
            metadata.update(self._extract_spark_metadata(incident))
        
        # Extract Kubernetes details
        elif incident.service in ["kubernetes", "k8s", "docker"]:
            metadata.update(self._extract_kubernetes_metadata(incident))
        
        # Extract disk/storage details
        elif "disk" in incident.tags or "storage" in incident.tags:
            metadata.update(self._extract_storage_metadata(incident))
        
        logger.info(f"🔍 Extracted enhanced metadata", 
                   service=incident.service, 
                   metadata_keys=list(metadata.keys()))
        
        return metadata
    
    def _extract_airflow_metadata(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Extract Airflow DAG-specific information."""
        metadata = {}
        
        # Extract DAG ID from description or title
        dag_patterns = [
            r"DAG[:\s]+([a-zA-Z0-9_\-\.]+)",
            r"dag_id[:\s=]+([a-zA-Z0-9_\-\.]+)",
            r"([a-zA-Z0-9_\-\.]+)[\s]+DAG"
        ]
        
        text = f"{incident.title} {incident.description}"
        for pattern in dag_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["dag_id"] = match.group(1)
                logger.debug(f"🎯 Extracted DAG ID: {metadata['dag_id']}")
                break
        
        # Extract task information
        task_patterns = [
            r"task[:\s]+([a-zA-Z0-9_\-\.]+)",
            r"task_id[:\s=]+([a-zA-Z0-9_\-\.]+)"
        ]
        
        for pattern in task_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["task_id"] = match.group(1)
                logger.debug(f"📝 Extracted task ID: {metadata['task_id']}")
                break
        
        # Extract execution date/time patterns
        datetime_patterns = [
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
            r"execution_date[:\s=]+([^\s]+)"
        ]
        
        for pattern in datetime_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["execution_date"] = match.group(1)
                logger.debug(f"📅 Extracted execution date: {metadata['execution_date']}")
                break
        
        # Extract state information
        if "stuck" in text.lower() or "timeout" in text.lower():
            metadata["state"] = "running"
            metadata["issue_type"] = "timeout"
        elif "failed" in text.lower():
            metadata["state"] = "failed"
            metadata["issue_type"] = "failure"
        
        # Extract duration information
        duration_patterns = [
            r"(\d+)[\s]*minutes?",
            r"(\d+)[\s]*hours?",
            r"(\d+)[\s]*seconds?"
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "minute" in pattern:
                    metadata["duration_minutes"] = int(match.group(1))
                elif "hour" in pattern:
                    metadata["duration_hours"] = int(match.group(1))
                elif "second" in pattern:
                    metadata["duration_seconds"] = int(match.group(1))
                break
        
        return metadata
    
    def _extract_database_metadata(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Extract database connection-specific information."""
        metadata = {}
        
        text = f"{incident.title} {incident.description}"
        
        # Extract database host/server
        host_patterns = [
            r"([a-zA-Z0-9\-\.]+):(\d+)",  # host:port
            r"server[:\s]+([a-zA-Z0-9\-\.]+)",
            r"host[:\s]+([a-zA-Z0-9\-\.]+)"
        ]
        
        for pattern in host_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if ":" in match.group(0):
                    host, port = match.group(1), match.group(2)
                    metadata["database_host"] = host
                    metadata["database_port"] = int(port)
                    logger.debug(f"🌐 Extracted DB host:port: {host}:{port}")
                else:
                    metadata["database_host"] = match.group(1)
                    logger.debug(f"🌐 Extracted DB host: {match.group(1)}")
                break
        
        # Extract database name
        db_patterns = [
            r"database[:\s]+([a-zA-Z0-9_\-]+)",
            r"db[:\s]+([a-zA-Z0-9_\-]+)",
            r"/([a-zA-Z0-9_\-]+)[\s]"  # from connection strings
        ]
        
        for pattern in db_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["database_name"] = match.group(1)
                logger.debug(f"💾 Extracted DB name: {metadata['database_name']}")
                break
        
        # Extract connection pool information
        if "pool" in text.lower():
            pool_patterns = [
                r"pool[:\s]+([a-zA-Z0-9_\-]+)",
                r"(\d+)[:\s]*connections?"
            ]
            
            for pattern in pool_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if pattern.endswith("connections?"):
                        metadata["connection_count"] = int(match.group(1))
                        logger.debug(f"🔗 Extracted connection count: {metadata['connection_count']}")
                    else:
                        metadata["connection_pool_name"] = match.group(1)
                        logger.debug(f"🏊 Extracted pool name: {metadata['connection_pool_name']}")
        
        # Extract timeout information
        timeout_match = re.search(r"(\d+)[:\s]*seconds?", text, re.IGNORECASE)
        if timeout_match:
            metadata["timeout_duration"] = int(timeout_match.group(1))
            logger.debug(f"⏱️ Extracted timeout: {metadata['timeout_duration']}s")
        
        # Extract error codes
        error_code_patterns = [
            r"error[:\s]+(\d+)",
            r"code[:\s]+(\d+)",
            r"(\d{5})"  # 5-digit error codes
        ]
        
        for pattern in error_code_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["error_code"] = match.group(1)
                logger.debug(f"❌ Extracted error code: {metadata['error_code']}")
                break
        
        return metadata
    
    def _extract_spark_metadata(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Extract Spark application-specific information."""
        metadata = {}
        
        text = f"{incident.title} {incident.description}"
        
        # Extract application ID
        app_id_pattern = r"application_(\d+_\d+)"
        match = re.search(app_id_pattern, text)
        if match:
            metadata["application_id"] = f"application_{match.group(1)}"
            logger.debug(f"🔥 Extracted Spark app ID: {metadata['application_id']}")
        
        # Extract executor information
        executor_patterns = [
            r"executor[:\s\-]+([a-zA-Z0-9\-]+)",
            r"spark-executor-(\d+)"
        ]
        
        for pattern in executor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["executor_id"] = match.group(0)
                logger.debug(f"⚡ Extracted executor ID: {metadata['executor_id']}")
                break
        
        # Extract memory information
        memory_patterns = [
            r"(\d+)g[\s]*heap",
            r"(\d+)GB[\s]*memory",
            r"executor.*?(\d+)g"
        ]
        
        for pattern in memory_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["executor_memory"] = f"{match.group(1)}g"
                logger.debug(f"🧠 Extracted executor memory: {metadata['executor_memory']}")
                break
        
        # Extract stage/task information
        stage_match = re.search(r"stage[:\s]+(\d+)", text, re.IGNORECASE)
        if stage_match:
            metadata["stage_id"] = int(stage_match.group(1))
            logger.debug(f"🎭 Extracted stage ID: {metadata['stage_id']}")
        
        task_match = re.search(r"task[:\s]+(\d+)", text, re.IGNORECASE)
        if task_match:
            metadata["task_id"] = int(task_match.group(1))
            logger.debug(f"📋 Extracted task ID: {metadata['task_id']}")
        
        # Detect issue type
        if "outofmemoryerror" in text.lower() or "oom" in text.lower():
            metadata["issue_type"] = "out_of_memory"
        elif "timeout" in text.lower():
            metadata["issue_type"] = "timeout"
        elif "connection" in text.lower():
            metadata["issue_type"] = "connection"
        
        return metadata
    
    def _extract_kubernetes_metadata(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Extract Kubernetes/container-specific information."""
        metadata = {}
        
        text = f"{incident.title} {incident.description}"
        
        # Extract pod/container names
        pod_patterns = [
            r"pod[:\s]+([a-zA-Z0-9\-]+)",
            r"container[:\s]+([a-zA-Z0-9\-]+)"
        ]
        
        for pattern in pod_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "pod" in pattern:
                    metadata["pod_name"] = match.group(1)
                    logger.debug(f"🥜 Extracted pod name: {metadata['pod_name']}")
                else:
                    metadata["container_name"] = match.group(1)
                    logger.debug(f"📦 Extracted container name: {metadata['container_name']}")
        
        # Extract namespace
        namespace_match = re.search(r"namespace[:\s]+([a-zA-Z0-9\-]+)", text, re.IGNORECASE)
        if namespace_match:
            metadata["namespace"] = namespace_match.group(1)
            logger.debug(f"🏷️ Extracted namespace: {metadata['namespace']}")
        
        # Extract resource limits
        memory_limit_match = re.search(r"memory.*?(\d+[MGT]i?B?)", text, re.IGNORECASE)
        if memory_limit_match:
            metadata["memory_limit"] = memory_limit_match.group(1)
            logger.debug(f"💾 Extracted memory limit: {metadata['memory_limit']}")
        
        # Detect restart/crash reasons
        if "oomkilled" in text.lower():
            metadata["reason"] = "OOMKilled"
            metadata["issue_type"] = "memory_limit_exceeded"
        elif "crashloopbackoff" in text.lower():
            metadata["reason"] = "CrashLoopBackOff"
            metadata["issue_type"] = "crash_loop"
        
        return metadata
    
    def _extract_storage_metadata(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Extract disk/storage-specific information."""
        metadata = {}
        
        text = f"{incident.title} {incident.description}"
        
        # Extract filesystem path
        path_patterns = [
            r"([/][a-zA-Z0-9/\-\_\.]*)",
            r"filesystem[:\s]+([/][a-zA-Z0-9/\-\_\.]*)"
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["filesystem_path"] = match.group(1)
                logger.debug(f"📁 Extracted filesystem path: {metadata['filesystem_path']}")
                break
        
        # Extract server/hostname
        server_patterns = [
            r"server[:\s]+([a-zA-Z0-9\-\.]+)",
            r"host[:\s]+([a-zA-Z0-9\-\.]+)"
        ]
        
        for pattern in server_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["server_hostname"] = match.group(1)
                logger.debug(f"🖥️ Extracted server hostname: {metadata['server_hostname']}")
                break
        
        # Extract disk usage percentages
        percentage_match = re.search(r"(\d+)%[\s]*full", text, re.IGNORECASE)
        if percentage_match:
            metadata["usage_percentage"] = int(percentage_match.group(1))
            logger.debug(f"📊 Extracted usage percentage: {metadata['usage_percentage']}%")
        
        # Extract size information
        size_patterns = [
            r"(\d+)GB[\s]*remaining",
            r"(\d+)GB[\s]*free",
            r"(\d+)GB[\s]*used"
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "remaining" in pattern or "free" in pattern:
                    metadata["available_size_gb"] = int(match.group(1))
                    logger.debug(f"💿 Extracted available size: {metadata['available_size_gb']}GB")
                else:
                    metadata["used_size_gb"] = int(match.group(1))
                    logger.debug(f"💾 Extracted used size: {metadata['used_size_gb']}GB")
        
        return metadata
    
    def _extract_error_patterns(self, incident: IncidentCreate) -> List[str]:
        """Extract error patterns from incident description."""
        patterns = []
        description = incident.description
        
        # Common error patterns
        error_indicators = [
            "timeout", "failed", "error", "exception", "out of memory",
            "connection refused", "not found", "permission denied"
        ]
        
        for indicator in error_indicators:
            if indicator in description.lower():
                patterns.append(indicator)
        
        return patterns
    
    async def _recommend_action_types(self, incident: IncidentCreate) -> List[str]:
        """Recommend action types based on incident analysis using ML Service."""
        try:
            # Use ML Service for action recommendation (uses saved models)
            incident_text = f"{incident.title} {incident.description} service:{incident.service}"
            action, confidence = await self.ml_service.recommend_action(incident_text)
            
            # Return the ML-recommended action
            actions = [action]
            
            logger.debug("ML Service action recommendation", 
                        action=action, confidence=confidence)
            
            return actions
            
        except Exception as e:
            logger.warning("ML Service action recommendation failed, using rule-based fallback", 
                          error=str(e))
            # Fallback to rule-based approach
            return self._rule_based_action_recommendations(incident)
    
    def _rule_based_action_recommendations(self, incident: IncidentCreate) -> List[str]:
        """Rule-based action recommendations as fallback."""
        actions = []
        description_lower = incident.description.lower()
        
        if any(term in description_lower for term in ["connection", "timeout", "database"]):
            actions.extend(["restart_database_connection", "restart_service"])
        
        if any(term in description_lower for term in ["memory", "oom"]):
            actions.extend(["restart_service", "scale_resources"])
        
        if "airflow" in description_lower:
            actions.append("restart_airflow_dag")
        
        if "spark" in description_lower:
            actions.append("restart_spark_job")
        
        if any(term in description_lower for term in ["cache", "redis"]):
            actions.append("clear_cache")
        
        return actions
    
    def _assess_risk(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Assess the risk of automated intervention."""
        risk_level = "low"
        risk_factors = []
        
        if incident.severity in ["critical", "high"]:
            risk_level = "medium"
            risk_factors.append("high_severity")
        
        if incident.service in ["payment", "auth", "critical-data"]:
            risk_level = "high"
            risk_factors.append("critical_service")
        
        return {
            "level": risk_level,
            "factors": risk_factors,
            "automation_recommended": risk_level in ["low", "medium"]
        }
    
    async def _find_knowledge_base_matches(
        self,
        incident: IncidentCreate,
        analysis: Dict[str, Any]
    ) -> List[KnowledgeBaseEntry]:
        """Find matching knowledge base entries for the incident."""
        logger.debug("Finding knowledge base matches", incident_title=incident.title)
        
        # Use real knowledge base service
        try:
            from src.services.knowledge_base import KnowledgeBaseService
            knowledge_base_service = KnowledgeBaseService()
            
            # Search for similar incidents using the real service
            matches = await knowledge_base_service.search_similar_incidents(
                error_message=incident.description,
                service=incident.service,
                severity=incident.severity,
                limit=5
            )
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    async def _should_automate_resolution(
        self,
        incident: IncidentCreate,
        analysis: Dict[str, Any],
        kb_matches: List[KnowledgeBaseEntry]
    ) -> Tuple[bool, float, List[ActionCreate]]:
        """Determine if automated resolution should be attempted."""
        
        # Base confidence from AI analysis
        confidence = analysis.get("confidence_score", 0.0)
        
        # Adjust confidence based on knowledge base matches
        if kb_matches:
            avg_success_rate = sum(entry.success_rate for entry in kb_matches) / len(kb_matches)
            confidence = (confidence + avg_success_rate) / 2
        else:
            confidence *= 0.5  # Reduce confidence if no KB matches
        
        # Check risk assessment
        risk_assessment = analysis.get("risk_assessment", {})
        if not risk_assessment.get("automation_recommended", True):
            return False, confidence, []
        
        # Generate recommended actions
        recommended_actions = []
        
        # Import ActionType for mapping
        from ..models.schemas import ActionType
        
        for kb_entry in kb_matches:
            for auto_action in kb_entry.automated_actions:
                # auto_action is a string, try to map to ActionType
                action_type = None
                action_str = auto_action if isinstance(auto_action, str) else str(auto_action)
                
                # Try to find matching ActionType
                for action_enum in ActionType:
                    if action_str.lower() in action_enum.value.lower() or action_enum.value.lower() in action_str.lower():
                        action_type = action_enum
                        break
                
                # Default to a generic action type if no match
                if action_type is None:
                    action_type = ActionType.RESTART_SERVICE  # Default fallback
                
                action = ActionCreate(
                    action_type=action_type,
                    parameters={},
                    timeout_seconds=300,
                    incident_id=None  # Will be set when incident is created
                )
                recommended_actions.append(action)
        
        # If no KB-based actions, generate from AI analysis
        if not recommended_actions:
            action_types = analysis.get("recommended_action_types", [])
            for action_type in action_types:
                action = ActionCreate(
                    action_type=action_type,
                    parameters=self._generate_action_parameters(action_type, incident),
                    timeout_seconds=300,
                    incident_id=None  # Will be set when incident is created
                )
                recommended_actions.append(action)
        
        should_automate = (
            confidence > 0.6 and  # Lower confidence threshold to 60%
            len(recommended_actions) > 0 and  # Must have actions to execute
            incident.severity not in ["critical"]  # Don't automate critical incidents
        )
        
        logger.info(
            "Automation decision made",
            incident_title=incident.title,
            should_automate=should_automate,
            confidence=f"{confidence:.2f}",
            min_threshold="0.60",
            has_actions=len(recommended_actions) > 0,
            action_count=len(recommended_actions),
            severity=incident.severity,
            kb_matches=len(kb_matches)
        )

        return should_automate, confidence, recommended_actions
    
    def _generate_action_parameters(self, action_type: str, incident: IncidentCreate) -> Dict[str, Any]:
        """Generate parameters for an action based on the incident."""
        if action_type == "restart_service":
            return {"service_name": incident.service}
        elif action_type == "restart_airflow_dag":
            return {"dag_id": "auto_detected", "execution_date": None}
        elif action_type == "restart_spark_job":
            return {"application_id": "auto_detected"}
        elif action_type == "restart_database_connection":
            return {"database_name": "default"}
        elif action_type == "clear_cache":
            return {"cache_type": "redis", "pattern": "*"}
        elif action_type == "scale_resources":
            return {"service_name": incident.service, "replicas": 2}
        else:
            return {}
    
    async def _execute_automated_actions(
        self,
        incident: IncidentCreate,
        actions: List[ActionCreate]
    ) -> bool:
        """Execute automated actions for incident resolution."""
        logger.info(
            "🔧 EXECUTING AUTOMATED ACTIONS",
            incident_title=incident.title,
            num_actions=len(actions)
        )
        
        overall_success = True
        successful_actions = 0
        failed_actions = 0
        action_attempts = []  # Track all action attempts for resolution monitoring
        
        for i, action in enumerate(actions, 1):
            action_start_time = time.time()
            action_id = str(uuid.uuid4())
            
            logger.info(
                f"📋 ACTION {i}/{len(actions)} - {action.action_type}",
                action_type=action.action_type,
                parameters=str(action.parameters)[:100],
                incident_title=incident.title,
                action_id=action_id
            )
            
            # Start detailed action logging
            attempt = action_logger.start_action_attempt(
                action_id=action_id,
                action_type=action.action_type,
                parameters=action.parameters,
                incident_id=getattr(incident, 'id', None)
            )
            
            try:
                attempt.log_step("pre_execution", "starting", {
                    "action_sequence_position": i,
                    "total_actions": len(actions),
                    "incident_service": incident.service,
                    "incident_severity": incident.severity
                })
                
                # Execute action using real action execution service
                success = await self._execute_real_action(action, incident)
                execution_time = time.time() - action_start_time
                
                # Record action attempt for resolution monitoring
                action_attempt = {
                    "action_id": action_id,
                    "action_type": action.action_type,
                    "parameters": action.parameters,
                    "success": success,
                    "execution_time": execution_time,
                    "sequence_position": i,
                    "timestamp": datetime.utcnow().isoformat()
                }
                action_attempts.append(action_attempt)
                
                if success:
                    successful_actions += 1
                    
                    attempt.log_step("execution_result", "success", {
                        "execution_time_seconds": execution_time,
                        "sequence_position": i
                    })
                    
                    # Complete action logging with success
                    action_logger.complete_action_attempt(
                        action_id, 
                        success=True, 
                        result={"execution_time": execution_time}
                    )
                    
                    logger.info(
                        f"✅ ACTION {i} COMPLETED SUCCESSFULLY",
                        action_type=action.action_type,
                        execution_time=f"{execution_time:.1f}s",
                        service=incident.service,
                        action_id=action_id
                    )
                else:
                    failed_actions += 1
                    overall_success = False
                    
                    attempt.log_step("execution_result", "failed", {
                        "execution_time_seconds": execution_time,
                        "sequence_position": i,
                        "failure_reason": "action_execution_failed"
                    })
                    
                    # Complete action logging with failure
                    action_logger.complete_action_attempt(
                        action_id, 
                        success=False, 
                        error="Action execution failed",
                        exception_details={"action_type": action.action_type}
                    )
                    
                    logger.error(
                        f"❌ ACTION {i} FAILED",
                        action_type=action.action_type,
                        execution_time=f"{execution_time:.1f}s",
                        service=incident.service,
                        action_id=action_id
                    )
                    
            except Exception as e:
                failed_actions += 1
                overall_success = False
                execution_time = time.time() - action_start_time
                
                # Record failed action attempt
                action_attempt = {
                    "action_id": action_id,
                    "action_type": action.action_type,
                    "parameters": action.parameters,
                    "success": False,
                    "execution_time": execution_time,
                    "sequence_position": i,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                action_attempts.append(action_attempt)
                
                attempt.log_step("execution_result", "exception", {
                    "execution_time_seconds": execution_time,
                    "sequence_position": i,
                    "exception_type": type(e).__name__,
                    "error_message": str(e)
                })
                
                # Complete action logging with exception
                exception_details = {
                    "exception_type": type(e).__name__,
                    "exception_module": type(e).__module__,
                    "sequence_position": i
                }
                action_logger.complete_action_attempt(
                    action_id, 
                    success=False, 
                    error=str(e),
                    exception_details=exception_details
                )
                
                logger.error(
                    f"💥 ACTION {i} EXCEPTION",
                    action_type=action.action_type,
                    error=str(e),
                    execution_time=f"{execution_time:.1f}s",
                    action_id=action_id
                )
        
        # Summary logging with detailed action tracking
        if overall_success:
            logger.info(
                "🎉 ALL ACTIONS COMPLETED SUCCESSFULLY",
                incident_title=incident.title,
                successful_actions=successful_actions,
                total_actions=len(actions),
                action_attempts=len(action_attempts)
            )
        else:
            logger.error(
                "⚠️  SOME ACTIONS FAILED",
                incident_title=incident.title,
                successful_actions=successful_actions,
                failed_actions=failed_actions,
                total_actions=len(actions),
                action_attempts=len(action_attempts)
            )
        
        # Store action attempts summary for resolution monitoring
        if action_attempts:
            try:
                # Save action summary for this resolution attempt
                resolution_summary = {
                    "incident_id": getattr(incident, 'id', None),
                    "incident_title": incident.title,
                    "incident_service": incident.service,
                    "incident_severity": incident.severity,
                    "timestamp": datetime.utcnow().isoformat(),
                    "overall_success": overall_success,
                    "total_actions": len(actions),
                    "successful_actions": successful_actions,
                    "failed_actions": failed_actions,
                    "action_attempts": action_attempts
                }
                
                # Store in resolution logs for monitoring
                resolution_logs_dir = Path("data/resolution_logs")
                resolution_logs_dir.mkdir(parents=True, exist_ok=True)
                
                # Save to daily file for easy querying
                date_str = datetime.utcnow().strftime("%Y-%m-%d")
                resolution_file = resolution_logs_dir / f"resolutions_{date_str}.jsonl"
                
                with open(resolution_file, 'a') as f:
                    f.write(json.dumps(resolution_summary) + '\n')
                
                logger.info(
                    "📊 Resolution summary saved for monitoring",
                    incident_title=incident.title,
                    overall_success=overall_success,
                    actions_logged=len(action_attempts)
                )
                
            except Exception as e:
                logger.error(
                    "Failed to save resolution summary",
                    incident_title=incident.title,
                    error=str(e)
                )
        
        return overall_success
    
    async def _execute_real_action(self, action: ActionCreate, incident: IncidentCreate) -> bool:
        """Execute real action using the action execution service."""
        try:
            from src.services.action_execution import ActionExecutionService
            action_execution_service = ActionExecutionService()
            
            # Convert action to dictionary format expected by the service
            action_dict = {
                "type": action.action_type,
                "parameters": action.parameters,
                "description": f"Action {action.action_type} with parameters: {action.parameters}"
            }
            
            # Convert incident to dictionary format
            incident_dict = {
                "id": getattr(incident, 'id', 'unknown'),
                "title": incident.title,
                "description": incident.description,
                "service": incident.service,
                "severity": incident.severity,
                "logs": getattr(incident, 'logs', 'No logs available')
            }
            
            # Execute the action using the real service
            success = await action_execution_service.execute_action(action_dict, incident_dict)
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing real action: {str(e)}")
            return False
    
    async def _create_manual_intervention_alert(
        self,
        incident: IncidentCreate,
        analysis: Dict[str, Any],
        reason: str
    ):
        """Create alert for manual intervention required."""
        logger.info(
            "Creating manual intervention alert",
            incident_title=incident.title,
            reason=reason
        )
        
        # Send manual escalation alert using notification service
        try:
            from src.services.notifications import NotificationService
            notification_service = NotificationService()
            
            # Convert IncidentCreate to dict format expected by notification service
            incident_dict = {
                'id': getattr(incident, 'id', f"inc-{incident.title[:10]}"),
                'title': incident.title,
                'service': incident.service,
                'severity': incident.severity,
                'timestamp': getattr(incident, 'timestamp', None)
            }
            
            await notification_service.send_manual_escalation_alert(
                incident=incident_dict,
                analysis=analysis,
                reason=reason
            )
            
            logger.info("Manual escalation alert sent successfully")
        except Exception as e:
            logger.error("Failed to send manual escalation alert", error=str(e))
            # Fallback to logging
            alert_message = f"""
            🚨 Manual Intervention Required
            
            Incident: {incident.title}
            Service: {incident.service}
            Severity: {incident.severity}
            
            Reason: {reason}
            
            AI Analysis:
            - Root Cause: {analysis.get('root_cause_category', 'Unknown')}
            - Confidence: {analysis.get('confidence_score', 0):.2f}
            - Affected Components: {', '.join(analysis.get('affected_components', []))}
            
            Please review and take manual action if necessary.
            """
            
            logger.warning("Manual intervention alert (fallback to logging)", message=alert_message[:200] + "...")
    
    async def analyze_log_patterns(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Analyze log patterns using AI."""
        logger.info("Analyzing log patterns with AI", num_logs=len(logs))
        
        # TODO: Implement actual AI analysis of log patterns
        # This could use OpenAI to identify anomalies, patterns, etc.
        
        return {
            "anomalies_detected": 2,
            "pattern_analysis": {
                "error_rate_trend": "increasing",
                "common_error_types": ["connection_timeout", "memory_error"],
                "affected_services": ["web-api", "etl-pipeline"]
            },
            "recommendations": [
                "Check database connectivity",
                "Monitor memory usage",
                "Review ETL pipeline configuration"
            ],
            "confidence": 0.78,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    # =============================================
    # ML TRAINING AND MODEL MANAGEMENT
    # =============================================
    
    def add_training_data(self, incident: IncidentCreate, outcome: str, resolution_time: float, success: bool):
        """Add training data from resolved incidents."""
        try:
            # Extract features from incident
            features = self._extract_ml_features(incident)
            
            # Store training data
            training_sample = {
                "incident": incident.dict(),
                "features": features,
                "outcome": outcome,
                "resolution_time": resolution_time,
                "success": success,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.training_data["incidents"].append(training_sample)
            logger.info("Added training sample", incident_title=incident.title, outcome=outcome, success=success)
            
        except Exception as e:
            logger.error("Error adding training data", error=str(e))
    
    async def add_training_data_async(self, incident: IncidentCreate, outcome: str, resolution_time: float, 
                                     success: bool, confidence_score: Optional[float] = None, 
                                     actions_executed: Optional[List[str]] = None):
        """Add training data from resolved incidents and save to database."""
        try:
            # Extract features from incident
            features = self._extract_ml_features(incident)
            
            # Create database record
            training_record = TrainingData(
                id=str(uuid.uuid4()),
                incident_id=getattr(incident, 'id', None),
                incident_title=incident.title,
                incident_service=incident.service,
                incident_severity=incident.severity,
                incident_description=incident.description,
                features=features,
                outcome=outcome,
                resolution_time=resolution_time,
                success=success,
                confidence_score=confidence_score,
                actions_executed=actions_executed or [],
                timestamp=datetime.utcnow()
            )
            
            # Save to database
            async for db in get_db_session():
                db.add(training_record)
                await db.commit()
                break
            
            # Also add to in-memory storage for immediate use
            training_sample = {
                "incident": incident.dict(),
                "features": features,
                "outcome": outcome,
                "resolution_time": resolution_time,
                "success": success,
                "confidence_score": confidence_score,
                "actions_executed": actions_executed,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.training_data["incidents"].append(training_sample)
            logger.info("Added training sample to database and memory", 
                       incident_title=incident.title, outcome=outcome, success=success)
            
            # Check if we should trigger incremental retraining
            await self._check_incremental_training()
            
        except Exception as e:
            logger.error("Error adding training data", error=str(e))
    
    async def _check_incremental_training(self):
        """Check if incremental training should be triggered based on new data."""
        try:
            total_samples = len(self.training_data["incidents"])
            last_training_samples = self.model_metadata.get("training_samples", 0)
            new_samples = total_samples - last_training_samples
            
            # Trigger incremental training if we have 25+ new samples or no models exist
            should_retrain = False
            reason = "no retraining needed"
            
            if new_samples >= 25 and total_samples >= 50:
                should_retrain = True
                reason = f"incremental training - {new_samples} new samples"
            elif not self.incident_classifier and total_samples >= 10:
                should_retrain = True
                reason = "no models exist and sufficient data available"
            
            if should_retrain:
                logger.info("🔄 Triggering incremental model training", 
                           total_samples=total_samples,
                           new_samples=new_samples,
                           reason=reason)
                
                # Train models asynchronously to not block incident processing
                results = self.train_models(min_samples=10)
                
                if results["success"]:
                    # Save the updated models
                    model_path = "models/ai_decision_engine.pkl"
                    save_success = self.save_model(model_path)
                    
                    logger.info("✅ Incremental training completed and saved", 
                               accuracy=results.get("evaluation", {}).get("classification_accuracy", 0),
                               model_saved=save_success)
                else:
                    logger.warning("⚠️ Incremental training failed", 
                                 error=results.get("error", "Unknown"))
            
        except Exception as e:
            logger.error("Error in incremental training check", error=str(e))
    
    async def _load_training_data_from_db(self):
        """Load training data from database into memory."""
        try:
            async for db in get_db_session():
                # Query all training data
                result = await db.execute(text("SELECT * FROM training_data ORDER BY timestamp DESC LIMIT 1000"))
                rows = result.fetchall()
                
                # Convert to training format
                self.training_data["incidents"] = []
                for row in rows:
                    training_sample = {
                        "incident": {
                            "title": row.incident_title,
                            "service": row.incident_service,
                            "severity": row.incident_severity,
                            "description": row.incident_description,
                            "tags": []  # Default empty tags
                        },
                        "features": row.features,
                        "outcome": row.outcome,
                        "resolution_time": row.resolution_time,
                        "success": row.success,
                        "confidence_score": row.confidence_score,
                        "actions_executed": row.actions_executed or [],
                        "timestamp": row.timestamp.isoformat() if hasattr(row.timestamp, 'isoformat') else str(row.timestamp)
                    }
                    self.training_data["incidents"].append(training_sample)
                
                logger.info("Loaded training data from database", 
                           samples=len(self.training_data["incidents"]))
                break
                
        except Exception as e:
            logger.error("Error loading training data from database", error=str(e))
            # Initialize empty if database load fails
            self.training_data["incidents"] = []
    
    async def retrain_models(self, min_samples: int = 50) -> Dict[str, Any]:
        """Retrain models with latest data from database."""
        try:
            # Reload training data from database
            await self._load_training_data_from_db()
            
            if len(self.training_data["incidents"]) < min_samples:
                return {
                    "success": False,
                    "error": f"Insufficient training data. Need {min_samples}, got {len(self.training_data['incidents'])}"
                }
            
            # Train models
            results = self.train_models(min_samples=min_samples)
            
            if results["success"]:
                # Save the retrained models
                model_path = "models/ai_decision_engine.pkl"
                save_success = self.save_model(model_path)
                
                logger.info("🔄 Models retrained and saved successfully", 
                           samples=results["training_samples"],
                           accuracy=results.get("evaluation", {}).get("classification_accuracy", 0),
                           model_saved=save_success,
                           model_path=model_path)
                
                # Add save status to results
                results["model_saved"] = save_success
                results["model_path"] = model_path
            
            return results
            
        except Exception as e:
            logger.error("Error retraining models", error=str(e))
            return {"success": False, "error": str(e)}
    
    def _extract_ml_features(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Extract ML features from incident for training."""
        features = {
            # Text features
            "title_length": len(incident.title),
            "description_length": len(incident.description),
            "num_tags": len(incident.tags),
            
            # Categorical features
            "service": incident.service,
            "severity": incident.severity,
            
            # Text content features (will be vectorized)
            "text_content": f"{incident.title} {incident.description}",
            
            # Pattern-based features
            "has_timeout": any(word in incident.description.lower() for word in ["timeout", "timed out"]),
            "has_memory_issue": any(word in incident.description.lower() for word in ["memory", "oom", "heap"]),
            "has_connection_issue": any(word in incident.description.lower() for word in ["connection", "connect", "refused"]),
            "has_database_issue": any(word in incident.description.lower() for word in ["database", "db", "sql"]),
            "has_spark_issue": any(word in incident.description.lower() for word in ["spark", "executor", "yarn"]),
            "has_airflow_issue": any(word in incident.description.lower() for word in ["airflow", "dag", "task"]),
            
            # Error pattern features
            "error_patterns": self._extract_error_patterns(incident),
            "num_error_patterns": len(self._extract_error_patterns(incident)),
            
            # Time-based features (can be added later)
            "hour_of_day": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday(),
        }
        
        return features
    
    def train_models(self, min_samples: int = 50) -> Dict[str, Any]:
        """Train ML models on collected incident data.
        
        Trains three models from accumulated training data:
        1. Incident Classifier: RandomForestClassifier for categorizing incidents
        2. Confidence Model: CalibratedClassifierCV for predicting resolution success
        3. Pattern Clustering: PCA + KMeans for anomaly detection
        
        Training Process:
        - Extracts features from incident text and metadata
        - Vectorizes text using TfidfVectorizer 
        - Trains classification models with cross-validation
        - Evaluates model performance on hold-out data
        - Updates model metadata with training results
        
        Args:
            min_samples (int): Minimum training samples required (default: 50)
            
        Returns:
            Dict[str, Any]: Training results containing:
                - success: bool indicating if training completed
                - training_samples: number of samples used
                - evaluation: model accuracy and performance metrics
                - metadata: training timestamp, version, etc.
                - error: error message if training failed
                
        Raises:
            ValueError: If insufficient training samples available
            
        Example:
            # Add training data from resolved incidents
            ai_engine.add_training_data(incident, outcome="resolved", 
                                      resolution_time=120.5, success=True)
            
            # Train models when enough data collected
            results = ai_engine.train_models(min_samples=100)
            if results["success"]:
                ai_engine.save_model("models/trained_model.pkl")
        """
        try:
            if len(self.training_data["incidents"]) < min_samples:
                return {
                    "success": False,
                    "error": f"Insufficient training data. Need {min_samples}, got {len(self.training_data['incidents'])}"
                }
            
            logger.info("Starting ML model training", num_samples=len(self.training_data["incidents"]))
            
            # Prepare training data
            X, y_category, y_success, y_confidence = self._prepare_training_data()
            
            # Train incident classification model
            self._train_incident_classifier(X, y_category)
            
            # Train success prediction model  
            self._train_confidence_model(X, y_success, y_confidence)
            
            # Train pattern clustering model
            self._train_pattern_clustering(X)
            
            # Update metadata
            self.model_metadata.update({
                "trained_at": datetime.utcnow().isoformat(),
                "training_samples": len(self.training_data["incidents"]),
                "version": "1.0.0"
            })
            
            # Evaluate models
            evaluation_results = self._evaluate_models(X, y_category, y_success)
            
            logger.info("ML model training completed", 
                       accuracy=evaluation_results.get("classification_accuracy", 0),
                       samples=len(self.training_data["incidents"]))
            
            return {
                "success": True,
                "training_samples": len(self.training_data["incidents"]),
                "evaluation": evaluation_results,
                "metadata": self.model_metadata
            }
            
        except Exception as e:
            logger.error("Error training ML models", error=str(e))
            return {"success": False, "error": str(e)}
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        features_list = []
        categories = []
        success_labels = []
        confidence_scores = []
        
        for sample in self.training_data["incidents"]:
            incident_data = sample["incident"]
            features = sample["features"]
            
            # Extract text content for vectorization
            text_content = features.get("text_content", "")
            
            # Create feature vector
            feature_vector = {
                "title_length": features.get("title_length", 0),
                "description_length": features.get("description_length", 0),
                "num_tags": features.get("num_tags", 0),
                "has_timeout": features.get("has_timeout", False),
                "has_memory_issue": features.get("has_memory_issue", False),
                "has_connection_issue": features.get("has_connection_issue", False),
                "has_database_issue": features.get("has_database_issue", False),
                "has_spark_issue": features.get("has_spark_issue", False),
                "has_airflow_issue": features.get("has_airflow_issue", False),
                "num_error_patterns": features.get("num_error_patterns", 0),
                "hour_of_day": features.get("hour_of_day", 0),
                "day_of_week": features.get("day_of_week", 0),
                "text": text_content
            }
            
            features_list.append(feature_vector)
            
            # Determine category based on incident
            category = self._determine_root_cause_category(IncidentCreate(**incident_data))
            categories.append(category)
            
            # Success and confidence labels
            success_labels.append(sample["success"])
            
            # Calculate confidence based on resolution time and success
            resolution_time = sample.get("resolution_time", 300)
            if sample["success"]:
                confidence = max(0.1, min(0.95, 1.0 - (resolution_time / 3600)))  # Higher confidence for faster resolution
            else:
                confidence = 0.1
            confidence_scores.append(confidence)
        
        # Create feature matrix
        X = self._vectorize_features(features_list)
        
        return X, np.array(categories), np.array(success_labels), np.array(confidence_scores)
    
    def _vectorize_features(self, features_list: List[Dict[str, Any]]) -> np.ndarray:
        """Convert feature dictionaries to numerical matrix."""
        # Initialize vectorizer if not exists
        if self.feature_vectorizer is None:
            self.feature_vectorizer = TfidfVectorizer(
                max_features=min(1000, len(features_list) * 10),  # Adaptive max features
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,  # Don't ignore any terms
                max_df=0.95  # Ignore very common terms
            )
        
        # Extract text features
        text_features = [f.get("text", "") for f in features_list]
        
        # Handle case where all text is empty or too short
        if not any(text.strip() for text in text_features):
            # Create dummy text features if no text content
            text_features = [f"sample_{i}" for i in range(len(text_features))]
        
        try:
            text_vectors_sparse = self.feature_vectorizer.fit_transform(text_features)
            # Convert sparse matrix to dense array
            text_vectors = np.array(text_vectors_sparse.todense())
        except ValueError as e:
            # Fallback: create zero matrix if TF-IDF fails
            logger.warning(f"TF-IDF vectorization failed, using zero matrix: {e}")
            text_vectors = np.zeros((len(features_list), 100))  # Fixed size fallback
        
        # Extract numerical features
        numerical_features = []
        for f in features_list:
            num_features = [
                f.get("title_length", 0),
                f.get("description_length", 0),
                f.get("num_tags", 0),
                int(f.get("has_timeout", False)),
                int(f.get("has_memory_issue", False)),
                int(f.get("has_connection_issue", False)),
                int(f.get("has_database_issue", False)),
                int(f.get("has_spark_issue", False)),
                int(f.get("has_airflow_issue", False)),
                f.get("num_error_patterns", 0),
                f.get("hour_of_day", 0),
                f.get("day_of_week", 0),
            ]
            numerical_features.append(num_features)
        
        numerical_array = np.array(numerical_features)
        
        # Combine text and numerical features
        X = np.hstack([text_vectors, numerical_array])
        
        return X
    
    def _train_incident_classifier(self, X: np.ndarray, y_category: np.ndarray):
        """Train incident classification model."""
        # Encode labels
        if "category" not in self.label_encoders:
            self.label_encoders["category"] = LabelEncoder()
        
        y_encoded = self.label_encoders["category"].fit_transform(y_category)
        
        # Create and train classifier
        self.incident_classifier = Pipeline([
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        self.incident_classifier.fit(X, y_encoded)
        
        # Store feature names and classes
        self.model_metadata["feature_names"] = [f"feature_{i}" for i in range(X.shape[1])]
        self.model_metadata["label_classes"] = self.label_encoders["category"].classes_.tolist()
    
    def _train_confidence_model(self, X: np.ndarray, y_success: np.ndarray, y_confidence: np.ndarray):
        """Train confidence prediction model."""
        # Use calibrated classifier for better confidence estimates
        base_classifier = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            class_weight='balanced'
        )
        
        self.confidence_model = CalibratedClassifierCV(
            base_classifier, 
            method='isotonic',
            cv=3
        )
        
        self.confidence_model.fit(X, y_success)
    
    def _train_pattern_clustering(self, X: np.ndarray):
        """Train pattern clustering model for anomaly detection."""
        # Use PCA for dimensionality reduction
        # Ensure we have enough samples and features for PCA
        n_components = min(50, X.shape[1], X.shape[0] - 1)
        
        if n_components > 1 and X.shape[0] > n_components:
            pca = PCA(n_components=n_components)
            try:
                X_reduced = pca.fit_transform(X)
                print(f"PCA reduced features from {X.shape[1]} to {n_components} dimensions")
            except Exception as e:
                print(f"PCA failed, using original features: {e}")
                pca = None
                X_reduced = X
        else:
            print(f"Too few samples ({X.shape[0]}) or features ({X.shape[1]}) for PCA, using original features")
            pca = None
            X_reduced = X
        
        # Train clustering model
        n_clusters = min(10, len(X) // 5, 3)  # Adaptive number of clusters with minimum
        if n_clusters < 2:
            n_clusters = 2  # Minimum clusters
            
        self.pattern_clustering = {
            'pca': pca,
            'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
            'n_clusters': n_clusters
        }
        
        self.pattern_clustering['kmeans'].fit(X_reduced)
    
    def _evaluate_models(self, X: np.ndarray, y_category: np.ndarray, y_success: np.ndarray) -> Dict[str, Any]:
        """Evaluate trained models."""
        results = {}
        
        try:
            # Split data for evaluation
            X_train, X_test, y_cat_train, y_cat_test, y_succ_train, y_succ_test = train_test_split(
                X, 
                self.label_encoders["category"].transform(y_category),
                y_success,
                test_size=0.2,
                random_state=42
            )
            
            # Evaluate classification model
            if self.incident_classifier:
                y_pred = self.incident_classifier.predict(X_test)
                results["classification_accuracy"] = accuracy_score(y_cat_test, y_pred)
                results["classification_report"] = classification_report(
                    y_cat_test, y_pred, 
                    target_names=self.label_encoders["category"].classes_,
                    output_dict=True
                )
            
            # Evaluate confidence model
            if self.confidence_model:
                y_pred_proba = self.confidence_model.predict_proba(X_test)
                results["confidence_accuracy"] = accuracy_score(y_succ_test, self.confidence_model.predict(X_test))
                
            self.model_metadata["accuracy"] = results.get("classification_accuracy", 0.0)
            
        except Exception as e:
            logger.error("Error evaluating models", error=str(e))
            results["evaluation_error"] = str(e)
        
        return results
    
    async def predict_incident_category(self, incident: IncidentCreate) -> Tuple[str, float]:
        """Predict incident category using trained ML models.
        
        Uses the ML Service's incident classifier to categorize incidents.
        This ensures we use the same models that are saved to disk and
        provides a consistent prediction interface.
        
        Args:
            incident (IncidentCreate): Incident to classify
            
        Returns:
            Tuple[str, float]: (category, confidence_score)
            
        Fallback:
            If ML Service models not loaded, uses rule-based classification
        """
        # Use ML Service for predictions to ensure consistency with saved models
        try:
            # Create text representation for ML Service
            incident_text = f"{incident.title} {incident.description} service:{incident.service}"
            
            # Use ML Service for prediction (this uses the joblib models from disk)
            severity, confidence = await self.ml_service.predict_incident_severity(incident_text)
            
            # Map severity to category (the ML Service predicts severity, we need category)
            category_mapping = {
                "critical": "memory_issues",
                "high": "workflow_failure", 
                "medium": "database_connectivity",
                "low": "data_availability"
            }
            
            category = category_mapping.get(severity, self._determine_root_cause_category(incident))
            
            logger.debug("ML Service prediction used", 
                        category=category, confidence=confidence, severity=severity)
            
            return category, confidence
            
        except Exception as e:
            logger.warning("ML Service prediction failed, using rule-based fallback", 
                          error=str(e))
            # Fallback to rule-based approach
            return self._determine_root_cause_category(incident), 0.5
    
    def predict_resolution_confidence(self, incident: IncidentCreate) -> float:
        """Predict confidence for automated resolution success.
        
        Uses the loaded confidence_model (CalibratedClassifierCV) to predict the
        probability that automated actions will successfully resolve the incident.
        
        Confidence levels:
        - >0.8: High confidence - automation very likely to succeed
        - 0.6-0.8: Medium confidence - automation may succeed  
        - <0.6: Low confidence - manual intervention recommended
        
        The confidence score is used to determine if automated actions should be
        executed (typically requires >0.6 confidence threshold).
        
        Args:
            incident (IncidentCreate): Incident to assess
            
        Returns:
            float: Confidence score between 0.0 and 1.0
            
        Fallback:
            Returns 0.5 if models not loaded, 0.3 if prediction fails
        """
        if not self.confidence_model or not self.feature_vectorizer:
            return 0.5  # Default confidence
        
        try:
            # Extract features
            features = self._extract_ml_features(incident)
            
            # Use same vectorization approach as classification
            try:
                text_content = features.get("text_content", "")
                if not text_content.strip():
                    text_content = f"{incident.service} {incident.severity} issue"
                
                # Transform using existing vectorizer
                text_vector_sparse = self.feature_vectorizer.transform([text_content])
                text_vector = np.array(text_vector_sparse.todense())
                
                # Combine with numerical features
                numerical_features = [
                    features.get("title_length", 0),
                    features.get("description_length", 0),
                    features.get("num_tags", 0),
                    int(features.get("has_timeout", False)),
                    int(features.get("has_memory_issue", False)),
                    int(features.get("has_connection_issue", False)),
                    int(features.get("has_database_issue", False)),
                    int(features.get("has_spark_issue", False)),
                    int(features.get("has_airflow_issue", False)),
                    features.get("num_error_patterns", 0),
                    features.get("hour_of_day", 0),
                    features.get("day_of_week", 0),
                ]
                
                X = np.hstack([text_vector, np.array(numerical_features).reshape(1, -1)])
                
            except Exception as vec_error:
                logger.warning(f"Vectorization failed in confidence prediction: {vec_error}")
                return 0.3
            
            # Predict confidence
            confidence_proba = self.confidence_model.predict_proba(X)[0]
            confidence = float(confidence_proba[1]) if len(confidence_proba) > 1 else 0.5
            
            return confidence
            
        except Exception as e:
            logger.error("Error predicting resolution confidence", error=str(e))
            return 0.3
    
    def detect_anomalies(self, incident: IncidentCreate) -> Dict[str, Any]:
        """Detect if incident is anomalous compared to training data."""
        if not self.pattern_clustering:
            return {"is_anomaly": False, "anomaly_score": 0.0}
        
        try:
            # Extract features
            features = self._extract_ml_features(incident)
            X = self._vectorize_features([features])
            
            # Reduce dimensions if PCA is available
            if self.pattern_clustering['pca'] is not None:
                X_reduced = self.pattern_clustering['pca'].transform(X)
            else:
                X_reduced = X
            
            # Find nearest cluster
            cluster_distances = self.pattern_clustering['kmeans'].transform(X_reduced)[0]
            min_distance = float(np.min(cluster_distances))
            
            # Calculate anomaly score (higher = more anomalous)
            # Use 95th percentile as threshold
            anomaly_threshold = np.percentile(cluster_distances, 95)
            is_anomaly = min_distance > anomaly_threshold
            anomaly_score = min_distance / (anomaly_threshold + 1e-6)
            
            return {
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(anomaly_score),
                "min_cluster_distance": min_distance,
                "threshold": float(anomaly_threshold)
            }
            
        except Exception as e:
            logger.error("Error detecting anomalies", error=str(e))
            return {"is_anomaly": False, "anomaly_score": 0.0, "error": str(e)}
    
    def save_model(self, file_path: str) -> bool:
        """Save trained models to file.
        
        Serializes all ML components to a PKL file including:
        - incident_classifier: RandomForestClassifier for incident categorization
        - confidence_model: CalibratedClassifierCV for resolution success prediction
        - pattern_clustering: dict with PCA + KMeans for anomaly detection
        - feature_vectorizer: TfidfVectorizer for text feature extraction
        - label_encoders: dict of LabelEncoders for category encoding/decoding
        - model_metadata: training info, accuracy, timestamps, etc.
        - training_data: optional historical training data
        
        Args:
            file_path (str): Path where to save the PKL file (e.g., "models/ai_model.pkl")
            
        Returns:
            bool: True if save successful, False otherwise
            
        Example:
            ai_engine.save_model("models/trained/my_model.pkl")
        """
        try:
            model_data = {
                "incident_classifier": self.incident_classifier,
                "confidence_model": self.confidence_model,
                "pattern_clustering": self.pattern_clustering,
                "feature_vectorizer": self.feature_vectorizer,
                "label_encoders": self.label_encoders,
                "model_metadata": self.model_metadata,
                "training_data": self.training_data
            }
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save with pickle
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully", file_path=file_path)
            return True
            
        except Exception as e:
            logger.error("Error saving model", error=str(e), file_path=file_path)
            return False
    
    def load_model(self, file_path: str) -> bool:
        """Load trained models from file.
        
        Deserializes ML components from a PKL file and loads them into the AI engine:
        - incident_classifier: for predicting incident categories
        - confidence_model: for predicting resolution success probability
        - pattern_clustering: for detecting anomalous incidents
        - feature_vectorizer: for converting text to numerical features
        - label_encoders: for encoding/decoding categorical labels
        - model_metadata: training information and model versioning
        
        Args:
            file_path (str): Path to the PKL file to load (e.g., "models/latest.pkl")
            
        Returns:
            bool: True if load successful, False if file not found or load failed
            
        Example:
            ai_engine.load_model("models/latest.pkl")
            
        Note:
            If loading fails, AI engine falls back to rule-based analysis
        """
        try:
            if not Path(file_path).exists():
                logger.warning("Model file not found", file_path=file_path)
                return False
            
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load model components
            self.incident_classifier = model_data.get("incident_classifier")
            self.confidence_model = model_data.get("confidence_model")
            self.pattern_clustering = model_data.get("pattern_clustering")
            self.feature_vectorizer = model_data.get("feature_vectorizer")
            self.label_encoders = model_data.get("label_encoders", {})
            self.model_metadata = model_data.get("model_metadata", {})
            
            # Optionally load training data
            if "training_data" in model_data:
                self.training_data = model_data["training_data"]
            
            logger.info("Model loaded successfully", 
                       file_path=file_path,
                       version=self.model_metadata.get("version", "unknown"),
                       samples=self.model_metadata.get("training_samples", 0))
            return True
            
        except Exception as e:
            logger.error("Error loading model", error=str(e), file_path=file_path)
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "metadata": self.model_metadata,
            "has_classifier": self.incident_classifier is not None,
            "has_confidence_model": self.confidence_model is not None,
            "has_clustering": self.pattern_clustering is not None,
            "has_vectorizer": self.feature_vectorizer is not None,
            "training_samples": len(self.training_data.get("incidents", [])),
            "label_classes": self.model_metadata.get("label_classes", []),
            "feature_count": len(self.model_metadata.get("feature_names", []))
        }
    
    def export_model_summary(self, file_path: str) -> bool:
        """Export model summary and metadata to JSON."""
        try:
            summary = {
                "model_info": self.get_model_info(),
                "training_history": self.training_data,
                "feature_importance": self._get_feature_importance(),
                "export_timestamp": datetime.utcnow().isoformat()
            }
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Model summary exported", file_path=file_path)
            return True
            
        except Exception as e:
            logger.error("Error exporting model summary", error=str(e))
            return False
    
    def _get_feature_importance(self) -> List[Dict[str, Any]]:
        """Get feature importance from trained models."""
        if not self.incident_classifier:
            return []
        
        try:
            # Get feature importance from random forest
            classifier = self.incident_classifier.named_steps['classifier']
            importances = classifier.feature_importances_
            feature_names = self.model_metadata.get("feature_names", [])
            
            importance_list = []
            for i, importance in enumerate(importances):
                feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                importance_list.append({
                    "feature": feature_name,
                    "importance": float(importance)
                })
            
            # Sort by importance
            importance_list.sort(key=lambda x: x["importance"], reverse=True)
            
            return importance_list[:20]  # Top 20 features
            
        except Exception as e:
            logger.error("Error getting feature importance", error=str(e))
            return []
    
    async def record_resolution_outcome(self, incident_id: str, actions_taken: List[Dict[str, Any]], 
                                      success: bool, resolution_time: int, 
                                      confidence_score: float) -> Dict[str, Any]:
        """Record the outcome of an incident resolution for continuous learning."""
        
        learning_record = {
            "incident_id": incident_id,
            "timestamp": datetime.utcnow().isoformat(),
            "actions_taken": actions_taken,
            "success": success,
            "resolution_time": resolution_time,
            "confidence_score": confidence_score,
            "learning_feedback": await self._generate_learning_feedback(
                incident_id, actions_taken, success, resolution_time, confidence_score
            )
        }
        
        # Add to training data for continuous learning
        await self._update_training_data(learning_record)
        
        # Update model confidence thresholds if needed
        await self._adjust_confidence_thresholds(learning_record)
        
        logger.info(f"📚 Recorded resolution outcome", 
                   incident_id=incident_id,
                   success=success,
                   resolution_time=resolution_time,
                   confidence=confidence_score)
        
        return learning_record
    
    async def _generate_learning_feedback(self, incident_id: str, actions_taken: List[Dict[str, Any]], 
                                        success: bool, resolution_time: int, 
                                        confidence_score: float) -> Dict[str, Any]:
        """Generate comprehensive learning feedback from resolution outcome."""
        
        feedback = {
            "confidence_score": confidence_score,
            "resolution_effectiveness": 1.0 if success else 0.0,
            "knowledge_base_updated": False,
            "new_patterns_learned": [],
            "performance_metrics": {
                "resolution_faster_than_average": False,
                "action_sequence_optimal": True,
                "resource_usage_efficient": True
            }
        }
        
        # Calculate pattern match strength based on previous similar incidents
        try:
            similar_incidents = await self._find_similar_historical_incidents(incident_id)
            feedback["similar_incidents_count"] = len(similar_incidents)
            
            if similar_incidents:
                # Calculate average resolution time for similar incidents
                avg_resolution_time = sum(inc.get("resolution_time", 300) for inc in similar_incidents) / len(similar_incidents)
                feedback["performance_metrics"]["resolution_faster_than_average"] = resolution_time < avg_resolution_time
                
                # Calculate success rate for similar incidents
                successful_similar = [inc for inc in similar_incidents if inc.get("success", False)]
                success_rate = len(successful_similar) / len(similar_incidents)
                feedback["pattern_match_strength"] = success_rate
            else:
                feedback["pattern_match_strength"] = 0.5  # Default for new patterns
        
        except Exception as e:
            logger.warning(f"⚠️ Error calculating similar incidents: {e}")
            feedback["pattern_match_strength"] = 0.5
            feedback["similar_incidents_count"] = 0
        
        # Analyze action sequence effectiveness
        if actions_taken:
            successful_actions = [action for action in actions_taken if action.get("result") == "success"]
            action_success_rate = len(successful_actions) / len(actions_taken)
            feedback["action_sequence_effectiveness"] = action_success_rate
            
            # Identify new patterns from failed actions
            if not success:
                failed_actions = [action for action in actions_taken if action.get("result") == "failed"]
                for action in failed_actions:
                    pattern_name = f"{action.get('action', 'unknown')}_failure_pattern"
                    feedback["new_patterns_learned"].append(pattern_name)
        
        # Update learning based on success/failure
        if success:
            feedback["resolution_effectiveness"] = min(1.0, 1.0 - (resolution_time / 3600))  # Higher effectiveness for faster resolution
            if resolution_time < 120:  # Less than 2 minutes
                feedback["new_patterns_learned"].append("fast_resolution_pattern")
        else:
            # Analyze failure for learning
            feedback["failure_analysis"] = await self._analyze_resolution_failure(actions_taken, incident_id)
        
        return feedback
    
    async def _analyze_resolution_failure(self, actions_taken: List[Dict[str, Any]], 
                                        incident_id: str) -> Dict[str, Any]:
        """Analyze why resolution failed to improve future decisions."""
        
        failure_analysis = {
            "root_cause": "unknown",
            "recommended_escalation": "manual_intervention_required",
            "improved_action_sequence": [],
            "confidence_threshold_adjustment": -0.1
        }
        
        if not actions_taken:
            failure_analysis["root_cause"] = "no_actions_attempted"
            failure_analysis["improved_action_sequence"] = ["diagnostic_action", "basic_restart"]
            return failure_analysis
        
        # Analyze failed actions
        failed_actions = [action for action in actions_taken if action.get("result") == "failed"]
        successful_actions = [action for action in actions_taken if action.get("result") == "success"]
        
        if len(failed_actions) > len(successful_actions):
            failure_analysis["root_cause"] = "insufficient_actions_for_severity"
            
            # Suggest escalated action sequence
            last_action_type = actions_taken[-1].get("action", "")
            if "restart" not in last_action_type:
                failure_analysis["improved_action_sequence"].append("restart_service")
            if "scale" not in last_action_type:
                failure_analysis["improved_action_sequence"].append("scale_resources")
            
            failure_analysis["improved_action_sequence"].append("manual_intervention")
        
        elif len(failed_actions) == 0:
            failure_analysis["root_cause"] = "external_dependency_issue"
            failure_analysis["recommended_escalation"] = "check_external_services"
        
        else:
            failure_analysis["root_cause"] = "complex_issue_requiring_investigation"
            failure_analysis["recommended_escalation"] = "engineering_team_review"
        
        return failure_analysis
    
    async def _update_training_data(self, learning_record: Dict[str, Any]):
        """Update training data with new resolution outcome."""
        try:
            # Load existing training data
            training_file = Path("data/continuous_learning.json")
            
            if training_file.exists():
                with open(training_file, "r") as f:
                    training_data = json.load(f)
            else:
                training_data = {"incidents": [], "metadata": {"last_updated": None}}
            
            # Add new learning record
            training_data["incidents"].append(learning_record)
            training_data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
            training_data["metadata"]["total_incidents"] = len(training_data["incidents"])
            
            # Keep only last 1000 incidents to prevent unlimited growth
            if len(training_data["incidents"]) > 1000:
                training_data["incidents"] = training_data["incidents"][-1000:]
            
            # Save updated training data
            training_file.parent.mkdir(exist_ok=True)
            with open(training_file, "w") as f:
                json.dump(training_data, f, indent=2)
            
            logger.info(f"📚 Updated continuous learning data", 
                       total_incidents=len(training_data["incidents"]))
            
        except Exception as e:
            logger.error(f"❌ Failed to update training data: {e}")
    
    async def _adjust_confidence_thresholds(self, learning_record: Dict[str, Any]):
        """Adjust confidence thresholds based on resolution outcomes."""
        
        success = learning_record["success"]
        confidence = learning_record["confidence_score"]
        
        # If high confidence prediction failed, lower threshold
        if not success and confidence > 0.8:
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
            logger.info(f"📉 Lowered confidence threshold to {self.confidence_threshold:.2f} due to failed high-confidence prediction")
        
        # If low confidence prediction succeeded, we can be more aggressive
        elif success and confidence < 0.6:
            self.confidence_threshold = min(0.8, self.confidence_threshold + 0.02)
            logger.info(f"📈 Raised confidence threshold to {self.confidence_threshold:.2f} due to successful low-confidence prediction")
    
    async def _find_similar_historical_incidents(self, incident_id: str) -> List[Dict[str, Any]]:
        """Find similar historical incidents for pattern analysis."""
        try:
            # Load continuous learning data
            training_file = Path("data/continuous_learning.json")
            
            if not training_file.exists():
                return []
            
            with open(training_file, "r") as f:
                training_data = json.load(f)
            
            # For now, return recent incidents (in production, this would use ML similarity)
            recent_incidents = training_data.get("incidents", [])[-10:]
            return recent_incidents
            
        except Exception as e:
            logger.warning(f"⚠️ Error finding similar incidents: {e}")
            return []
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the continuous learning process."""
        try:
            training_file = Path("data/continuous_learning.json")
            
            if not training_file.exists():
                return {
                    "total_incidents": 0,
                    "success_rate": 0.0,
                    "average_resolution_time": 0,
                    "learning_trends": {}
                }
            
            with open(training_file, "r") as f:
                training_data = json.load(f)
            
            incidents = training_data.get("incidents", [])
            
            if not incidents:
                return {
                    "total_incidents": 0,
                    "success_rate": 0.0,
                    "average_resolution_time": 0,
                    "learning_trends": {}
                }
            
            # Calculate statistics
            successful_incidents = [inc for inc in incidents if inc.get("success", False)]
            success_rate = len(successful_incidents) / len(incidents)
            
            resolution_times = [inc.get("resolution_time", 0) for inc in incidents]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
            
            # Calculate recent trends (last 10 vs previous 10)
            recent_incidents = incidents[-10:] if len(incidents) >= 10 else incidents
            recent_success_rate = len([inc for inc in recent_incidents if inc.get("success", False)]) / len(recent_incidents)
            
            return {
                "total_incidents": len(incidents),
                "success_rate": success_rate,
                "average_resolution_time": avg_resolution_time,
                "recent_success_rate": recent_success_rate,
                "confidence_threshold": getattr(self, 'confidence_threshold', 0.6),
                "learning_trends": {
                    "improving": recent_success_rate > success_rate,
                    "trend_direction": "up" if recent_success_rate > success_rate else "down"
                },
                "last_updated": training_data.get("metadata", {}).get("last_updated")
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting learning statistics: {e}")
            return {"error": str(e)}
