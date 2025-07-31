#!/usr/bin/env python3
"""
Generate comprehensive training data for AI On-Call Agent.
Creates 10,000+ examples of incidents and their resolutions.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

def generate_comprehensive_training_data() -> List[Dict[str, Any]]:
    """Generate 10,000+ training examples for ML model."""
    
    training_data = []
    
    # Realistic infrastructure data
    server_names = [
        'airflow-prod-01', 'airflow-prod-02', 'airflow-worker-01', 'airflow-worker-02', 'airflow-worker-03',
        'spark-master-01', 'spark-worker-01', 'spark-worker-02', 'spark-worker-03', 'spark-worker-04',
        'db-primary-01', 'db-replica-01', 'db-replica-02', 'redis-cluster-01', 'redis-cluster-02',
        'api-gateway-01', 'api-gateway-02', 'web-api-01', 'web-api-02', 'web-api-03'
    ]
    
    ip_addresses = [
        '10.1.1.10', '10.1.1.11', '10.1.1.12', '10.1.1.13', '10.1.1.14',
        '10.1.2.10', '10.1.2.11', '10.1.2.12', '10.1.2.13', '10.1.2.14',
        '10.1.3.10', '10.1.3.11', '10.1.3.12', '10.1.3.13', '10.1.3.14',
        '10.1.4.10', '10.1.4.11', '10.1.4.12', '10.1.4.13', '10.1.4.14'
    ]
    
    # Airflow DAG incidents (2000 examples) - ENHANCED with infrastructure details
    dag_names = [
        'daily_etl_pipeline', 'customer_analytics', 'financial_reporting', 'user_segmentation',
        'data_warehouse_sync', 'marketing_attribution', 'fraud_detection', 'recommendation_engine',
        'payment_processing', 'inventory_sync', 'sales_reporting', 'user_behavior_analysis',
        'data_quality_checks', 'compliance_reporting', 'real_time_metrics', 'batch_ingestion'
    ]
    
    task_names = [
        'extract_data', 'transform_data', 'load_data', 'validate_data', 'send_notification',
        'check_source_data', 'run_quality_checks', 'archive_data', 'update_metadata',
        'trigger_downstream', 'cleanup_temp_files', 'generate_report'
    ]
    
    dag_error_patterns = [
        "failed with task timeout", "stuck in running state", "failed with memory error",
        "failed with connection timeout", "failed with disk space error", "failed with permission denied",
        "task failed with exit code 1", "sensor timeout waiting for file", "upstream task failed",
        "failed with SQL syntax error", "failed with API rate limiting", "failed with authentication error"
    ]
    
    for i in range(2000):
        dag_name = random.choice(dag_names)
        task_name = random.choice(task_names)
        error_pattern = random.choice(dag_error_patterns)
        server_name = random.choice([s for s in server_names if 'airflow' in s])
        ip_address = random.choice(ip_addresses)
        severity = random.choice(['high', 'critical', 'medium'])
        
        # Generate realistic DAG run ID and execution date
        from datetime import datetime, timedelta
        exec_date = (datetime.now() - timedelta(days=random.randint(0, 7))).strftime('%Y-%m-%d')
        dag_run_id = f"scheduled__{exec_date}T00:00:00+00:00"
        
        # High confidence for restart_dag when pattern suggests restart will work
        if any(keyword in error_pattern for keyword in ['timeout', 'stuck', 'connection', 'exit code']):
            action = 'restart_dag'
            confidence = random.uniform(0.8, 0.95)
        else:
            action = random.choice(['restart_dag', 'check_logs', 'check_data'])
            confidence = random.uniform(0.6, 0.85)
        
        # Enhanced incident data with infrastructure details
        incident_data = {
            "incident": f"Airflow DAG {dag_name} task {task_name} {error_pattern}",
            "severity": severity,
            "action": action,
            "confidence": confidence,
            "category": "airflow_dag_failure",
            "infrastructure": {
                "dag_id": dag_name,
                "task_id": task_name,
                "dag_run_id": dag_run_id,
                "execution_date": exec_date,
                "server_name": server_name,
                "ip_address": ip_address,
                "airflow_url": f"http://{ip_address}:8080",
                "worker_node": random.choice([s for s in server_names if 'worker' in s]),
                "pool": random.choice(['default_pool', 'etl_pool', 'analytics_pool']),
                "queue": random.choice(['default', 'celery', 'kubernetes'])
            },
            "context": {
                "service": "airflow",
                "error_code": random.choice(['1', '2', '124', '255']),
                "retry_count": random.randint(0, 3),
                "max_retries": 3,
                "duration": random.randint(30, 3600),
                "log_url": f"http://{ip_address}:8080/log?dag_id={dag_name}&task_id={task_name}&execution_date={exec_date}"
            }
        }
        
        training_data.append(incident_data)
    
    # Spark job incidents (1500 examples) - ENHANCED
    spark_apps = [
        'customer_segmentation', 'real_time_processing', 'data_transformation', 'ml_training',
        'feature_engineering', 'batch_scoring', 'stream_processing', 'data_aggregation'
    ]
    
    spark_errors = [
        "out of memory error", "executor lost", "driver crashed", "stage failed",
        "serialization exception", "shuffle fetch failed", "disk space full", "network timeout"
    ]
    
    for i in range(1500):
        app_name = random.choice(spark_apps)
        error = random.choice(spark_errors)
        server_name = random.choice([s for s in server_names if 'spark' in s])
        ip_address = random.choice(ip_addresses)
        
        # Generate realistic Spark application ID
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        app_id = f"app-{timestamp}-{random.randint(1000, 9999)}"
        
        if 'memory' in error or 'executor lost' in error:
            action = 'scale_up'
            severity = 'high'
            confidence = random.uniform(0.85, 0.95)
        elif 'driver crashed' in error or 'stage failed' in error:
            action = 'restart_service'
            severity = 'critical'
            confidence = random.uniform(0.8, 0.9)
        else:
            action = random.choice(['restart_service', 'check_logs', 'scale_up'])
            severity = random.choice(['high', 'medium'])
            confidence = random.uniform(0.7, 0.85)
        
        incident_data = {
            "incident": f"Spark job {app_name} {error}",
            "severity": severity,
            "action": action,
            "confidence": confidence,
            "category": "spark_job_failure",
            "infrastructure": {
                "application_id": app_id,
                "application_name": app_name,
                "server_name": server_name,
                "ip_address": ip_address,
                "master_url": f"spark://{ip_address}:7077",
                "spark_ui_url": f"http://{ip_address}:4040",
                "executor_count": random.randint(2, 10),
                "driver_memory": f"{random.choice([2, 4, 8, 16])}g",
                "executor_memory": f"{random.choice([4, 8, 16, 32])}g",
                "cores_per_executor": random.choice([2, 4, 8])
            },
            "context": {
                "service": "spark",
                "stage_id": random.randint(0, 50),
                "task_id": random.randint(1000, 9999),
                "attempt_id": random.randint(0, 3),
                "partition_id": random.randint(0, 200),
                "executor_id": f"executor_{random.randint(1, 10)}",
                "rdd_id": random.randint(1, 100)
            }
        }
        
        training_data.append(incident_data)
    
    # Database incidents (1200 examples) - ENHANCED
    db_types = ['postgresql', 'mysql', 'mongodb', 'redis']
    db_issues = [
        "connection timeout", "high CPU usage", "deadlock detected", "slow query performance",
        "disk space full", "connection pool exhausted", "replication lag", "backup failure",
        "index corruption", "table lock timeout", "memory usage spike", "transaction log full"
    ]
    
    for i in range(1200):
        db_type = random.choice(db_types)
        db_issue = random.choice(db_issues)
        server_name = random.choice([s for s in server_names if 'db' in s or 'redis' in s])
        ip_address = random.choice(ip_addresses)
        
        if 'connection' in db_issue:
            action = 'restart_service'
            severity = 'high'
            confidence = random.uniform(0.8, 0.9)
        elif 'disk space' in db_issue or 'log full' in db_issue:
            action = 'cleanup_disk'
            severity = 'critical'
            confidence = random.uniform(0.85, 0.95)
        elif 'slow query' in db_issue or 'deadlock' in db_issue:
            action = 'optimize_query'
            severity = 'medium'
            confidence = random.uniform(0.75, 0.85)
        elif 'memory' in db_issue or 'CPU' in db_issue:
            action = 'scale_up'
            severity = 'high'
            confidence = random.uniform(0.8, 0.9)
        else:
            action = random.choice(['restart_service', 'check_logs', 'optimize_query'])
            severity = random.choice(['medium', 'high'])
            confidence = random.uniform(0.6, 0.8)
        
        # Database-specific ports
        port_map = {'postgresql': 5432, 'mysql': 3306, 'mongodb': 27017, 'redis': 6379}
        port = port_map.get(db_type, 5432)
        
        incident_data = {
            "incident": f"Database {db_issue} in production environment",
            "severity": severity,
            "action": action,
            "confidence": confidence,
            "category": "database_issue",
            "infrastructure": {
                "database_type": db_type,
                "server_name": server_name,
                "ip_address": ip_address,
                "port": port,
                "connection_string": f"{db_type}://{ip_address}:{port}/production",
                "cluster_role": random.choice(['primary', 'replica', 'standby']),
                "database_name": random.choice(['production', 'analytics', 'warehouse', 'logs']),
                "schema": random.choice(['public', 'analytics', 'reporting', 'etl']),
                "version": random.choice(['13.7', '14.5', '15.2', '16.1'])
            },
            "context": {
                "service": db_type,
                "connection_count": random.randint(50, 500),
                "max_connections": random.choice([100, 200, 500, 1000]),
                "query_duration": random.uniform(0.1, 30.0),
                "lock_wait_time": random.uniform(0, 60.0),
                "table_name": random.choice(['users', 'orders', 'products', 'events', 'logs']),
                "index_name": random.choice(['idx_created_at', 'idx_user_id', 'idx_status', 'idx_email'])
            }
        }
        
        training_data.append(incident_data)
    
    # Network and connectivity issues (800 examples)
    network_issues = [
        "API timeout", "connection refused", "DNS resolution failed", "SSL handshake failed",
        "network latency spike", "packet loss detected", "port unreachable", "firewall blocking"
    ]
    
    for i in range(800):
        network_issue = random.choice(network_issues)
        
        training_data.append({
            "incident": f"Network issue: {network_issue}",
            "severity": random.choice(['medium', 'high']),
            "action": 'check_network',
            "confidence": random.uniform(0.8, 0.95),
            "category": "network_issue"
        })
    
    # Service and application incidents (1500 examples)
    services = [
        'web-api', 'user-service', 'payment-service', 'notification-service',
        'auth-service', 'file-processor', 'email-service', 'analytics-api'
    ]
    
    service_issues = [
        "not responding", "returning 500 errors", "high response time", "memory leak detected",
        "crashed unexpectedly", "health check failing", "startup failed", "configuration error"
    ]
    
    for i in range(1500):
        service = random.choice(services)
        issue = random.choice(service_issues)
        
        if any(keyword in issue for keyword in ['crashed', 'not responding', '500 errors', 'startup failed']):
            action = 'restart_service'
            severity = 'high'
            confidence = random.uniform(0.85, 0.95)
        elif 'memory leak' in issue or 'high response time' in issue:
            action = 'scale_up'
            severity = 'medium'
            confidence = random.uniform(0.75, 0.85)
        elif 'configuration' in issue:
            action = 'check_config'
            severity = 'medium'
            confidence = random.uniform(0.8, 0.9)
        else:
            action = random.choice(['restart_service', 'check_logs'])
            severity = random.choice(['medium', 'high'])
            confidence = random.uniform(0.7, 0.85)
        
        training_data.append({
            "incident": f"Service {service} {issue}",
            "severity": severity,
            "action": action,
            "confidence": confidence,
            "category": "service_failure"
        })
    
    # Infrastructure issues (1000 examples)
    infra_issues = [
        "high CPU usage on server", "disk space running low", "memory usage spike",
        "SSL certificate expired", "load balancer unhealthy", "container restart loop",
        "pod evicted due to resource pressure", "node not ready in cluster"
    ]
    
    for i in range(1000):
        infra_issue = random.choice(infra_issues)
        
        if 'disk space' in infra_issue:
            action = 'cleanup_disk'
            severity = 'high'
            confidence = random.uniform(0.85, 0.95)
        elif 'certificate expired' in infra_issue:
            action = 'renew_certificate'
            severity = 'critical'
            confidence = random.uniform(0.9, 0.98)
        elif any(keyword in infra_issue for keyword in ['CPU', 'memory', 'resource pressure']):
            action = 'scale_up'
            severity = 'high'
            confidence = random.uniform(0.8, 0.9)
        elif 'restart loop' in infra_issue or 'unhealthy' in infra_issue:
            action = 'restart_service'
            severity = 'high'
            confidence = random.uniform(0.8, 0.9)
        else:
            action = random.choice(['check_logs', 'check_config'])
            severity = random.choice(['medium', 'high'])
            confidence = random.uniform(0.6, 0.8)
        
        training_data.append({
            "incident": f"Infrastructure: {infra_issue}",
            "severity": severity,
            "action": action,
            "confidence": confidence,
            "category": "infrastructure_issue"
        })
    
    # Data quality and ETL issues (1000 examples)
    data_issues = [
        "schema drift detected", "null values in required field", "duplicate records found",
        "data freshness SLA violated", "row count mismatch", "data type conversion error",
        "foreign key constraint violation", "data validation failed"
    ]
    
    for i in range(1000):
        data_issue = random.choice(data_issues)
        
        training_data.append({
            "incident": f"Data quality issue: {data_issue}",
            "severity": random.choice(['low', 'medium', 'high']),
            "action": 'check_data',
            "confidence": random.uniform(0.75, 0.9),
            "category": "data_quality_issue"
        })
    
    # Security and authentication issues (500 examples)
    security_issues = [
        "authentication failure spike", "unauthorized access detected", "API key expired",
        "permission denied errors", "token validation failed", "certificate mismatch"
    ]
    
    for i in range(500):
        security_issue = random.choice(security_issues)
        
        if 'certificate' in security_issue or 'API key expired' in security_issue:
            action = 'renew_certificate'
            severity = 'high'
            confidence = random.uniform(0.85, 0.95)
        else:
            action = random.choice(['check_config', 'check_logs'])
            severity = random.choice(['medium', 'high'])
            confidence = random.uniform(0.7, 0.85)
        
        training_data.append({
            "incident": f"Security issue: {security_issue}",
            "severity": severity,
            "action": action,
            "confidence": confidence,
            "category": "security_issue"
        })
    
    # Performance and monitoring issues (500 examples)
    performance_issues = [
        "response time degraded", "throughput below threshold", "cache hit ratio low",
        "queue backlog growing", "resource utilization high", "SLA breach detected"
    ]
    
    for i in range(500):
        perf_issue = random.choice(performance_issues)
        
        if any(keyword in perf_issue for keyword in ['throughput', 'resource utilization', 'queue backlog']):
            action = 'scale_up'
            severity = 'medium'
            confidence = random.uniform(0.75, 0.85)
        elif 'cache' in perf_issue:
            action = 'restart_service'
            severity = 'medium'
            confidence = random.uniform(0.7, 0.8)
        else:
            action = random.choice(['optimize_query', 'check_logs'])
            severity = random.choice(['low', 'medium'])
            confidence = random.uniform(0.6, 0.75)
        
        training_data.append({
            "incident": f"Performance issue: {perf_issue}",
            "severity": severity,
            "action": action,
            "confidence": confidence,
            "category": "performance_issue"
        })
    
    return training_data

if __name__ == "__main__":
    print("Generating comprehensive training data...")
    data = generate_comprehensive_training_data()
    print(f"Generated {len(data)} training examples")
    
    # Save to JSON file
    with open('/Users/nickpeachey/Developer/projects/on-call-agent/data/comprehensive_training.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Training data saved to comprehensive_training.json")
