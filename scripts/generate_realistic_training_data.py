#!/usr/bin/env python3
"""
Generate realistic training data with proper metadata for AI On-Call Agent
"""
import random
import json
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'oncall_agent',
    'user': 'oncall_user',
    'password': 'oncall_password'
}

# Realistic data templates
AIRFLOW_INCIDENTS = [
    {
        'title': 'DAG {dag_name} failed in task {task_id}',
        'description': 'Airflow DAG {dag_name} failed at task {task_id} with error: {error_message}. Task instance {task_instance_id} in dag_run {dag_run_id}. Execution date: {execution_date}. Log location: /opt/airflow/logs/{dag_name}/{task_id}/{execution_date}',
        'metadata': {
            'dag_name': ['user_etl_pipeline', 'data_warehouse_sync', 'ml_training_pipeline', 'report_generation', 'data_quality_checks', 'customer_analytics', 'inventory_sync', 'financial_reporting'],
            'task_id': ['extract_data', 'transform_data', 'load_data', 'validate_data', 'send_notification', 'cleanup_temp_files', 'run_quality_checks', 'generate_report'],
            'error_message': [
                'Connection timeout to database server db-prod-01:5432',
                'Memory allocation failed: Cannot allocate 2GB for task execution',
                'File not found: /data/incoming/daily_export_2024-07-31.csv',
                'Permission denied: Cannot write to /data/warehouse/staging/',
                'SQL execution failed: relation "temp_staging_table" does not exist',
                'HTTP 503 Service Unavailable from API endpoint',
                'Task exceeded maximum runtime of 3600 seconds'
            ]
        }
    },
    {
        'title': 'Airflow scheduler unresponsive on {server_name}',
        'description': 'Airflow scheduler process on server {server_name} is not responding. Last heartbeat: {last_heartbeat}. Process ID: {process_id}. Memory usage: {memory_usage}GB. CPU usage: {cpu_usage}%. No new task instances scheduled in the last {minutes_since_last_schedule} minutes.',
        'metadata': {
            'server_name': ['airflow-scheduler-01', 'airflow-scheduler-02', 'airflow-prod-scheduler'],
            'last_heartbeat': ['2024-07-31 14:23:15', '2024-07-31 14:20:42', '2024-07-31 14:18:33'],
            'process_id': ['12345', '23456', '34567'],
            'memory_usage': ['8.2', '12.5', '15.8'],
            'cpu_usage': ['95', '98', '87'],
            'minutes_since_last_schedule': ['15', '22', '8']
        }
    }
]

DATABASE_INCIDENTS = [
    {
        'title': 'Database connection timeout on {server_name}',
        'description': 'PostgreSQL database on server {server_name} experiencing connection timeouts. Active connections: {active_connections}/{max_connections}. Response time: {response_time}ms. Error: {error_code} - {error_message}. Last successful connection: {last_success_time}.',
        'metadata': {
            'server_name': ['db-prod-01', 'db-prod-02', 'db-warehouse-01', 'db-analytics-01'],
            'active_connections': ['95', '148', '200', '180'],
            'max_connections': ['100', '150', '200', '200'],
            'response_time': ['15000', '23000', '8500', '12000'],
            'error_code': ['08006', '08001', '53300'],
            'error_message': [
                'Connection timeout expired',
                'Too many connections',
                'Connection refused',
                'Database is starting up'
            ],
            'last_success_time': ['2024-07-31 14:15:30', '2024-07-31 14:12:45', '2024-07-31 14:08:12']
        }
    },
    {
        'title': 'Deadlock detected in database {database_name}',
        'description': 'Deadlock detected in database {database_name} on table {table_name}. Transaction IDs: {transaction_ids}. Processes involved: {process_ids}. Lock type: {lock_type}. Query: {query_snippet}',
        'metadata': {
            'database_name': ['warehouse', 'analytics', 'user_data', 'inventory'],
            'table_name': ['user_transactions', 'product_inventory', 'order_details', 'customer_profiles'],
            'transaction_ids': ['12345,12346', '23456,23457', '34567,34568'],
            'process_ids': ['1234,1235', '2345,2346', '3456,3457'],
            'lock_type': ['ExclusiveLock', 'ShareLock', 'RowExclusiveLock'],
            'query_snippet': [
                'UPDATE user_transactions SET amount = ... WHERE id = ...',
                'INSERT INTO product_inventory (sku, quantity) VALUES ...',
                'DELETE FROM order_details WHERE order_id = ...'
            ]
        }
    }
]

SPARK_INCIDENTS = [
    {
        'title': 'Spark job {job_name} OOM on executor {executor_id}',
        'description': 'Spark job {job_name} failed with OutOfMemoryError on executor {executor_id}. Driver memory: {driver_memory}GB, Executor memory: {executor_memory}GB. Stage: {stage_id}, Task: {task_id}. Input size: {input_size}GB. Shuffle data: {shuffle_size}GB. Application ID: {app_id}',
        'metadata': {
            'job_name': ['daily_etl_job', 'ml_feature_extraction', 'data_aggregation', 'report_generation'],
            'executor_id': ['1', '2', '3', '4', '5'],
            'driver_memory': ['4', '8', '16'],
            'executor_memory': ['8', '16', '32'],
            'stage_id': ['1', '2', '3', '4'],
            'task_id': ['123', '456', '789'],
            'input_size': ['50', '120', '250'],
            'shuffle_size': ['25', '60', '125'],
            'app_id': ['application_1690880000000_0001', 'application_1690880000000_0002']
        }
    }
]

KUBERNETES_INCIDENTS = [
    {
        'title': 'Pod {pod_name} CrashLoopBackOff in namespace {namespace}',
        'description': 'Pod {pod_name} in namespace {namespace} is in CrashLoopBackOff state. Exit code: {exit_code}. Restart count: {restart_count}. Last termination reason: {termination_reason}. Node: {node_name}. Container image: {image_name}. Resource limits: CPU {cpu_limit}, Memory {memory_limit}',
        'metadata': {
            'pod_name': ['api-service-5f7b8c9d-x8z9p', 'worker-process-6g8h9i-y1a2q', 'data-processor-7h9j0k-z2b3r'],
            'namespace': ['production', 'staging', 'data-pipeline'],
            'exit_code': ['1', '2', '137', '143'],
            'restart_count': ['5', '12', '23'],
            'termination_reason': ['Error', 'OOMKilled', 'Completed'],
            'node_name': ['worker-node-01', 'worker-node-02', 'worker-node-03'],
            'image_name': ['myapp:v1.2.3', 'worker:latest', 'processor:v2.1.0'],
            'cpu_limit': ['500m', '1000m', '2000m'],
            'memory_limit': ['1Gi', '2Gi', '4Gi']
        }
    }
]

API_INCIDENTS = [
    {
        'title': 'API endpoint {endpoint} returning {status_code} errors',
        'description': 'API endpoint {endpoint} on server {server_name} returning {status_code} errors. Error rate: {error_rate}%. Response time: {response_time}ms. Total requests: {total_requests}. Failed requests: {failed_requests}. Error message: {error_message}',
        'metadata': {
            'endpoint': ['/api/v1/users', '/api/v1/orders', '/api/v1/products', '/api/v1/payments'],
            'server_name': ['api-server-01', 'api-server-02', 'api-gateway-01'],
            'status_code': ['500', '502', '503', '504', '429'],
            'error_rate': ['15.5', '23.2', '8.7', '45.1'],
            'response_time': ['5000', '8000', '12000'],
            'total_requests': ['10000', '25000', '50000'],
            'failed_requests': ['1550', '5800', '4350'],
            'error_message': [
                'Internal Server Error: Database connection failed',
                'Bad Gateway: Upstream server timeout',
                'Service Unavailable: Rate limit exceeded',
                'Gateway Timeout: Request processing timeout'
            ]
        }
    }
]

STORAGE_INCIDENTS = [
    {
        'title': 'Disk space critical on server {server_name} partition {partition}',
        'description': 'Disk space on server {server_name} partition {partition} is at {usage_percentage}% ({used_space}GB/{total_space}GB). Available space: {available_space}GB. Largest files: {largest_files}. Growth rate: {growth_rate}GB/hour',
        'metadata': {
            'server_name': ['data-server-01', 'log-server-01', 'backup-server-01'],
            'partition': ['/data', '/var/log', '/opt/backups', '/tmp'],
            'usage_percentage': ['95', '98', '92'],
            'used_space': ['950', '490', '1800'],
            'total_space': ['1000', '500', '2000'],
            'available_space': ['50', '10', '200'],
            'largest_files': [
                '/data/warehouse/daily_export_2024-07-31.csv (25GB)',
                '/var/log/application.log (15GB)',
                '/opt/backups/db_backup_2024-07-30.sql (120GB)'
            ],
            'growth_rate': ['5', '2.5', '10']
        }
    }
]

ALL_INCIDENT_TYPES = [
    ('airflow', AIRFLOW_INCIDENTS),
    ('database', DATABASE_INCIDENTS),
    ('spark-cluster', SPARK_INCIDENTS),
    ('kubernetes', KUBERNETES_INCIDENTS),
    ('api-service', API_INCIDENTS),
    ('storage', STORAGE_INCIDENTS)
]

def generate_realistic_incident(service_type, incident_template):
    """Generate a realistic incident with proper metadata"""
    title_template = incident_template['title']
    description_template = incident_template['description']
    metadata = incident_template['metadata']
    
    # Fill in template variables
    variables = {}
    for key, values in metadata.items():
        variables[key] = random.choice(values)
    
    # Add some randomization
    variables['execution_date'] = (datetime.now() - timedelta(hours=random.randint(1, 24))).strftime('%Y-%m-%d %H:%M:%S')
    variables['task_instance_id'] = f"task_{random.randint(1000, 9999)}"
    variables['dag_run_id'] = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    title = title_template.format(**variables)
    description = description_template.format(**variables)
    
    return title, description, variables

def clear_old_training_data():
    """Remove generic training data"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print("Removing generic training data...")
    cur.execute("""
        DELETE FROM incident_resolutions WHERE incident_id IN (
            SELECT id FROM incidents WHERE description = 'Automated training incident for ML model'
        );
        DELETE FROM incidents WHERE description = 'Automated training incident for ML model';
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Generic training data removed.")

def generate_training_data(num_records=9980):
    """Generate realistic training data"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print(f"Generating {num_records} realistic training records...")
    
    severities = ['low', 'medium', 'high', 'critical']
    severity_weights = [0.2, 0.4, 0.3, 0.1]  # More medium/high incidents
    
    for i in range(num_records):
        # Choose service type and incident template
        service_type, incident_templates = random.choice(ALL_INCIDENT_TYPES)
        incident_template = random.choice(incident_templates)
        
        # Generate realistic incident
        title, description, metadata = generate_realistic_incident(service_type, incident_template)
        
        # Choose severity based on service type
        if service_type in ['airflow', 'database']:
            severity = random.choices(severities, weights=[0.1, 0.3, 0.4, 0.2])[0]
        else:
            severity = random.choices(severities, weights=severity_weights)[0]
        
        # Insert incident
        created_at = datetime.now() - timedelta(days=random.randint(1, 90))
        updated_at = created_at + timedelta(minutes=random.randint(10, 300))
        
        cur.execute("""
            INSERT INTO incidents (title, description, service, severity, status, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 'resolved', %s, %s, %s)
            RETURNING id
        """, (title, description, service_type, severity, json.dumps(metadata), created_at, updated_at))
        
        incident_id = cur.fetchone()[0]
        
        # Generate resolution
        success_rate = {
            'critical': 0.75,
            'high': 0.85,
            'medium': 0.90,
            'low': 0.95
        }
        
        resolution_time = {
            'critical': random.randint(60, 300),
            'high': random.randint(120, 600),
            'medium': random.randint(300, 1200),
            'low': random.randint(600, 2400)
        }
        
        success = random.random() < success_rate[severity]
        time_to_resolve = resolution_time[severity]
        
        # Choose appropriate action
        if service_type == 'airflow':
            actions = ['restart_dag', 'clear_dag_tasks', 'retry_failed_task', 'restart_airflow_scheduler']
        elif service_type == 'database':
            actions = ['restart_database', 'kill_hung_process', 'restart_service']
        elif service_type == 'kubernetes':
            actions = ['scale_pods', 'restart_service']
        else:
            actions = ['restart_service', 'clear_cache', 'check_disk_space']
        
        executed_actions = [random.choice(actions)]
        
        cur.execute("""
            INSERT INTO incident_resolutions (incident_id, success, resolution_time, actions_executed, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (incident_id, success, time_to_resolve, executed_actions, updated_at))
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} records...")
            conn.commit()
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"Generated {num_records} realistic training records!")

if __name__ == "__main__":
    clear_old_training_data()
    generate_training_data()
    print("Training data generation complete!")
