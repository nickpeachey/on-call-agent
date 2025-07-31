-- Clear existing training data and regenerate with proper metadata
\echo 'Clearing existing incident data...'
DELETE FROM incident_resolutions;
DELETE FROM incidents;

\echo 'Starting generation of 10,000 realistic training incidents with structured metadata...'

-- Generate 10,000 realistic training records with proper metadata
DO $$
DECLARE
    i INT;
    incident_id UUID;
    
    -- Airflow-specific data
    dag_names TEXT[] := ARRAY[
        'data_pipeline_etl', 'user_analytics_daily', 'financial_reporting', 'data_warehouse_sync',
        'log_processing_hourly', 'ml_model_training', 'backup_database_nightly', 'data_quality_checks',
        'customer_segmentation', 'inventory_sync', 'sales_reporting', 'fraud_detection_pipeline',
        'recommendation_engine', 'real_time_alerts', 'data_validation_daily'
    ];
    
    task_names TEXT[] := ARRAY[
        'extract_data', 'transform_data', 'load_to_warehouse', 'validate_schema', 'send_notifications',
        'cleanup_temp_files', 'generate_report', 'update_dashboard', 'run_quality_checks',
        'backup_results', 'send_email_alert', 'update_metadata', 'archive_old_data'
    ];
    
    -- Server/infrastructure data
    server_names TEXT[] := ARRAY[
        'prod-web-01', 'prod-web-02', 'prod-db-01', 'prod-cache-01', 'prod-worker-01',
        'staging-app-01', 'dev-test-01', 'analytics-cluster-01', 'backup-server-01',
        'monitoring-01', 'elk-stack-01', 'redis-cluster-01', 'spark-master-01', 'spark-worker-01'
    ];
    
    error_messages TEXT[] := ARRAY[
        'Connection timeout after 30 seconds',
        'Out of memory: Java heap space',
        'Database connection pool exhausted',
        'SSL handshake failed',
        'Task failed with exit code 1',
        'Permission denied: /var/log/airflow',
        'Disk space usage above 90%',
        'Network unreachable: 10.0.1.100',
        'Authentication failed for user',
        'Lock wait timeout exceeded',
        'Container killed due to memory limit',
        'Certificate has expired',
        'Max retries exceeded',
        'HTTP 503 Service Unavailable',
        'Deadlock detected'
    ];
    
    -- Variables for random selection
    selected_dag TEXT;
    selected_task TEXT;
    selected_server TEXT;
    selected_error TEXT;
    selected_service TEXT;
    selected_severity TEXT;
    selected_title TEXT;
    selected_description TEXT;
    resolution_time INT;
    success_rate BOOLEAN;
    
    services TEXT[] := ARRAY[
        'airflow', 'database', 'api-service', 'spark-cluster', 'redis', 
        'kubernetes', 'storage', 'auth', 'monitoring', 'logging',
        'etl-pipeline', 'web-app', 'cache', 'messaging', 'network',
        'security', 'payments', 'email', 'search', 'analytics'
    ];
    
    severities TEXT[] := ARRAY['low', 'medium', 'high', 'critical'];
    
    -- Realistic status distribution
    statuses TEXT[] := ARRAY['open', 'in_progress', 'resolved', 'closed'];
    
    -- Variables for realistic status selection
    selected_status TEXT;
    should_have_resolution BOOLEAN;
    
BEGIN
    RAISE NOTICE 'Generating 10,000 realistic training incidents with structured metadata...';
    
    FOR i IN 1..10000 LOOP
        -- Select random elements with safe bounds
        selected_service := COALESCE(services[1 + (random() * (GREATEST(array_length(services, 1), 1) - 1))::INT], 'monitoring');
        selected_severity := COALESCE(severities[1 + (random() * (GREATEST(array_length(severities, 1), 1) - 1))::INT], 'medium');
        selected_dag := COALESCE(dag_names[1 + (random() * (GREATEST(array_length(dag_names, 1), 1) - 1))::INT], 'default_dag');
        selected_task := COALESCE(task_names[1 + (random() * (GREATEST(array_length(task_names, 1), 1) - 1))::INT], 'default_task');
        selected_server := COALESCE(server_names[1 + (random() * (GREATEST(array_length(server_names, 1), 1) - 1))::INT], 'server-01');
        selected_error := COALESCE(error_messages[1 + (random() * (GREATEST(array_length(error_messages, 1), 1) - 1))::INT], 'Unknown error');
        
        -- Generate service-specific incidents with proper metadata
        IF selected_service = 'airflow' THEN
            selected_title := CASE (random() * 8)::INT
                WHEN 0 THEN 'DAG ' || selected_dag || ' failed on task ' || selected_task
                WHEN 1 THEN 'Airflow scheduler not responding on ' || selected_server
                WHEN 2 THEN 'Task ' || selected_task || ' in DAG ' || selected_dag || ' stuck in running state'
                WHEN 3 THEN 'DAG ' || selected_dag || ' missing dependencies'
                WHEN 4 THEN 'Airflow worker ' || selected_server || ' out of memory'
                WHEN 5 THEN 'Task ' || selected_task || ' retry limit exceeded in ' || selected_dag
                WHEN 6 THEN 'Airflow webserver timeout on ' || selected_server
                ELSE 'DAG ' || selected_dag || ' execution timeout'
            END;
            
            selected_description := 'Airflow incident on server ' || selected_server || 
                '. DAG: ' || selected_dag || ', Task: ' || selected_task || 
                '. Error: ' || selected_error || 
                '. Log location: /var/log/airflow/' || selected_dag || '_' || 
                to_char(NOW() - (random() * INTERVAL '90 days'), 'YYYY-MM-DD') || '.log';
                
        ELSIF selected_service = 'database' THEN
            selected_title := CASE (random() * 6)::INT
                WHEN 0 THEN 'Database connection timeout on ' || selected_server
                WHEN 1 THEN 'Deadlock detected on ' || selected_server
                WHEN 2 THEN 'High CPU usage on database ' || selected_server
                WHEN 3 THEN 'Database backup failed on ' || selected_server
                WHEN 4 THEN 'Slow query detected on ' || selected_server
                ELSE 'Database disk space low on ' || selected_server
            END;
            
            selected_description := 'Database incident on server ' || selected_server || 
                '. Error: ' || selected_error || 
                '. Query time: ' || (50 + random() * 5000)::INT || 'ms' ||
                '. Affected tables: user_data, transactions';
                
        ELSIF selected_service = 'spark-cluster' THEN
            selected_title := CASE (random() * 5)::INT
                WHEN 0 THEN 'Spark job OOM on cluster ' || selected_server
                WHEN 1 THEN 'Spark driver failure on ' || selected_server
                WHEN 2 THEN 'Spark executor lost on ' || selected_server
                WHEN 3 THEN 'Spark shuffle service down on ' || selected_server
                ELSE 'Spark job timeout on ' || selected_server
            END;
            
            selected_description := 'Spark cluster incident on ' || selected_server || 
                '. Job ID: job_' || (1000 + random() * 9000)::INT ||
                '. Error: ' || selected_error ||
                '. Memory usage: ' || (50 + random() * 50)::INT || 'GB';
                
        ELSIF selected_service = 'kubernetes' THEN
            selected_title := CASE (random() * 6)::INT
                WHEN 0 THEN 'Pod crash loop on ' || selected_server
                WHEN 1 THEN 'Service unavailable in namespace prod'
                WHEN 2 THEN 'Node ' || selected_server || ' not ready'
                WHEN 3 THEN 'Pod evicted due to resource pressure'
                WHEN 4 THEN 'Ingress controller timeout'
                ELSE 'ConfigMap update failed'
            END;
            
            selected_description := 'Kubernetes incident on ' || selected_server || 
                '. Namespace: production' ||
                '. Pod: app-' || (100 + random() * 900)::INT ||
                '. Error: ' || selected_error;
                
        ELSE
            -- Generic service incidents
            selected_title := CASE (random() * 5)::INT
                WHEN 0 THEN selected_service || ' service down on ' || selected_server
                WHEN 1 THEN 'High latency in ' || selected_service || ' on ' || selected_server
                WHEN 2 THEN selected_service || ' authentication failure'
                WHEN 3 THEN selected_service || ' disk space alert on ' || selected_server
                ELSE selected_service || ' connection timeout'
            END;
            
            selected_description := 'Service incident on ' || selected_server || 
                '. Service: ' || selected_service ||
                '. Error: ' || selected_error ||
                '. Response time: ' || (100 + random() * 2000)::INT || 'ms';
        END IF;
        
        -- Determine resolution time based on severity
        resolution_time := CASE 
            WHEN selected_severity = 'critical' THEN 60 + (random() * 240)::INT
            WHEN selected_severity = 'high' THEN 120 + (random() * 480)::INT
            WHEN selected_severity = 'medium' THEN 300 + (random() * 900)::INT
            ELSE 600 + (random() * 1800)::INT
        END;
        
        -- Success rate based on severity
        success_rate := CASE
            WHEN selected_severity IN ('low', 'medium') THEN random() > 0.1
            WHEN selected_severity = 'high' THEN random() > 0.15
            ELSE random() > 0.25
        END;
        
        -- Realistic status distribution based on severity and time
        selected_status := CASE 
            WHEN selected_severity = 'critical' THEN
                CASE 
                    WHEN random() < 0.7 THEN 'resolved'
                    WHEN random() < 0.85 THEN 'closed'
                    WHEN random() < 0.95 THEN 'in_progress'
                    ELSE 'open'
                END
            WHEN selected_severity = 'high' THEN
                CASE 
                    WHEN random() < 0.6 THEN 'resolved'
                    WHEN random() < 0.8 THEN 'closed'
                    WHEN random() < 0.92 THEN 'in_progress'
                    ELSE 'open'
                END
            WHEN selected_severity = 'medium' THEN
                CASE 
                    WHEN random() < 0.5 THEN 'resolved'
                    WHEN random() < 0.7 THEN 'closed'
                    WHEN random() < 0.88 THEN 'in_progress'
                    ELSE 'open'
                END
            ELSE -- low severity
                CASE 
                    WHEN random() < 0.3 THEN 'resolved'
                    WHEN random() < 0.5 THEN 'closed'
                    WHEN random() < 0.8 THEN 'in_progress'
                    ELSE 'open'
                END
        END;
        
        -- Only resolved/closed incidents should have resolutions
        should_have_resolution := selected_status IN ('resolved', 'closed');
        
        -- Insert incident with proper metadata and realistic status
        INSERT INTO incidents (title, description, service, severity, status, metadata, created_at, updated_at, resolved_at)
        VALUES (
            selected_title,
            selected_description,
            selected_service,
            selected_severity,
            selected_status,
            CASE 
                WHEN selected_service = 'airflow' THEN
                    jsonb_build_object(
                        'dag_name', selected_dag,
                        'task_name', selected_task,
                        'dag_id', selected_dag || '_' || (1000 + random() * 9000)::INT,
                        'task_id', selected_task || '_' || (100 + random() * 900)::INT,
                        'run_id', 'manual_' || to_char(NOW() - (random() * INTERVAL '90 days'), 'YYYY-MM-DD_HH24-MI-SS'),
                        'server_name', selected_server,
                        'error_code', (100 + random() * 500)::INT,
                        'log_path', '/var/log/airflow/' || selected_dag || '_' || to_char(NOW() - (random() * INTERVAL '90 days'), 'YYYY-MM-DD') || '.log',
                        'retry_count', (random() * 5)::INT,
                        'execution_date', to_char(NOW() - (random() * INTERVAL '90 days'), 'YYYY-MM-DD HH24:MI:SS')
                    )
                WHEN selected_service = 'spark-cluster' THEN
                    jsonb_build_object(
                        'job_id', 'job_' || (1000 + random() * 9000)::INT,
                        'application_id', 'application_' || (1000 + random() * 9000)::INT,
                        'server_name', selected_server,
                        'driver_memory', (2 + random() * 16)::INT || 'GB',
                        'executor_memory', (1 + random() * 8)::INT || 'GB',
                        'executor_cores', (1 + random() * 8)::INT,
                        'cluster_id', 'cluster_' || (1 + random() * 10)::INT,
                        'error_code', (100 + random() * 500)::INT,
                        'stage_id', (random() * 100)::INT,
                        'task_count', (10 + random() * 1000)::INT
                    )
                WHEN selected_service = 'database' THEN
                    jsonb_build_object(
                        'server_name', selected_server,
                        'database_name', (ARRAY['production', 'analytics', 'reporting', 'staging'])[1 + (random() * 3)::INT],
                        'query_id', 'query_' || (1000 + random() * 9000)::INT,
                        'connection_id', (10000 + random() * 90000)::INT,
                        'table_name', (ARRAY['users', 'transactions', 'orders', 'products', 'logs'])[1 + (random() * 4)::INT],
                        'query_duration_ms', (50 + random() * 5000)::INT,
                        'rows_affected', (random() * 10000)::INT,
                        'error_code', (1000 + random() * 9000)::INT,
                        'lock_timeout', (random() > 0.7)::BOOLEAN
                    )
                WHEN selected_service = 'kubernetes' THEN
                    jsonb_build_object(
                        'namespace', 'production',
                        'pod_name', 'app-' || (100 + random() * 900)::INT,
                        'node_name', selected_server,
                        'container_name', (ARRAY['app', 'sidecar', 'init', 'proxy'])[1 + (random() * 3)::INT],
                        'deployment_name', (ARRAY['web-app', 'api-service', 'worker', 'scheduler'])[1 + (random() * 3)::INT],
                        'replica_count', (1 + random() * 10)::INT,
                        'restart_count', (random() * 20)::INT,
                        'cpu_limit', (100 + random() * 2000)::INT || 'm',
                        'memory_limit', (512 + random() * 4096)::INT || 'Mi',
                        'exit_code', (random() * 255)::INT
                    )
                ELSE
                    jsonb_build_object(
                        'server_name', selected_server,
                        'service_name', selected_service,
                        'port', (8000 + random() * 2000)::INT,
                        'process_id', (1000 + random() * 30000)::INT,
                        'cpu_usage', (random() * 100)::DECIMAL(5,2) || '%',
                        'memory_usage', (random() * 100)::DECIMAL(5,2) || '%',
                        'disk_usage', (random() * 100)::DECIMAL(5,2) || '%',
                        'response_time_ms', (100 + random() * 2000)::INT,
                        'error_code', (400 + random() * 200)::INT,
                        'upstream_service', (ARRAY['auth-service', 'database', 'cache', 'queue'])[1 + (random() * 3)::INT]
                    )
            END,
            NOW() - (random() * INTERVAL '90 days'),
            NOW() - (random() * INTERVAL '89 days'),
            CASE 
                WHEN selected_status IN ('resolved', 'closed') THEN NOW() - (random() * INTERVAL '88 days')
                ELSE NULL
            END
        ) RETURNING id INTO incident_id;
        
        -- Insert resolution only for resolved/closed incidents
        IF should_have_resolution THEN
            INSERT INTO incident_resolutions (incident_id, success, resolution_time, actions_executed, created_at)
            VALUES (
                incident_id,
                success_rate,
                resolution_time,
                CASE 
                    WHEN selected_service = 'airflow' OR selected_title ILIKE '%airflow%' OR selected_title ILIKE '%dag%' THEN
                        ARRAY[COALESCE((ARRAY['restart_dag', 'clear_dag_tasks', 'pause_dag', 'unpause_dag', 'restart_airflow_scheduler', 'retry_failed_task'])[1 + (random() * 5)::INT], 'restart_dag')]
                    ELSE
                        ARRAY[COALESCE((ARRAY['restart_service', 'scale_pods', 'clear_cache', 'restart_database', 'check_disk_space', 'kill_hung_process'])[1 + (random() * 5)::INT], 'restart_service')]
                END,
                NOW() - (random() * INTERVAL '87 days')
            );
        END IF;
        
        -- Log progress every 1000 records
        IF i % 1000 = 0 THEN
            RAISE NOTICE 'Generated % training incidents...', i;
        END IF;
    END LOOP;
    
    RAISE NOTICE 'Training data generation complete! Total incidents: %', (SELECT COUNT(*) FROM incidents);
END
$$;

\echo 'Training data generation complete!'
