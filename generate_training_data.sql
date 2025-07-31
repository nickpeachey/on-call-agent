-- Generate 10,000 comprehensive training records
DO $$
DECLARE
    i INT;
    incident_id UUID;
    incident_titles TEXT[] := ARRAY[
        'Airflow DAG failure in data pipeline',
        'Database connection timeout',
        'High memory usage in Spark cluster',
        'Kubernetes pod crash loop',
        'API rate limit exceeded',
        'Disk space full on server',
        'Redis cache miss rate high',
        'SSL certificate expired',
        'Network timeout in microservice',
        'Log aggregation pipeline failed',
        'ETL job memory overflow',
        'Authentication service down',
        'File upload service error',
        'Background job queue backed up',
        'Load balancer health check failing',
        'Database deadlock detected',
        'Airflow scheduler not responding',
        'Spark driver out of memory',
        'Container registry unavailable',
        'Message queue consumer lag',
        'Data warehouse sync failed',
        'Monitoring alert storm',
        'CDN cache invalidation failed',
        'Search index corruption',
        'Payment gateway timeout',
        'Email delivery service down',
        'Session timeout issues',
        'Cross-region replication lag',
        'Backup job failed',
        'Log rotation service error',
        'Airflow task dependency failure',
        'Database migration stuck',
        'API gateway circuit breaker open',
        'Distributed cache eviction',
        'Stream processing backlog',
        'Compliance scan failures',
        'Auto-scaling policy triggered',
        'Health check endpoint timeout',
        'Certificate renewal failed',
        'Data validation pipeline error'
    ];
    
    services TEXT[] := ARRAY[
        'airflow', 'database', 'api-service', 'spark-cluster', 'redis', 
        'kubernetes', 'storage', 'auth', 'monitoring', 'logging',
        'etl-pipeline', 'web-app', 'cache', 'messaging', 'network',
        'security', 'payments', 'email', 'search', 'analytics'
    ];
    
    severities TEXT[] := ARRAY['low', 'medium', 'high', 'critical'];
    
    airflow_actions TEXT[] := ARRAY[
        'restart_dag', 'clear_dag_tasks', 'pause_dag', 'unpause_dag',
        'restart_airflow_scheduler', 'retry_failed_task', 'skip_failed_task',
        'check_dag_dependencies', 'restart_dag_run', 'reset_dag_state'
    ];
    
    general_actions TEXT[] := ARRAY[
        'restart_service', 'scale_pods', 'clear_cache', 'restart_database',
        'check_disk_space', 'kill_hung_process'
    ];
    
    action_set TEXT[];
    selected_title TEXT;
    selected_service TEXT;
    selected_severity TEXT;
    resolution_time INT;
    success_rate BOOLEAN;
    
BEGIN
    FOR i IN 1..9980 LOOP  -- Generate 9,980 more to reach 10,000 total
        -- Select random incident details
        selected_title := incident_titles[1 + (random() * array_length(incident_titles, 1))::INT];
        selected_service := services[1 + (random() * array_length(services, 1))::INT];
        selected_severity := severities[1 + (random() * array_length(severities, 1))::INT];
        
        -- Determine resolution time based on severity
        resolution_time := CASE 
            WHEN selected_severity = 'critical' THEN 60 + (random() * 240)::INT
            WHEN selected_severity = 'high' THEN 120 + (random() * 480)::INT
            WHEN selected_severity = 'medium' THEN 300 + (random() * 900)::INT
            ELSE 600 + (random() * 1800)::INT
        END;
        
        -- Higher success rate for automated actions
        success_rate := CASE
            WHEN selected_severity IN ('low', 'medium') THEN random() > 0.1
            WHEN selected_severity = 'high' THEN random() > 0.15
            ELSE random() > 0.25
        END;
        
        -- Choose appropriate action set
        IF selected_service = 'airflow' OR selected_title ILIKE '%airflow%' OR selected_title ILIKE '%dag%' THEN
            action_set := airflow_actions;
        ELSE
            action_set := general_actions;
        END IF;
        
        -- Insert incident
        INSERT INTO incidents (title, description, service, severity, status, created_at, updated_at)
        VALUES (
            selected_title,
            'Automated incident for service ' || selected_service || ' with severity ' || selected_severity,
            selected_service,
            selected_severity,
            'resolved',
            NOW() - (random() * INTERVAL '90 days'),
            NOW() - (random() * INTERVAL '89 days')
        ) RETURNING id INTO incident_id;
        
        -- Insert resolution
        INSERT INTO incident_resolutions (incident_id, success, resolution_time, actions_executed, created_at)
        VALUES (
            incident_id,
            success_rate,
            resolution_time,
            ARRAY[action_set[1 + (random() * array_length(action_set, 1))::INT]],
            NOW() - (random() * INTERVAL '89 days')
        );
        
        -- Log progress every 1000 records
        IF i % 1000 = 0 THEN
            RAISE NOTICE 'Generated % records (total with original: %)', i, i + 20;
        END IF;
    END LOOP;
    
    RAISE NOTICE 'Training data generation complete! Total records: %', (SELECT COUNT(*) FROM incidents);
END
$$;
