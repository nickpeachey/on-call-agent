# Training Data Schema for AI On-Call Agent

## Overview

The training data structure is designed to enable effective automated resolution and continuous learning. Each incident record contains detailed metadata, action sequences, and learning feedback to improve the AI's decision-making over time.

## Schema Structure

### Base Incident Record

```json
{
  "incident": {
    "title": "string",
    "description": "string", 
    "service": "string",
    "severity": "low|medium|high|critical",
    "tags": ["array", "of", "strings"],
    "metadata": {
      // Service-specific detailed information
    }
  },
  "outcome": "string",
  "resolution_time": "number (seconds)",
  "success": "boolean",
  "actions_taken": [
    // Array of action objects
  ],
  "learning_feedback": {
    // Learning and optimization data
  }
}
```

## Service-Specific Metadata Schemas

### Airflow DAG Issues

```json
"metadata": {
  "dag_id": "string - unique DAG identifier",
  "dag_run_id": "string - specific run identifier", 
  "task_id": "string - failed/stuck task",
  "execution_date": "ISO 8601 datetime",
  "state": "running|failed|success|upstream_failed",
  "duration_seconds": "number",
  "expected_duration_seconds": "number",
  "airflow_instance": "string - server hostname",
  "error_details": {
    "last_heartbeat": "ISO 8601 datetime",
    "worker_log_excerpt": "string",
    "retry_count": "number",
    "max_retries": "number",
    "task_dependencies": ["array", "of", "task_ids"],
    "downstream_tasks_affected": ["array", "of", "task_ids"]
  }
}
```

### Database Connection Issues

```json
"metadata": {
  "database_host": "string - hostname or IP",
  "database_name": "string",
  "database_port": "number",
  "database_type": "postgres|mysql|mongodb|redis",
  "connection_pool_name": "string",
  "pool_config": {
    "max_connections": "number",
    "current_active": "number", 
    "current_idle": "number",
    "wait_queue_size": "number",
    "connection_timeout": "number"
  },
  "error_details": {
    "error_code": "string - database specific error code",
    "sqlstate": "string",
    "timeout_duration": "number",
    "last_successful_connection": "ISO 8601 datetime",
    "failed_attempts_count": "number",
    "connection_string": "string - sanitized connection info"
  },
  "affected_services": ["array", "of", "service_names"]
}
```

### Spark Application Issues

```json
"metadata": {
  "application_id": "string - Spark application ID",
  "application_name": "string",
  "executor_id": "string - if executor-specific",
  "driver_host": "string",
  "spark_master": "string - master URL",
  "error_details": {
    "exception_class": "string - Java exception class",
    "exception_message": "string",
    "executor_memory": "string - e.g., '4g'",
    "executor_max_heap": "string",
    "heap_usage_before_oom": "string",
    "gc_time_percentage": "number",
    "stage_id": "number",
    "task_id": "number",
    "partition_size_mb": "number"
  },
  "resource_config": {
    "executor_memory": "string",
    "executor_cores": "number",
    "num_executors": "number", 
    "driver_memory": "string"
  },
  "data_processed": {
    "input_records": "number",
    "processed_records": "number",
    "failed_at_record": "number"
  }
}
```

### Kubernetes Pod/Container Issues

```json
"metadata": {
  "namespace": "string",
  "pod_name": "string",
  "container_name": "string", 
  "deployment_name": "string",
  "service_name": "string",
  "node_name": "string",
  "error_details": {
    "exit_code": "number",
    "restart_count": "number",
    "last_state": "string",
    "reason": "string - e.g., OOMKilled, CrashLoopBackOff",
    "message": "string",
    "resource_limits": {
      "memory": "string",
      "cpu": "string"
    },
    "resource_requests": {
      "memory": "string", 
      "cpu": "string"
    },
    "current_usage": {
      "memory": "string",
      "cpu": "string"
    }
  }
}
```

### Disk/Storage Issues

```json
"metadata": {
  "server_hostname": "string",
  "filesystem_path": "string",
  "filesystem_type": "string",
  "total_size_gb": "number",
  "used_size_gb": "number",
  "available_size_gb": "number",
  "usage_percentage": "number",
  "inode_usage_percentage": "number",
  "error_details": {
    "disk_growth_rate_gb_per_hour": "number",
    "estimated_time_to_full": "string",
    "largest_directories": [
      {"path": "size"}
    ],
    "recent_large_files": [
      {
        "path": "string",
        "size": "string", 
        "modified": "ISO 8601 datetime"
      }
    ]
  }
}
```

## Action Schema

Each action taken during resolution follows this structure:

```json
{
  "action": "string - action type identifier",
  "parameters": {
    // Action-specific parameters
  },
  "result": "success|failed|partial_success",
  "error": "string - error message if failed",
  "timestamp": "ISO 8601 datetime",
  // Additional result-specific fields
}
```

### Common Action Types

#### Airflow Actions
- `clear_dag_run` - Clear a DAG run to restart
- `trigger_dag_run` - Start a new DAG run
- `pause_dag` - Pause DAG scheduling
- `unpause_dag` - Resume DAG scheduling
- `kill_dag_run` - Force kill running DAG
- `skip_task` - Skip a specific task
- `mark_task_success` - Mark task as successful

#### Database Actions
- `restart_connection_pool` - Restart database connection pool
- `kill_idle_connections` - Kill idle database connections
- `verify_database_connectivity` - Test database connection
- `scale_connection_pool` - Adjust pool size
- `restart_database_service` - Restart database server

#### Spark Actions
- `kill_spark_application` - Kill running Spark application
- `adjust_spark_config` - Modify Spark configuration
- `restart_spark_job` - Restart Spark job with new config
- `scale_spark_cluster` - Add/remove Spark workers

#### Kubernetes Actions
- `restart_pod` - Restart Kubernetes pod
- `scale_deployment` - Scale deployment replicas
- `update_resource_limits` - Modify resource constraints
- `drain_node` - Drain problematic node
- `apply_pod_disruption_budget` - Set disruption limits

#### System Actions
- `restart_service` - Restart system service
- `clean_disk_space` - Free up disk space
- `rotate_logs` - Rotate and compress logs
- `kill_process` - Kill specific process
- `scale_resources` - Adjust resource allocation

## Learning Feedback Schema

```json
"learning_feedback": {
  "confidence_score": "number (0.0-1.0) - AI confidence in decision",
  "pattern_match_strength": "number (0.0-1.0) - How well incident matched known patterns",
  "resolution_effectiveness": "number (0.0-1.0) - How effective the resolution was",
  "similar_incidents_count": "number - Count of similar historical incidents",
  "knowledge_base_updated": "boolean - Whether KB was updated",
  "new_patterns_learned": ["array", "of", "pattern_names"],
  "performance_metrics": {
    "resolution_faster_than_average": "boolean",
    "action_sequence_optimal": "boolean", 
    "resource_usage_efficient": "boolean"
  },
  "failure_analysis": {
    // Only present if success: false
    "root_cause": "string",
    "recommended_escalation": "string",
    "improved_action_sequence": ["array", "of", "actions"],
    "confidence_threshold_adjustment": "number"
  },
  "model_adjustments": {
    "confidence_threshold_lowered": "boolean",
    "new_thresholds": {},
    "escalation_trigger_updated": "boolean",
    "pattern_weights_adjusted": "boolean"
  }
}
```

## Data Collection Guidelines

### For DAG Issues
1. **Always include** `dag_id` and `dag_run_id` for precise targeting
2. **Capture task context** including dependencies and downstream impact
3. **Record timing information** for performance pattern analysis
4. **Include worker information** for resource correlation

### For Database Issues  
1. **Specify exact connection details** including pool configuration
2. **Include connection string** (sanitized) for pattern matching
3. **Record affected services** for impact assessment
4. **Capture timing patterns** for load correlation

### For Spark Issues
1. **Include application and executor IDs** for precise targeting
2. **Record resource configuration** for optimization decisions
3. **Capture data processing context** for checkpoint recovery
4. **Include performance metrics** for resource tuning

### For All Incidents
1. **Detailed error context** - stack traces, error codes, timing
2. **Environmental information** - server, cluster, namespace details
3. **Resource state** - memory, CPU, disk usage at time of incident
4. **Dependency information** - upstream/downstream service impact

## Continuous Learning Implementation

### Pattern Recognition Enhancement
- Each resolution updates pattern matching algorithms
- Failed resolutions create negative training examples
- Success patterns strengthen confidence thresholds
- New incident types expand the knowledge base

### Action Sequence Optimization
- Track which action sequences are most effective
- Learn optimal timing between actions
- Identify when to escalate vs. retry
- Optimize resource allocation during resolution

### Predictive Capabilities
- Use historical patterns to predict incident likelihood
- Preemptive actions based on early warning signs
- Resource scaling based on incident patterns
- Maintenance scheduling to prevent known issues

This enhanced schema enables the AI to make precise, context-aware decisions and continuously improve its resolution capabilities through detailed feedback analysis.
