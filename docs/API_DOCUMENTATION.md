# AI On-Call Agent API Documentation

## Overview

The AI On-Call Agent provides REST APIs for incident submission, analysis, and automated resolution. The system processes incidents through a sophisticated AI pipeline that includes machine learning classification, risk assessment, and automated action execution.

## API Endpoints

### 1. Incident Management

#### Create Incident

**POST** `/api/incidents`

Submit a new incident for AI analysis and potential automated resolution.

**Request Body:**
```json
{
  "title": "Database Connection Timeout",
  "description": "Application experiencing timeouts when connecting to PostgreSQL database",
  "service": "api-service", 
  "severity": "high",
  "tags": ["database", "timeout"],
  "metadata": {
    "source": "monitoring_system",
    "alert_id": "alert-123"
  }
}
```

**Response:**
```json
{
  "id": "incident-456",
  "status": "processing",
  "created_at": "2025-07-31T10:30:00Z",
  "ai_analysis": {
    "queued": true,
    "estimated_processing_time": "2-5 seconds"
  }
}
```

#### Get Incident Status

**GET** `/api/incidents/{incident_id}`

Retrieve current status and analysis results for an incident.

**Response:**
```json
{
  "id": "incident-456",
  "title": "Database Connection Timeout",
  "status": "analyzed",
  "created_at": "2025-07-31T10:30:00Z",
  "updated_at": "2025-07-31T10:30:05Z",
  "ai_analysis": {
    "root_cause_category": "database_connectivity",
    "category_confidence": 0.87,
    "affected_components": ["api-service", "database"],
    "error_patterns": ["timeout", "connection"],
    "recommended_action_types": ["restart_service"],
    "risk_assessment": {
      "level": "medium",
      "factors": ["high_severity"],
      "automation_recommended": true
    },
    "confidence_score": 0.78,
    "analysis_method": "ml_service_models",
    "analysis_timestamp": "2025-07-31T10:30:03Z"
  },
  "automation": {
    "should_automate": true,
    "confidence": 0.75,
    "threshold_met": true,
    "actions_planned": 1
  }
}
```

#### List Incidents

**GET** `/api/incidents`

Retrieve list of incidents with optional filtering.

**Query Parameters:**
- `status`: Filter by status (processing, analyzed, resolving, resolved, failed)
- `service`: Filter by service name
- `severity`: Filter by severity level
- `automated`: Filter by automation status (true/false)
- `limit`: Number of results to return (default: 50)
- `offset`: Pagination offset

**Response:**
```json
{
  "incidents": [
    {
      "id": "incident-456",
      "title": "Database Connection Timeout",
      "service": "api-service",
      "severity": "high", 
      "status": "resolved",
      "automated": true,
      "created_at": "2025-07-31T10:30:00Z",
      "resolved_at": "2025-07-31T10:32:15Z"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### 2. Action Management

#### Get Action History

**GET** `/api/incidents/{incident_id}/actions`

Retrieve all actions executed for an incident.

**Response:**
```json
{
  "incident_id": "incident-456",
  "actions": [
    {
      "id": "action-789",
      "type": "restart_service",
      "parameters": {
        "service_name": "api-service"
      },
      "status": "completed",
      "started_at": "2025-07-31T10:30:10Z",
      "completed_at": "2025-07-31T10:30:45Z",
      "execution_time_seconds": 35,
      "success": true,
      "logs": [
        {
          "timestamp": "2025-07-31T10:30:10Z",
          "level": "info",
          "message": "Starting service restart for api-service"
        },
        {
          "timestamp": "2025-07-31T10:30:45Z", 
          "level": "info",
          "message": "Service restart completed successfully"
        }
      ]
    }
  ]
}
```

#### Manual Action Execution

**POST** `/api/incidents/{incident_id}/actions`

Manually trigger specific actions for an incident.

**Request Body:**
```json
{
  "action_type": "restart_service",
  "parameters": {
    "service_name": "api-service"
  },
  "reason": "Manual intervention requested by operator"
}
```

**Response:**
```json
{
  "action_id": "action-890",
  "status": "queued",
  "estimated_execution_time": "30-60 seconds"
}
```

### 3. AI Analysis

#### Get AI Analysis

**GET** `/api/incidents/{incident_id}/analysis`

Retrieve detailed AI analysis for an incident.

**Response:**
```json
{
  "incident_id": "incident-456",
  "analysis": {
    "root_cause_category": "database_connectivity",
    "category_confidence": 0.87,
    "affected_components": ["api-service", "database"],
    "error_patterns": ["timeout", "connection"],
    "risk_assessment": {
      "level": "medium",
      "factors": ["high_severity"],
      "automation_recommended": true
    },
    "enhanced_metadata": {
      "database_host": "db-server-01",
      "database_port": 5432,
      "timeout_duration": 30,
      "extraction_timestamp": "2025-07-31T10:30:03Z"
    },
    "knowledge_base_matches": [
      {
        "id": "kb-123",
        "title": "PostgreSQL Connection Timeout Resolution",
        "similarity_score": 0.91,
        "success_rate": 0.85,
        "automated_actions": ["restart_service"]
      }
    ],
    "anomaly_detection": {
      "is_anomaly": false,
      "anomaly_score": 0.23,
      "similar_incidents_count": 15
    }
  }
}
```

#### Reanalyze Incident

**POST** `/api/incidents/{incident_id}/reanalyze`

Trigger fresh AI analysis for an incident.

**Request Body:**
```json
{
  "reason": "New information available",
  "force_retrain": false
}
```

**Response:**
```json
{
  "status": "reanalysis_queued", 
  "estimated_time": "2-5 seconds"
}
```

### 4. System Status

#### Health Check

**GET** `/api/health`

Check system health and component status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-31T10:35:00Z",
  "components": {
    "ai_decision_engine": {
      "status": "running",
      "queue_size": 0,
      "processed_today": 45
    },
    "ml_service": {
      "status": "healthy",
      "models_loaded": true,
      "incident_classifier": true,
      "action_recommender": true,
      "text_vectorizer": true
    },
    "knowledge_base": {
      "status": "connected",
      "entries_count": 1247
    },
    "action_execution": {
      "status": "ready",
      "available_actions": 12
    }
  }
}
```

#### System Metrics

**GET** `/api/metrics`

Retrieve system performance metrics.

**Response:**
```json
{
  "period": "24h",
  "metrics": {
    "incidents": {
      "total": 45,
      "automated": 38,
      "manual": 7,
      "success_rate": 0.84
    },
    "analysis": {
      "avg_processing_time_ms": 1847,
      "ml_predictions": 45,
      "fallback_rules": 0
    },
    "actions": {
      "total_executed": 52,
      "successful": 44,
      "failed": 8,
      "avg_execution_time_ms": 28500
    },
    "models": {
      "incident_classifier_accuracy": 0.87,
      "action_recommender_accuracy": 0.78,
      "confidence_calibration": 0.82
    }
  }
}
```

### 5. Configuration

#### Get Configuration

**GET** `/api/config`

Retrieve current system configuration.

**Response:**
```json
{
  "automation": {
    "confidence_threshold": 0.6,
    "max_concurrent_actions": 5,
    "timeout_seconds": 300
  },
  "ml_models": {
    "model_path": "models/",
    "auto_retrain": true,
    "min_training_samples": 10
  },
  "risk_assessment": {
    "critical_services": ["payment", "auth", "critical-data"],
    "automation_disabled_severities": ["critical"]
  }
}
```

#### Update Configuration

**POST** `/api/config`

Update system configuration parameters.

**Request Body:**
```json
{
  "automation": {
    "confidence_threshold": 0.65
  }
}
```

**Response:**
```json
{
  "status": "updated",
  "changes_applied": ["automation.confidence_threshold"],
  "restart_required": false
}
```

## Authentication

All API endpoints require authentication using API keys:

```http
Authorization: Bearer <api_key>
```

## Rate Limits

- **Incident Creation**: 100 requests per minute
- **Status Queries**: 1000 requests per minute  
- **Analysis Requests**: 200 requests per minute
- **Action Execution**: 50 requests per minute

## Error Responses

### Standard Error Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Incident title is required",
    "details": {
      "field": "title",
      "constraint": "min_length_1"
    }
  },
  "timestamp": "2025-07-31T10:35:00Z",
  "request_id": "req-123456"
}
```

### Common Error Codes

- `INVALID_REQUEST`: Malformed request data
- `INCIDENT_NOT_FOUND`: Incident ID does not exist
- `ANALYSIS_FAILED`: AI analysis encountered an error
- `ACTION_FAILED`: Action execution failed
- `RATE_LIMITED`: Too many requests
- `UNAUTHORIZED`: Invalid or missing API key
- `SYSTEM_ERROR`: Internal server error

## WebSocket Events

For real-time updates, connect to the WebSocket endpoint:

**Endpoint:** `ws://api/events`

### Event Types

#### Incident Analysis Complete

```json
{
  "event": "incident_analyzed",
  "incident_id": "incident-456",
  "timestamp": "2025-07-31T10:30:05Z",
  "data": {
    "confidence": 0.78,
    "automation_recommended": true,
    "actions_planned": 1
  }
}
```

#### Action Execution Update

```json
{
  "event": "action_progress", 
  "incident_id": "incident-456",
  "action_id": "action-789",
  "timestamp": "2025-07-31T10:30:30Z",
  "data": {
    "status": "executing",
    "progress": 0.6,
    "message": "Restarting service containers"
  }
}
```

#### Resolution Complete

```json
{
  "event": "incident_resolved",
  "incident_id": "incident-456", 
  "timestamp": "2025-07-31T10:32:15Z",
  "data": {
    "success": true,
    "resolution_time_seconds": 135,
    "actions_executed": 1
  }
}
```

## SDKs and Libraries

### Python SDK

```python
from oncall_agent import OnCallClient

client = OnCallClient(api_key="your-api-key")

# Submit incident
incident = client.create_incident(
    title="Database Connection Timeout",
    description="App timeouts connecting to PostgreSQL",
    service="api-service",
    severity="high"
)

# Wait for analysis
analysis = client.wait_for_analysis(incident.id, timeout=30)
print(f"Confidence: {analysis.confidence_score}")

# Monitor resolution
for event in client.stream_events(incident.id):
    if event.type == "incident_resolved":
        print(f"Resolved in {event.data.resolution_time_seconds}s")
        break
```

### JavaScript SDK

```javascript
import { OnCallAgent } from '@oncall/agent-sdk';

const client = new OnCallAgent({ apiKey: 'your-api-key' });

// Submit incident
const incident = await client.createIncident({
  title: 'Database Connection Timeout',
  description: 'App timeouts connecting to PostgreSQL',
  service: 'api-service', 
  severity: 'high'
});

// Stream real-time events
client.streamEvents(incident.id)
  .on('incident_analyzed', (data) => {
    console.log(`Analysis complete: ${data.confidence}`);
  })
  .on('incident_resolved', (data) => {
    console.log(`Resolved in ${data.resolution_time_seconds}s`);
  });
```

This API enables seamless integration with monitoring systems, allowing automated incident submission and real-time tracking of AI-powered resolution attempts.
