-- Enhanced database schema for action logging and resolution monitoring
-- This extends the existing schema with detailed action tracking capabilities

-- Create action_attempts table for detailed action execution logging
CREATE TABLE IF NOT EXISTS action_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    action_id VARCHAR(255) NOT NULL, -- The action execution ID
    incident_id UUID REFERENCES incidents(id) ON DELETE CASCADE,
    action_type VARCHAR(100) NOT NULL,
    parameters JSONB DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'success', 'failed')),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_seconds DECIMAL(10,3),
    sequence_position INTEGER, -- Position in action sequence (1, 2, 3...)
    result JSONB DEFAULT '{}',
    error_message TEXT,
    exception_details JSONB DEFAULT '{}',
    execution_logs JSONB DEFAULT '[]', -- Array of log entries
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create resolution_attempts table for tracking complete resolution attempts
CREATE TABLE IF NOT EXISTS resolution_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID REFERENCES incidents(id) ON DELETE CASCADE,
    ai_confidence DECIMAL(4,3), -- AI confidence score (0.000-1.000)
    resolution_method VARCHAR(50) NOT NULL CHECK (resolution_method IN ('automated', 'manual', 'escalated')),
    overall_success BOOLEAN NOT NULL DEFAULT false,
    total_actions INTEGER DEFAULT 0,
    successful_actions INTEGER DEFAULT 0,
    failed_actions INTEGER DEFAULT 0,
    total_execution_time_seconds DECIMAL(10,3),
    resolution_notes TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create action_logs table for step-by-step action execution logs
CREATE TABLE IF NOT EXISTS action_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    action_attempt_id UUID REFERENCES action_attempts(id) ON DELETE CASCADE,
    step_name VARCHAR(100) NOT NULL,
    step_status VARCHAR(50) NOT NULL,
    step_details JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create resolution_metrics table for aggregated monitoring data
CREATE TABLE IF NOT EXISTS resolution_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date_bucket DATE NOT NULL,
    hour_bucket INTEGER CHECK (hour_bucket >= 0 AND hour_bucket <= 23),
    service VARCHAR(100),
    action_type VARCHAR(100),
    total_attempts INTEGER DEFAULT 0,
    successful_attempts INTEGER DEFAULT 0,
    failed_attempts INTEGER DEFAULT 0,
    avg_execution_time_seconds DECIMAL(10,3),
    min_execution_time_seconds DECIMAL(10,3),
    max_execution_time_seconds DECIMAL(10,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(date_bucket, hour_bucket, service, action_type)
);

-- Create failure_patterns table for tracking common failure reasons
CREATE TABLE IF NOT EXISTS failure_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    error_signature VARCHAR(500) NOT NULL, -- First 500 chars of error message
    action_type VARCHAR(100) NOT NULL,
    service VARCHAR(100),
    occurrence_count INTEGER DEFAULT 1,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolution_suggestions TEXT,
    is_known_issue BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(error_signature, action_type, service)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_action_attempts_incident_id ON action_attempts(incident_id);
CREATE INDEX IF NOT EXISTS idx_action_attempts_action_type ON action_attempts(action_type);
CREATE INDEX IF NOT EXISTS idx_action_attempts_status ON action_attempts(status);
CREATE INDEX IF NOT EXISTS idx_action_attempts_started_at ON action_attempts(started_at);
CREATE INDEX IF NOT EXISTS idx_action_attempts_sequence_position ON action_attempts(sequence_position);

CREATE INDEX IF NOT EXISTS idx_resolution_attempts_incident_id ON resolution_attempts(incident_id);
CREATE INDEX IF NOT EXISTS idx_resolution_attempts_resolution_method ON resolution_attempts(resolution_method);
CREATE INDEX IF NOT EXISTS idx_resolution_attempts_started_at ON resolution_attempts(started_at);
CREATE INDEX IF NOT EXISTS idx_resolution_attempts_overall_success ON resolution_attempts(overall_success);

CREATE INDEX IF NOT EXISTS idx_action_logs_action_attempt_id ON action_logs(action_attempt_id);
CREATE INDEX IF NOT EXISTS idx_action_logs_step_name ON action_logs(step_name);
CREATE INDEX IF NOT EXISTS idx_action_logs_timestamp ON action_logs(timestamp);

CREATE INDEX IF NOT EXISTS idx_resolution_metrics_date_bucket ON resolution_metrics(date_bucket);
CREATE INDEX IF NOT EXISTS idx_resolution_metrics_service ON resolution_metrics(service);
CREATE INDEX IF NOT EXISTS idx_resolution_metrics_action_type ON resolution_metrics(action_type);

CREATE INDEX IF NOT EXISTS idx_failure_patterns_action_type ON failure_patterns(action_type);
CREATE INDEX IF NOT EXISTS idx_failure_patterns_service ON failure_patterns(service);
CREATE INDEX IF NOT EXISTS idx_failure_patterns_last_seen ON failure_patterns(last_seen);

-- Create views for easy monitoring queries

-- View for recent action success rates by type
CREATE OR REPLACE VIEW action_success_rates AS
SELECT 
    action_type,
    COUNT(*) as total_attempts,
    COUNT(*) FILTER (WHERE status = 'success') as successful_attempts,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_attempts,
    ROUND(
        COUNT(*) FILTER (WHERE status = 'success')::DECIMAL / COUNT(*) * 100, 2
    ) as success_rate_percentage,
    ROUND(AVG(execution_time_seconds), 3) as avg_execution_time_seconds
FROM action_attempts 
WHERE started_at >= NOW() - INTERVAL '24 hours'
GROUP BY action_type
ORDER BY total_attempts DESC;

-- View for incident resolution performance
CREATE OR REPLACE VIEW incident_resolution_performance AS
SELECT 
    i.service,
    i.severity,
    COUNT(ra.*) as total_resolution_attempts,
    COUNT(*) FILTER (WHERE ra.overall_success = true) as successful_resolutions,
    ROUND(
        COUNT(*) FILTER (WHERE ra.overall_success = true)::DECIMAL / COUNT(ra.*) * 100, 2
    ) as success_rate_percentage,
    ROUND(AVG(ra.total_execution_time_seconds), 3) as avg_resolution_time_seconds,
    ROUND(AVG(ra.ai_confidence), 3) as avg_ai_confidence
FROM incidents i
LEFT JOIN resolution_attempts ra ON i.id = ra.incident_id
WHERE i.created_at >= NOW() - INTERVAL '7 days'
GROUP BY i.service, i.severity
ORDER BY total_resolution_attempts DESC;

-- View for recent failure patterns
CREATE OR REPLACE VIEW recent_failure_patterns AS
SELECT 
    fp.action_type,
    fp.service,
    fp.error_signature,
    fp.occurrence_count,
    fp.last_seen,
    fp.is_known_issue,
    fp.resolution_suggestions
FROM failure_patterns fp
WHERE fp.last_seen >= NOW() - INTERVAL '7 days'
ORDER BY fp.occurrence_count DESC, fp.last_seen DESC;

-- View for action execution timeline
CREATE OR REPLACE VIEW action_execution_timeline AS
SELECT 
    aa.action_id,
    aa.incident_id,
    aa.action_type,
    aa.status,
    aa.started_at,
    aa.completed_at,
    aa.execution_time_seconds,
    aa.sequence_position,
    i.title as incident_title,
    i.service as incident_service,
    i.severity as incident_severity
FROM action_attempts aa
JOIN incidents i ON aa.incident_id = i.id
WHERE aa.started_at >= NOW() - INTERVAL '24 hours'
ORDER BY aa.started_at DESC, aa.sequence_position ASC;

-- Function to automatically update resolution metrics
CREATE OR REPLACE FUNCTION update_resolution_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update or insert metrics when action attempt is completed
    IF NEW.completed_at IS NOT NULL AND OLD.completed_at IS NULL THEN
        INSERT INTO resolution_metrics (
            date_bucket,
            hour_bucket,
            service,
            action_type,
            total_attempts,
            successful_attempts,
            failed_attempts,
            avg_execution_time_seconds,
            min_execution_time_seconds,
            max_execution_time_seconds
        )
        SELECT 
            DATE(NEW.started_at),
            EXTRACT(HOUR FROM NEW.started_at)::INTEGER,
            i.service,
            NEW.action_type,
            1,
            CASE WHEN NEW.status = 'success' THEN 1 ELSE 0 END,
            CASE WHEN NEW.status = 'failed' THEN 1 ELSE 0 END,
            NEW.execution_time_seconds,
            NEW.execution_time_seconds,
            NEW.execution_time_seconds
        FROM incidents i 
        WHERE i.id = NEW.incident_id
        ON CONFLICT (date_bucket, hour_bucket, service, action_type) 
        DO UPDATE SET
            total_attempts = resolution_metrics.total_attempts + 1,
            successful_attempts = resolution_metrics.successful_attempts + 
                CASE WHEN NEW.status = 'success' THEN 1 ELSE 0 END,
            failed_attempts = resolution_metrics.failed_attempts + 
                CASE WHEN NEW.status = 'failed' THEN 1 ELSE 0 END,
            avg_execution_time_seconds = (
                resolution_metrics.avg_execution_time_seconds * resolution_metrics.total_attempts + 
                COALESCE(NEW.execution_time_seconds, 0)
            ) / (resolution_metrics.total_attempts + 1),
            min_execution_time_seconds = LEAST(
                resolution_metrics.min_execution_time_seconds, 
                COALESCE(NEW.execution_time_seconds, resolution_metrics.min_execution_time_seconds)
            ),
            max_execution_time_seconds = GREATEST(
                resolution_metrics.max_execution_time_seconds, 
                COALESCE(NEW.execution_time_seconds, resolution_metrics.max_execution_time_seconds)
            ),
            updated_at = NOW();
            
        -- Update failure patterns if action failed
        IF NEW.status = 'failed' AND NEW.error_message IS NOT NULL THEN
            INSERT INTO failure_patterns (
                error_signature,
                action_type,
                service,
                occurrence_count,
                first_seen,
                last_seen
            )
            SELECT 
                LEFT(NEW.error_message, 500),
                NEW.action_type,
                i.service,
                1,
                NEW.completed_at,
                NEW.completed_at
            FROM incidents i 
            WHERE i.id = NEW.incident_id
            ON CONFLICT (error_signature, action_type, service) 
            DO UPDATE SET
                occurrence_count = failure_patterns.occurrence_count + 1,
                last_seen = NEW.completed_at,
                updated_at = NOW();
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update metrics
DROP TRIGGER IF EXISTS trigger_update_resolution_metrics ON action_attempts;
CREATE TRIGGER trigger_update_resolution_metrics
    AFTER UPDATE ON action_attempts
    FOR EACH ROW
    EXECUTE FUNCTION update_resolution_metrics();

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON action_attempts TO ai_agent_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON resolution_attempts TO ai_agent_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON action_logs TO ai_agent_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON resolution_metrics TO ai_agent_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON failure_patterns TO ai_agent_app;
-- GRANT SELECT ON action_success_rates TO ai_agent_app;
-- GRANT SELECT ON incident_resolution_performance TO ai_agent_app;
-- GRANT SELECT ON recent_failure_patterns TO ai_agent_app;
-- GRANT SELECT ON action_execution_timeline TO ai_agent_app;
