-- PostgreSQL initialization script for AI On-Call Agent
-- This script sets up the initial database schema

-- Create the main database if it doesn't exist
-- (Already created via POSTGRES_DB environment variable)

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create incidents table
CREATE TABLE IF NOT EXISTS incidents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    service VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    actions_taken TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Create actions table
CREATE TABLE IF NOT EXISTS actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    command TEXT,
    parameters JSONB DEFAULT '{}',
    success_rate DECIMAL(3,2) DEFAULT 0.0,
    execution_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- Create knowledge_base table
CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    problem_pattern TEXT NOT NULL,
    solution_steps TEXT[] NOT NULL,
    tags TEXT[] DEFAULT '{}',
    severity VARCHAR(20) NOT NULL,
    service VARCHAR(100),
    success_rate DECIMAL(3,2) DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- Create action_executions table
CREATE TABLE IF NOT EXISTS action_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID REFERENCES incidents(id) ON DELETE CASCADE,
    action_id UUID REFERENCES actions(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'success', 'failed', 'timeout')),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    output TEXT,
    error_message TEXT,
    execution_time_ms INTEGER,
    metadata JSONB DEFAULT '{}'
);

-- Create log_entries table for storing processed logs
CREATE TABLE IF NOT EXISTS log_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    level VARCHAR(20) NOT NULL,
    service VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    incident_id UUID REFERENCES incidents(id) ON DELETE SET NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
CREATE INDEX IF NOT EXISTS idx_incidents_service ON incidents(service);
CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at);
CREATE INDEX IF NOT EXISTS idx_incidents_tags ON incidents USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_actions_type ON actions(type);
CREATE INDEX IF NOT EXISTS idx_actions_active ON actions(is_active);

CREATE INDEX IF NOT EXISTS idx_kb_service ON knowledge_base(service);
CREATE INDEX IF NOT EXISTS idx_kb_severity ON knowledge_base(severity);
CREATE INDEX IF NOT EXISTS idx_kb_tags ON knowledge_base USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_kb_active ON knowledge_base(is_active);

CREATE INDEX IF NOT EXISTS idx_action_executions_incident ON action_executions(incident_id);
CREATE INDEX IF NOT EXISTS idx_action_executions_status ON action_executions(status);

CREATE INDEX IF NOT EXISTS idx_log_entries_timestamp ON log_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_log_entries_service ON log_entries(service);
CREATE INDEX IF NOT EXISTS idx_log_entries_level ON log_entries(level);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_incidents_updated_at BEFORE UPDATE ON incidents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_actions_updated_at BEFORE UPDATE ON actions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_base_updated_at BEFORE UPDATE ON knowledge_base
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some initial data
INSERT INTO actions (name, type, description, command) VALUES
('restart_service', 'service', 'Restart a systemd service', 'systemctl restart {service_name}'),
('scale_pods', 'kubernetes', 'Scale Kubernetes deployment', 'kubectl scale deployment {deployment} --replicas={replicas}'),
('clear_cache', 'redis', 'Clear Redis cache', 'redis-cli FLUSHDB'),
('restart_database', 'database', 'Restart database service', 'systemctl restart postgresql'),
('check_disk_space', 'system', 'Check available disk space', 'df -h'),
('kill_hung_process', 'system', 'Kill hung process by PID', 'kill -9 {pid}');

INSERT INTO knowledge_base (title, description, problem_pattern, solution_steps, tags, severity, service) VALUES
('Database Connection Timeout', 'Resolves database connection timeout issues', 'connection.*timeout.*database', 
 ARRAY['Check database connectivity', 'Restart connection pool', 'Scale database if needed'], 
 ARRAY['database', 'timeout', 'connection'], 'high', 'database'),
('Out of Memory Error', 'Handles Java/Spark out of memory errors', '(OutOfMemoryError|OOM|out of memory)', 
 ARRAY['Identify memory-intensive process', 'Scale resources', 'Restart application'], 
 ARRAY['memory', 'oom', 'java', 'spark'], 'high', 'spark-cluster'),
('Airflow DAG Failure', 'Resolves common Airflow DAG failures', 'airflow.*dag.*fail', 
 ARRAY['Check DAG dependencies', 'Restart Airflow scheduler', 'Clear failed task instances'], 
 ARRAY['airflow', 'dag', 'scheduler'], 'medium', 'data-pipeline');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO oncall_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO oncall_user;

-- Create initial admin user (for future authentication)
-- This would be replaced with proper user management
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'operator', 'user')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

GRANT ALL PRIVILEGES ON TABLE users TO oncall_user;
