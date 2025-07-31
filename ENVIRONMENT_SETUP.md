# AI On-Call Agent - Environment Setup Guide

## 🚀 Quick Start

### 1. Environment Configuration

The AI On-Call Agent requires several environment variables to be configured for external service integrations.

#### Step 1: Copy Environment Template
```bash
cp .env.example .env
```

#### Step 2: Configure Required Variables

Edit the `.env` file and update the following **critical** variables:

```bash
# Database Configuration (REQUIRED)
DATABASE_PASSWORD=your_secure_database_password

# OpenAI Configuration (REQUIRED for AI features)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Security Configuration (REQUIRED)
SECRET_KEY=your-256-bit-secret-key-here-must-be-at-least-32-characters
JWT_SECRET_KEY=your-jwt-secret-key-here-must-be-at-least-32-characters

# Email Configuration (REQUIRED for alerts)
ALERT_EMAIL_FROM=oncall-agent@yourcompany.com
ALERT_EMAIL_TO=["oncall@yourcompany.com","admin@yourcompany.com"]
```

### 2. External Service Configuration

#### Airflow Integration
```bash
# Update these if your Airflow instance differs
AIRFLOW_BASE_URL=http://your-airflow-host:8080
AIRFLOW_USERNAME=your_airflow_username
AIRFLOW_PASSWORD=your_airflow_password
```

#### Database Configuration
```bash
# PostgreSQL connection details
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=oncall_agent
DATABASE_USER=oncall_user
DATABASE_PASSWORD=your_secure_password
```

#### Spark Integration
```bash
# Spark cluster configuration
SPARK_MASTER_URL=spark://your-spark-master:7077
SPARK_HISTORY_SERVER_URL=http://your-spark-history:18080
SPARK_USERNAME=spark_user
SPARK_PASSWORD=spark_password
```

### 3. Optional Service Integrations

#### Slack Notifications
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SLACK_CHANNEL=#on-call-alerts
```

#### PagerDuty Integration
```bash
PAGERDUTY_API_KEY=your_pagerduty_api_key
PAGERDUTY_SERVICE_ID=your_service_id
PAGERDUTY_ENABLED=true
```

#### Kubernetes Integration
```bash
K8S_CONFIG_PATH=/home/user/.kube/config
K8S_NAMESPACE=production
K8S_CONTEXT=production-cluster
```

#### Docker Integration
```bash
DOCKER_HOST=unix:///var/run/docker.sock
DOCKER_API_VERSION=auto
```

## 🔧 Configuration Reference

### Complete Environment Variables

| Category | Variable | Required | Default | Description |
|----------|----------|----------|---------|-------------|
| **Database** | `DATABASE_HOST` | No | `localhost` | PostgreSQL host |
| | `DATABASE_PORT` | No | `5432` | PostgreSQL port |
| | `DATABASE_NAME` | No | `oncall_agent` | Database name |
| | `DATABASE_USER` | No | `oncall_user` | Database user |
| | `DATABASE_PASSWORD` | **Yes** | - | Database password |
| | `DATABASE_URL` | No | Auto-generated | Complete database URL |
| **Redis** | `REDIS_HOST` | No | `localhost` | Redis host |
| | `REDIS_PORT` | No | `6379` | Redis port |
| | `REDIS_PASSWORD` | No | - | Redis password |
| | `REDIS_DB` | No | `0` | Redis database number |
| **AI/OpenAI** | `OPENAI_API_KEY` | **Yes** | - | OpenAI API key |
| | `OPENAI_MODEL` | No | `gpt-4` | OpenAI model to use |
| | `OPENAI_MAX_TOKENS` | No | `2000` | Max tokens per request |
| | `OPENAI_TEMPERATURE` | No | `0.1` | Model temperature |
| **Airflow** | `AIRFLOW_BASE_URL` | **Yes** | - | Airflow web server URL |
| | `AIRFLOW_USERNAME` | **Yes** | - | Airflow username |
| | `AIRFLOW_PASSWORD` | **Yes** | - | Airflow password |
| | `AIRFLOW_API_VERSION` | No | `v1` | Airflow API version |
| | `AIRFLOW_TIMEOUT` | No | `30` | Request timeout (seconds) |
| **Spark** | `SPARK_MASTER_URL` | **Yes** | - | Spark master URL |
| | `SPARK_HISTORY_SERVER_URL` | **Yes** | - | Spark history server URL |
| | `SPARK_USERNAME` | No | - | Spark username |
| | `SPARK_PASSWORD` | No | - | Spark password |
| **Security** | `SECRET_KEY` | **Yes** | - | Application secret key |
| | `JWT_SECRET_KEY` | **Yes** | - | JWT signing key |
| | `JWT_ALGORITHM` | No | `HS256` | JWT algorithm |
| | `JWT_EXPIRATION_HOURS` | No | `24` | JWT expiration time |
| **Monitoring** | `LOG_PATHS` | No | `/var/log/app/*.log` | Log file paths to monitor |
| | `LOG_POLL_INTERVAL_SECONDS` | No | `5` | Log polling interval |
| | `CONFIDENCE_THRESHOLD` | No | `0.60` | AI confidence threshold |
| | `MAX_QUEUE_SIZE` | No | `1000` | Max incident queue size |
| **Alerts** | `ALERT_EMAIL_FROM` | **Yes** | - | Alert sender email |
| | `ALERT_EMAIL_TO` | **Yes** | - | Alert recipient emails |
| | `SLACK_WEBHOOK_URL` | No | - | Slack webhook URL |
| | `PAGERDUTY_API_KEY` | No | - | PagerDuty API key |

## 🏃‍♂️ Running the System

### Method 1: Using the Startup Script (Recommended)
```bash
python start.py
```

The startup script will:
- ✅ Validate your environment configuration
- 🔍 Test connections to external services
- 📋 Show configuration summary
- 🚀 Start all system components
- 📖 Display helpful getting started information

### Method 2: Direct Python Execution
```bash
# Set environment variables
export $(cat .env | xargs)

# Start the application
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Method 3: Using Docker
```bash
# Build the container
docker build -t ai-oncall-agent .

# Run with environment file
docker run --env-file .env -p 8000:8000 ai-oncall-agent
```

## 🔍 Configuration Validation

### Test Your Configuration
```bash
# Test configuration loading
python -c "from src.config import validate_configuration; validate_configuration()"

# View configuration summary
python -c "from src.config import get_settings; get_settings().log_configuration_summary()"
```

### Common Configuration Issues

#### 1. Database Connection Fails
```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Test connection manually
psql postgresql://oncall_user:password@localhost:5432/oncall_agent
```

#### 2. Redis Connection Fails
```bash
# Check if Redis is running
redis-cli ping

# Test connection with password
redis-cli -h localhost -p 6379 -a your_password ping
```

#### 3. Airflow Connection Fails
```bash
# Test Airflow API
curl -u username:password http://localhost:8080/api/v1/health
```

#### 4. OpenAI API Issues
```bash
# Test OpenAI API key
curl -H "Authorization: Bearer sk-your-key" https://api.openai.com/v1/models
```

## 📁 Log Files and Monitoring

### Log Locations
- **Application Logs**: `logs/app.log`
- **AI Decision Logs**: `logs/ai_decisions.log`  
- **Action Execution Logs**: `logs/actions.log`
- **Startup Logs**: `logs/startup.log`

### Monitoring Configuration
```bash
# Configure which log files to monitor
LOG_PATHS=/var/log/app/*.log,/var/log/airflow/*.log,/var/log/spark/*.log

# Set polling interval
LOG_POLL_INTERVAL_SECONDS=5
```

## 🚨 Security Best Practices

### 1. Generate Secure Keys
```bash
# Generate a secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate a secure JWT key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Database Security
- Use a strong, unique password for the database user
- Limit database user permissions to only necessary tables
- Use SSL connections in production

### 3. API Keys
- Store API keys securely (consider using a secret management service)
- Rotate API keys regularly
- Use environment-specific keys (dev/staging/prod)

### 4. Network Security
- Use HTTPS in production
- Implement proper firewall rules
- Use VPN or private networks for service-to-service communication

## 🐛 Troubleshooting

### Environment Loading Issues
```bash
# Check if .env file exists and is readable
ls -la .env

# Verify environment variables are loaded
python -c "import os; print(os.getenv('DATABASE_PASSWORD', 'NOT_SET'))"
```

### Service Connection Issues
```bash
# Check service status
systemctl status postgresql
systemctl status redis
systemctl status docker

# Check port availability
netstat -tuln | grep :5432  # PostgreSQL
netstat -tuln | grep :6379  # Redis
netstat -tuln | grep :8080  # Airflow
```

### Permission Issues
```bash
# Check file permissions
ls -la .env
ls -la logs/

# Fix log directory permissions
chmod 755 logs/
chmod 644 logs/*.log
```

## 📞 Support

For additional help:
1. Check the application logs in `logs/`
2. Review the configuration summary in startup output
3. Verify all external services are running and accessible
4. Ensure all required environment variables are set

The startup script provides detailed validation and will guide you through any configuration issues.
