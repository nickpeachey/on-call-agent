"""
Configuration management for AI On-Call Agent.
Loads and validates environment variables with secure defaults.
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """Get environment variable with optional default and validation."""
    value = os.getenv(key, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_list(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
    """Get list environment variable."""
    if default is None:
        default = []
    value = os.getenv(key)
    if not value:
        return default
    return [item.strip() for item in value.split(separator)]


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    
    host: str = field(default_factory=lambda: get_env("DATABASE_HOST", "localhost"))
    port: int = field(default_factory=lambda: get_env_int("DATABASE_PORT", 5432))
    name: str = field(default_factory=lambda: get_env("DATABASE_NAME", "oncall_agent"))
    user: str = field(default_factory=lambda: get_env("DATABASE_USER", "oncall_user"))
    password: str = field(default_factory=lambda: get_env("DATABASE_PASSWORD", "change_me_please"))
    pool_size: int = field(default_factory=lambda: get_env_int("DATABASE_POOL_SIZE", 20))
    max_overflow: int = field(default_factory=lambda: get_env_int("DATABASE_MAX_OVERFLOW", 0))
    
    @property
    def url(self) -> str:
        """Build database URL from components."""
        env_url = get_env("DATABASE_URL")
        if env_url:
            return env_url
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RedisConfig:
    """Redis configuration for caching and queue management."""
    
    host: str = field(default_factory=lambda: get_env("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: get_env_int("REDIS_PORT", 6379))
    password: Optional[str] = field(default_factory=lambda: get_env("REDIS_PASSWORD"))
    db: int = field(default_factory=lambda: get_env_int("REDIS_DB", 0))
    ttl: int = field(default_factory=lambda: get_env_int("REDIS_TTL", 3600))
    
    @property
    def url(self) -> str:
        """Build Redis URL from components."""
        env_url = get_env("REDIS_URL")
        if env_url:
            return env_url
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class AirflowConfig:
    """Airflow integration configuration."""
    
    base_url: str = field(default_factory=lambda: get_env("AIRFLOW_BASE_URL", "http://localhost:8080"))
    username: str = field(default_factory=lambda: get_env("AIRFLOW_USERNAME", "admin"))
    password: str = field(default_factory=lambda: get_env("AIRFLOW_PASSWORD", "admin"))
    api_version: str = field(default_factory=lambda: get_env("AIRFLOW_API_VERSION", "v1"))
    timeout: int = field(default_factory=lambda: get_env_int("AIRFLOW_TIMEOUT", 30))
    verify_ssl: bool = field(default_factory=lambda: get_env_bool("AIRFLOW_VERIFY_SSL", False))
    connection_id: str = field(default_factory=lambda: get_env("AIRFLOW_CONNECTION_ID", "airflow_default"))
    dag_timeout: int = field(default_factory=lambda: get_env_int("AIRFLOW_DAG_TIMEOUT", 600))
    
    def __post_init__(self):
        """Validate Airflow configuration."""
        if not self.base_url.startswith(('http://', 'https://')):
            raise ValueError('Airflow base URL must start with http:// or https://')
        self.base_url = self.base_url.rstrip('/')


@dataclass
class SparkConfig:
    """Spark integration configuration."""
    
    history_server_url: str = field(default_factory=lambda: get_env("SPARK_HISTORY_SERVER_URL", "http://localhost:18080"))
    master_url: str = field(default_factory=lambda: get_env("SPARK_MASTER_URL", "spark://localhost:7077"))
    username: Optional[str] = field(default_factory=lambda: get_env("SPARK_USERNAME"))
    password: Optional[str] = field(default_factory=lambda: get_env("SPARK_PASSWORD"))
    web_ui_port: int = field(default_factory=lambda: get_env_int("SPARK_APPLICATION_WEB_UI_PORT", 4040))


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    
    api_key: str = field(default_factory=lambda: get_env("OPENAI_API_KEY", "your_openai_api_key_here"))
    model: str = field(default_factory=lambda: get_env("OPENAI_MODEL", "gpt-4"))
    max_tokens: int = field(default_factory=lambda: get_env_int("OPENAI_MAX_TOKENS", 2000))
    temperature: float = field(default_factory=lambda: get_env_float("OPENAI_TEMPERATURE", 0.1))
    
    def __post_init__(self):
        """Validate OpenAI configuration."""
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            logger.warning("⚠️ OpenAI API key not properly configured")


@dataclass
class MonitoringConfig:
    """Log monitoring and source configuration."""
    
    elasticsearch_urls: List[str] = field(default_factory=lambda: get_env_list("ELASTICSEARCH_URLS", ["http://localhost:9200"]))
    elasticsearch_username: Optional[str] = field(default_factory=lambda: get_env("ELASTICSEARCH_USERNAME"))
    elasticsearch_password: Optional[str] = field(default_factory=lambda: get_env("ELASTICSEARCH_PASSWORD"))
    
    splunk_host: Optional[str] = field(default_factory=lambda: get_env("SPLUNK_HOST"))
    splunk_port: int = field(default_factory=lambda: get_env_int("SPLUNK_PORT", 8089))
    splunk_username: Optional[str] = field(default_factory=lambda: get_env("SPLUNK_USERNAME"))
    splunk_password: Optional[str] = field(default_factory=lambda: get_env("SPLUNK_PASSWORD"))
    
    log_paths: str = field(default_factory=lambda: get_env("LOG_PATHS", "/var/log/app/*.log"))
    log_poll_interval: int = field(default_factory=lambda: get_env_int("LOG_POLL_INTERVAL_SECONDS", 5))
    log_buffer_size: int = field(default_factory=lambda: get_env_int("LOG_BUFFER_SIZE", 1000))
    
    @property
    def log_path_list(self) -> List[str]:
        """Convert comma-separated log paths to list."""
        return [path.strip() for path in self.log_paths.split(',')]


@dataclass
class AIConfig:
    """AI Decision Engine configuration."""
    
    confidence_threshold: float = field(default_factory=lambda: get_env_float("CONFIDENCE_THRESHOLD", 0.60))
    max_queue_size: int = field(default_factory=lambda: get_env_int("MAX_QUEUE_SIZE", 1000))
    processing_timeout: int = field(default_factory=lambda: get_env_int("PROCESSING_TIMEOUT_SECONDS", 300))
    decision_loop_interval: int = field(default_factory=lambda: get_env_int("AI_DECISION_LOOP_INTERVAL_SECONDS", 2))
    
    def __post_init__(self):
        """Validate AI configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')


@dataclass
class ActionEngineConfig:
    """Action execution configuration."""
    
    max_concurrent_actions: int = field(default_factory=lambda: get_env_int("MAX_CONCURRENT_ACTIONS", 5))
    action_timeout: int = field(default_factory=lambda: get_env_int("ACTION_TIMEOUT_SECONDS", 300))
    retry_attempts: int = field(default_factory=lambda: get_env_int("RETRY_ATTEMPTS", 3))
    retry_delay: int = field(default_factory=lambda: get_env_int("RETRY_DELAY_SECONDS", 10))


@dataclass
class KubernetesConfig:
    """Kubernetes integration configuration."""
    
    config_path: Optional[str] = field(default_factory=lambda: get_env("K8S_CONFIG_PATH"))
    namespace: str = field(default_factory=lambda: get_env("K8S_NAMESPACE", "default"))
    context: Optional[str] = field(default_factory=lambda: get_env("K8S_CONTEXT"))


@dataclass
class DockerConfig:
    """Docker integration configuration."""
    
    host: str = field(default_factory=lambda: get_env("DOCKER_HOST", "unix:///var/run/docker.sock"))
    api_version: str = field(default_factory=lambda: get_env("DOCKER_API_VERSION", "auto"))


@dataclass
class AlertConfig:
    """Alert and notification configuration."""
    
    # Email settings
    email_enabled: bool = field(default_factory=lambda: get_env_bool("ALERT_EMAIL_ENABLED", True))
    email_smtp_host: str = field(default_factory=lambda: get_env("ALERT_EMAIL_SMTP_HOST", "smtp.gmail.com"))
    email_smtp_port: int = field(default_factory=lambda: get_env_int("ALERT_EMAIL_SMTP_PORT", 587))
    email_username: Optional[str] = field(default_factory=lambda: get_env("ALERT_EMAIL_USERNAME"))
    email_password: Optional[str] = field(default_factory=lambda: get_env("ALERT_EMAIL_PASSWORD"))
    email_from: str = field(default_factory=lambda: get_env("ALERT_EMAIL_FROM", "oncall-agent@yourcompany.com"))
    email_to: List[str] = field(default_factory=lambda: get_env_list("ALERT_EMAIL_TO", ["oncall@yourcompany.com"]))
    
    # Slack settings
    slack_webhook_url: Optional[str] = field(default_factory=lambda: get_env("SLACK_WEBHOOK_URL"))
    slack_channel: str = field(default_factory=lambda: get_env("SLACK_CHANNEL", "#oncall-alerts"))
    
    # PagerDuty settings
    pagerduty_api_key: Optional[str] = field(default_factory=lambda: get_env("PAGERDUTY_API_KEY"))
    pagerduty_service_id: Optional[str] = field(default_factory=lambda: get_env("PAGERDUTY_SERVICE_ID"))
    pagerduty_enabled: bool = field(default_factory=lambda: get_env_bool("PAGERDUTY_ENABLED", False))


@dataclass
class SecurityConfig:
    """Security and authentication configuration."""
    
    secret_key: str = field(default_factory=lambda: get_env("SECRET_KEY", "your_secret_key_here_must_be_32_chars_minimum"))
    jwt_secret_key: str = field(default_factory=lambda: get_env("JWT_SECRET_KEY", "your_jwt_secret_key_here_must_be_32_chars"))
    jwt_algorithm: str = field(default_factory=lambda: get_env("JWT_ALGORITHM", "HS256"))
    jwt_expiration_hours: int = field(default_factory=lambda: get_env_int("JWT_EXPIRATION_HOURS", 24))
    
    def __post_init__(self):
        """Validate security configuration."""
        if len(self.secret_key) < 32:
            logger.warning("⚠️ Secret key should be at least 32 characters long")
        if len(self.jwt_secret_key) < 32:
            logger.warning("⚠️ JWT secret key should be at least 32 characters long")


@dataclass
class AppConfig:
    """Main application configuration."""
    
    name: str = field(default_factory=lambda: get_env("APP_NAME", "AI On-Call Agent"))
    version: str = field(default_factory=lambda: get_env("APP_VERSION", "1.0.0"))
    debug: bool = field(default_factory=lambda: get_env_bool("DEBUG", False))
    log_level: str = field(default_factory=lambda: get_env("LOG_LEVEL", "INFO"))
    host: str = field(default_factory=lambda: get_env("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: get_env_int("API_PORT", 8000))
    
    # Rate limiting
    rate_limit_per_minute: int = field(default_factory=lambda: get_env_int("RATE_LIMIT_PER_MINUTE", 100))
    rate_limit_burst: int = field(default_factory=lambda: get_env_int("RATE_LIMIT_BURST", 20))
    
    # Metrics
    prometheus_port: int = field(default_factory=lambda: get_env_int("PROMETHEUS_PORT", 9090))
    metrics_enabled: bool = field(default_factory=lambda: get_env_bool("METRICS_ENABLED", True))


@dataclass
class Settings:
    """Main configuration container."""
    
    app: AppConfig = field(default_factory=AppConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    airflow: AirflowConfig = field(default_factory=AirflowConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    actions: ActionEngineConfig = field(default_factory=ActionEngineConfig)
    kubernetes: KubernetesConfig = field(default_factory=KubernetesConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    def validate_required_configs(self) -> Dict[str, List[str]]:
        """Validate that all required configurations are properly set."""
        errors = {}
        
        # Check critical configurations
        try:
            # Test database connection URL
            if not self.database.url:
                errors.setdefault("database", []).append("Database URL cannot be built")
            
            # Test Redis connection URL
            if not self.redis.url:
                errors.setdefault("redis", []).append("Redis URL cannot be built")
            
            # Check OpenAI configuration
            if self.openai.api_key == "your_openai_api_key_here":
                errors.setdefault("openai", []).append("OpenAI API key must be configured")
                
        except Exception as e:
            errors.setdefault("validation", []).append(f"Configuration validation error: {str(e)}")
        
        return errors
    
    def log_configuration_summary(self):
        """Log a summary of the current configuration (without sensitive data)."""
        logger.info("🔧 AI On-Call Agent Configuration Summary:")
        logger.info(f"📱 App: {self.app.name} v{self.app.version}")
        logger.info(f"🌐 Server: {self.app.host}:{self.app.port}")
        logger.info(f"🗄️ Database: {self.database.host}:{self.database.port}/{self.database.name}")
        logger.info(f"📊 Redis: {self.redis.host}:{self.redis.port}/{self.redis.db}")
        logger.info(f"🎬 Airflow: {self.airflow.base_url}")
        logger.info(f"⚡ Spark: {self.spark.master_url}")
        logger.info(f"🤖 AI Confidence Threshold: {self.ai.confidence_threshold}")
        logger.info(f"📁 Log Paths: {len(self.monitoring.log_path_list)} paths")
        logger.info(f"🔧 Max Concurrent Actions: {self.actions.max_concurrent_actions}")
        logger.info(f"📈 Metrics Enabled: {self.app.metrics_enabled}")


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings()


def validate_configuration() -> bool:
    """Validate the configuration and log any errors."""
    try:
        settings = load_settings()
        errors = settings.validate_required_configs()
        
        if errors:
            logger.error("❌ Configuration validation failed:")
            for section, error_list in errors.items():
                for error in error_list:
                    logger.error(f"  {section}: {error}")
            return False
        
        logger.info("✅ Configuration validation passed")
        settings.log_configuration_summary()
        return True
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")
        return False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


if __name__ == "__main__":
    # Test configuration loading
    print("Loading configuration...")
    try:
        valid = validate_configuration()
        if valid:
            print("✅ Configuration loaded successfully")
        else:
            print("❌ Configuration validation failed")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
