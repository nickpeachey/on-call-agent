"""Core configuration management."""

from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    debug: bool = False
    log_level: str = "INFO"
    quiet_mode: bool = True  # When True, reduces verbose logging for production use
    secret_key: str = Field(..., min_length=32)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Database
    database_url: str = Field(..., description="PostgreSQL connection URL")
    database_pool_size: int = 20
    database_max_overflow: int = 0
    
    # Redis
    redis_url: str = Field(..., description="Redis connection URL")
    redis_password: Optional[str] = None
    redis_ttl: int = 3600
    
    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = "gpt-4"
    openai_max_tokens: int = 2000
    
    # Monitoring Sources
    elasticsearch_urls: List[str] = Field(default=["http://localhost:9200"])
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[str] = None
    splunk_host: str = "localhost"
    splunk_port: int = 8089
    splunk_username: str = "admin"
    splunk_password: str = "changeme"
    
    # Airflow Integration
    airflow_base_url: str = "http://localhost:8080"
    airflow_username: str = "admin"
    airflow_password: str = "admin"
    airflow_api_version: str = "v1"
    
    # Spark Integration
    spark_history_server_url: str = "http://localhost:18080"
    spark_master_url: str = "spark://localhost:7077"
    
    # Action Engine
    max_concurrent_actions: int = 5
    action_timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: int = 10
    
    # Alerts
    alert_email_enabled: bool = True
    alert_email_smtp_host: str = "smtp.gmail.com"
    alert_email_smtp_port: int = 587
    alert_email_username: str = ""
    alert_email_password: str = ""
    alert_email_from: str = ""
    alert_email_to: List[str] = Field(default=[])
    
    # Slack
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#oncall-alerts"
    
    # Security
    jwt_secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Rate Limiting
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 20
    
    # Metrics
    prometheus_port: int = 9090
    metrics_enabled: bool = True
    
    @validator("elasticsearch_urls", pre=True)
    def parse_elasticsearch_urls(cls, v):
        """Parse elasticsearch URLs from string or list."""
        if isinstance(v, str):
            # Handle JSON string format
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v
    
    @validator("alert_email_to", pre=True)
    def parse_email_list(cls, v):
        """Parse email list from string or list."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [email.strip() for email in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
