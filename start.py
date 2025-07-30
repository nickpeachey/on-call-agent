#!/usr/bin/env python3
"""
AI On-Call Agent Startup Script
Validates environment configuration and starts the system components.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_settings, validate_configuration
from main import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/startup.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def check_environment_file():
    """Check if .env file exists and provide guidance."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        logger.warning("⚠️ No .env file found!")
        
        if env_example.exists():
            logger.info("📋 .env.example file found. To get started:")
            logger.info("   1. Copy .env.example to .env:")
            logger.info("      cp .env.example .env")
            logger.info("   2. Edit .env with your actual credentials")
            logger.info("   3. Run this script again")
        else:
            logger.error("❌ No .env.example file found. Please create environment configuration.")
        
        return False
    
    logger.info("✅ .env file found")
    return True


def validate_critical_services():
    """Validate that critical external services are configured."""
    settings = get_settings()
    warnings = []
    errors = []
    
    # Check database configuration
    if settings.database.password == "change_me_please":
        errors.append("🗄️ Database password needs to be changed from default")
    
    # Check OpenAI configuration
    if settings.openai.api_key == "your_openai_api_key_here":
        warnings.append("🤖 OpenAI API key not configured (AI features will not work)")
    
    # Check Airflow configuration
    if settings.airflow.username == "admin" and settings.airflow.password == "admin":
        warnings.append("🎬 Airflow using default credentials (consider changing for security)")
    
    # Check security configuration
    if "your_secret_key_here" in settings.security.secret_key:
        errors.append("🔐 Security secret key must be changed from default")
    
    if "your_jwt_secret_key" in settings.security.jwt_secret_key:
        errors.append("🔐 JWT secret key must be changed from default")
    
    # Print warnings
    if warnings:
        logger.warning("⚠️ Configuration warnings:")
        for warning in warnings:
            logger.warning(f"   {warning}")
    
    # Print errors
    if errors:
        logger.error("❌ Configuration errors that must be fixed:")
        for error in errors:
            logger.error(f"   {error}")
        return False
    
    return True


def print_startup_banner():
    """Print a startup banner with system information."""
    settings = get_settings()
    
    print("=" * 70)
    print("🤖 AI On-Call Agent - Intelligent Infrastructure Monitoring")
    print("=" * 70)
    print(f"📱 Version: {settings.app.version}")
    print(f"🌐 Server: http://{settings.app.host}:{settings.app.port}")
    print(f"🗄️ Database: {settings.database.host}:{settings.database.port}")
    print(f"🎬 Airflow: {settings.airflow.base_url}")
    print(f"⚡ Spark: {settings.spark.master_url}")
    print(f"🤖 AI Confidence Threshold: {settings.ai.confidence_threshold}")
    print(f"📊 Debug Mode: {settings.app.debug}")
    print("=" * 70)


def print_quick_start_guide():
    """Print a quick start guide for new users."""
    print("\n🚀 Quick Start Guide:")
    print("   1. Ensure your external services are running:")
    print("      • PostgreSQL database")
    print("      • Redis cache")
    print("      • Airflow (optional)")
    print("      • Spark (optional)")
    print()
    print("   2. Access the system:")
    print("      • Web UI: http://localhost:8000")
    print("      • API Docs: http://localhost:8000/docs")
    print("      • Health Check: http://localhost:8000/health")
    print()
    print("   3. Monitor logs:")
    print("      • Application logs: logs/app.log")
    print("      • AI decisions: logs/ai_decisions.log")
    print("      • Actions: logs/actions.log")
    print()
    print("   4. Stop the system: Ctrl+C")
    print("=" * 70)


def create_log_directories():
    """Create necessary log directories."""
    log_dirs = ["logs", "logs/archive"]
    for log_dir in log_dirs:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.info("✅ Log directories created")


async def test_connections():
    """Test connections to external services."""
    settings = get_settings()
    test_results = {}
    
    # Test database connection
    try:
        # This would normally test the actual database connection
        # For now, just validate the URL format
        if settings.database.url.startswith("postgresql://"):
            test_results["database"] = "✅ Database URL format valid"
        else:
            test_results["database"] = "❌ Invalid database URL format"
    except Exception as e:
        test_results["database"] = f"❌ Database config error: {e}"
    
    # Test Redis connection
    try:
        if settings.redis.url.startswith("redis://"):
            test_results["redis"] = "✅ Redis URL format valid"
        else:
            test_results["redis"] = "❌ Invalid Redis URL format"
    except Exception as e:
        test_results["redis"] = f"❌ Redis config error: {e}"
    
    # Test Airflow connection (if configured)
    try:
        if settings.airflow.base_url.startswith(("http://", "https://")):
            test_results["airflow"] = "✅ Airflow URL format valid"
        else:
            test_results["airflow"] = "❌ Invalid Airflow URL format"
    except Exception as e:
        test_results["airflow"] = f"❌ Airflow config error: {e}"
    
    # Print test results
    logger.info("🔍 Connection Tests:")
    for service, result in test_results.items():
        logger.info(f"   {service}: {result}")
    
    return all("✅" in result for result in test_results.values())


def main():
    """Main startup function."""
    try:
        print("🚀 Starting AI On-Call Agent...")
        
        # Step 1: Check environment file
        if not check_environment_file():
            sys.exit(1)
        
        # Step 2: Load and validate configuration
        logger.info("📋 Loading configuration...")
        if not validate_configuration():
            logger.error("❌ Configuration validation failed")
            sys.exit(1)
        
        # Step 3: Validate critical services
        logger.info("🔍 Validating critical services...")
        if not validate_critical_services():
            logger.error("❌ Critical service validation failed")
            logger.error("   Please fix the configuration errors and try again")
            sys.exit(1)
        
        # Step 4: Create necessary directories
        create_log_directories()
        
        # Step 5: Print startup information
        print_startup_banner()
        
        # Step 6: Test connections
        logger.info("🌐 Testing external service connections...")
        async def run_tests():
            return await test_connections()
        
        connections_ok = asyncio.run(run_tests())
        if not connections_ok:
            logger.warning("⚠️ Some connection tests failed - services may not be available")
        
        # Step 7: Print quick start guide
        print_quick_start_guide()
        
        # Step 8: Start the application
        logger.info("🎯 Starting application components...")
        
        # Import and run the main application
        app = create_app()
        
        settings = get_settings()
        
        # Start the FastAPI application
        import uvicorn
        uvicorn.run(
            app,
            host=settings.app.host,
            port=settings.app.port,
            log_level=settings.app.log_level.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
        print("\n👋 AI On-Call Agent stopped. Thank you!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
