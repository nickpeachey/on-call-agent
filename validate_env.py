#!/usr/bin/env python3
"""
Environment validation script for AI On-Call Agent.
Run this to check if your environment is properly configured.
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Run: cp .env.example .env")
        return False
    print("✅ .env file found")
    return True

def check_required_vars():
    """Check required environment variables."""
    required_vars = {
        "DATABASE_PASSWORD": "Database password",
        "OPENAI_API_KEY": "OpenAI API key for AI features",
        "SECRET_KEY": "Application secret key (32+ chars)",
        "JWT_SECRET_KEY": "JWT secret key (32+ chars)",
        "AIRFLOW_BASE_URL": "Airflow server URL",
        "AIRFLOW_USERNAME": "Airflow username",
        "AIRFLOW_PASSWORD": "Airflow password",
        "ALERT_EMAIL_FROM": "Alert sender email",
    }
    
    missing = []
    weak_keys = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing.append(f"   {var}: {description}")
        elif var in ["SECRET_KEY", "JWT_SECRET_KEY"] and len(value) < 32:
            weak_keys.append(f"   {var}: Must be at least 32 characters")
        elif value in ["your_openai_api_key_here", "your_secret_key_here", "admin"]:
            missing.append(f"   {var}: Must be changed from default value")
    
    if missing:
        print("❌ Missing or invalid required variables:")
        for item in missing:
            print(item)
    
    if weak_keys:
        print("⚠️ Weak security keys:")
        for item in weak_keys:
            print(item)
    
    if not missing and not weak_keys:
        print("✅ All required variables configured")
        return True
    
    return False

def check_optional_vars():
    """Check optional but recommended variables."""
    optional_vars = {
        "SLACK_WEBHOOK_URL": "Slack notifications",
        "PAGERDUTY_API_KEY": "PagerDuty integration",
        "K8S_CONFIG_PATH": "Kubernetes integration",
    }
    
    configured = []
    for var, description in optional_vars.items():
        if os.getenv(var):
            configured.append(f"   {var}: {description}")
    
    if configured:
        print("✅ Optional integrations configured:")
        for item in configured:
            print(item)
    else:
        print("ℹ️ No optional integrations configured (this is fine)")

def main():
    """Main validation function."""
    print("🔍 AI On-Call Agent Environment Validation")
    print("=" * 50)
    
    # Load .env file if it exists
    if Path(".env").exists():
        from dotenv import load_dotenv
        load_dotenv()
    
    valid = True
    
    # Check .env file
    valid &= check_env_file()
    print()
    
    # Check required variables
    valid &= check_required_vars()
    print()
    
    # Check optional variables
    check_optional_vars()
    print()
    
    if valid:
        print("🎉 Environment validation passed!")
        print("   You can now run: python start.py")
    else:
        print("❌ Environment validation failed!")
        print("   Please fix the issues above and run this script again.")
        print("   See ENVIRONMENT_SETUP.md for detailed configuration guide.")
    
    return valid

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
