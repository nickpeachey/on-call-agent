#!/usr/bin/env python3
"""
AI On-Call Agent Setup Script

This script will install all dependencies and configure the system for:
1. Local development mode
2. Production deployment mode

Usage:
    python3 setup.py --mode dev     # Local development setup
    python3 setup.py --mode prod    # Production setup
    python3 setup.py --help         # Show help
"""

import sys
import os
import subprocess
import json
import argparse
from pathlib import Path
import platform
import shutil

class AIOnCallAgentSetup:
    """Setup manager for AI On-Call Agent system."""
    
    def __init__(self, mode="dev"):
        self.mode = mode
        self.project_root = Path(__file__).parent
        self.python_executable = sys.executable
        self.is_windows = platform.system() == "Windows"
        self.is_macos = platform.system() == "Darwin"
        self.is_linux = platform.system() == "Linux"
        
        print(f"🚀 AI On-Call Agent Setup - {mode.upper()} Mode")
        print("=" * 60)
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version}")
        print(f"Project Root: {self.project_root}")
        print()
    
    def check_prerequisites(self):
        """Check system prerequisites."""
        print("🔍 Checking Prerequisites...")
        
        prerequisites = {
            "python3": "Python 3.8+",
            "pip3": "pip package manager",
            "git": "Git version control"
        }
        
        missing = []
        
        for cmd, description in prerequisites.items():
            if shutil.which(cmd):
                print(f"✅ {description} found")
            else:
                print(f"❌ {description} not found")
                missing.append(cmd)
        
        if missing:
            print(f"\n❌ Missing prerequisites: {', '.join(missing)}")
            print("Please install missing components and run setup again.")
            return False
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        
        print("✅ All prerequisites satisfied")
        return True
    
    def create_project_structure(self):
        """Create necessary project directories."""
        print("\n📁 Creating Project Structure...")
        
        directories = [
            "src",
            "config", 
            "data",
            "logs",
            "docs",
            "reports",
            "tests",
            "scripts",
            ".venv" if self.mode == "dev" else "venv"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created: {directory}/")
        
        print("✅ Project structure created")
    
    def install_python_dependencies(self):
        """Install Python dependencies."""
        print("\n📦 Installing Python Dependencies...")
        
        # Core dependencies for all modes
        core_deps = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
            "requests>=2.31.0",
            "redis>=5.0.0",
            "psycopg2-binary>=2.9.0",
            "sqlalchemy>=2.0.0",
            "alembic>=1.13.0",
            "celery>=5.3.0",
            "python-multipart>=0.0.6",
            "python-jose[cryptography]>=3.3.0",
            "passlib[bcrypt]>=1.7.4",
            "python-dotenv>=1.0.0",
            "pyyaml>=6.0.1",
            "jinja2>=3.1.0",
            "aiofiles>=23.2.1"
        ]
        
        # AI/ML dependencies
        ai_deps = [
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "pandas>=2.1.0",
            "openai>=1.3.0",
            "langchain>=0.1.0",
            "transformers>=4.35.0"
        ]
        
        # CLI dependencies
        cli_deps = [
            "typer>=0.9.0",
            "rich>=13.7.0",
            "click>=8.1.0"
        ]
        
        # Development dependencies
        dev_deps = [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
            "jupyter>=1.0.0",
            "ipython>=8.17.0"
        ]
        
        # Documentation dependencies
        doc_deps = [
            "markdown2>=2.4.0",
            "weasyprint>=60.0",
            "reportlab>=4.0.0"
        ]
        
        # Production dependencies
        prod_deps = [
            "gunicorn>=21.2.0",
            "docker>=6.1.0",
            "kubernetes>=28.1.0",
            "prometheus-client>=0.19.0",
            "sentry-sdk[fastapi]>=1.38.0"
        ]
        
        # Select dependencies based on mode
        dependencies = core_deps + ai_deps + cli_deps + doc_deps
        
        if self.mode == "dev":
            dependencies.extend(dev_deps)
            print("📚 Including development dependencies...")
        elif self.mode == "prod":
            dependencies.extend(prod_deps)
            print("🏭 Including production dependencies...")
        
        # Install dependencies
        for dep in dependencies:
            try:
                print(f"📦 Installing {dep}...")
                result = subprocess.run([
                    self.python_executable, "-m", "pip", "install", dep
                ], capture_output=True, text=True, check=True)
                print(f"✅ Installed {dep}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Warning: Failed to install {dep}")
                print(f"   Error: {e.stderr.strip()}")
        
        print("✅ Python dependencies installation completed")
    
    def create_configuration_files(self):
        """Create configuration files."""
        print("\n⚙️  Creating Configuration Files...")
        
        # Create main config file
        config = {
            "mode": self.mode,
            "app": {
                "name": "AI On-Call Agent",
                "version": "1.0.0",
                "debug": self.mode == "dev",
                "host": "localhost" if self.mode == "dev" else "0.0.0.0",
                "port": 8000
            },
            "database": {
                "url": "sqlite:///./data/oncall_agent.db" if self.mode == "dev" else "postgresql://user:pass@localhost/oncall_agent",
                "echo": self.mode == "dev"
            },
            "redis": {
                "url": "redis://localhost:6379/0",
                "decode_responses": True
            },
            "logging": {
                "level": "DEBUG" if self.mode == "dev" else "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/oncall_agent.log"
            },
            "ai": {
                "openai_api_key": "your-openai-api-key-here",
                "model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.3
            },
            "monitoring": {
                "poll_interval": 60,
                "alert_threshold": 5,
                "max_retries": 3
            }
        }
        
        config_file = self.project_root / "config" / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created: {config_file}")
        
        # Create environment file
        env_content = f"""# AI On-Call Agent Environment Configuration
# Mode: {self.mode}

# Application Settings
APP_NAME=AI On-Call Agent
APP_VERSION=1.0.0
DEBUG={"true" if self.mode == "dev" else "false"}
HOST={"localhost" if self.mode == "dev" else "0.0.0.0"}
PORT=8000

# Database Configuration
DATABASE_URL={"sqlite:///./data/oncall_agent.db" if self.mode == "dev" else "postgresql://user:pass@localhost/oncall_agent"}

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# AI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Logging Configuration
LOG_LEVEL={"DEBUG" if self.mode == "dev" else "INFO"}
LOG_FILE=logs/oncall_agent.log

# Security
SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Monitoring
POLL_INTERVAL=60
ALERT_THRESHOLD=5
MAX_RETRIES=3
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created: {env_file}")
        
        # Create requirements.txt
        requirements_content = """# AI On-Call Agent Dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
requests>=2.31.0
redis>=5.0.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
alembic>=1.13.0
celery>=5.3.0
python-multipart>=0.0.6
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-dotenv>=1.0.0
pyyaml>=6.0.1
jinja2>=3.1.0
aiofiles>=23.2.1
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.1.0
openai>=1.3.0
langchain>=0.1.0
transformers>=4.35.0
typer>=0.9.0
rich>=13.7.0
click>=8.1.0
markdown2>=2.4.0
weasyprint>=60.0
reportlab>=4.0.0
"""
        
        if self.mode == "dev":
            requirements_content += """
# Development Dependencies
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.11.0
flake8>=6.1.0
mypy>=1.7.0
pre-commit>=3.5.0
jupyter>=1.0.0
ipython>=8.17.0
"""
        elif self.mode == "prod":
            requirements_content += """
# Production Dependencies
gunicorn>=21.2.0
docker>=6.1.0
kubernetes>=28.1.0
prometheus-client>=0.19.0
sentry-sdk[fastapi]>=1.38.0
"""
        
        requirements_file = self.project_root / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        print(f"✅ Created: {requirements_file}")
        
        print("✅ Configuration files created")
    
    def setup_development_tools(self):
        """Setup development tools (dev mode only)."""
        if self.mode != "dev":
            return
        
        print("\n🛠️  Setting Up Development Tools...")
        
        # Create pre-commit config
        precommit_content = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
"""
        
        precommit_file = self.project_root / ".pre-commit-config.yaml"
        with open(precommit_file, 'w') as f:
            f.write(precommit_content)
        print(f"✅ Created: {precommit_file}")
        
        # Create pytest config
        pytest_content = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --tb=short
"""
        
        pytest_file = self.project_root / "pytest.ini"
        with open(pytest_file, 'w') as f:
            f.write(pytest_content)
        print(f"✅ Created: {pytest_file}")
        
        # Install pre-commit hooks
        try:
            subprocess.run([
                self.python_executable, "-m", "pre_commit", "install"
            ], check=True, capture_output=True)
            print("✅ Pre-commit hooks installed")
        except subprocess.CalledProcessError:
            print("⚠️  Warning: Failed to install pre-commit hooks")
        
        print("✅ Development tools setup completed")
    
    def setup_production_tools(self):
        """Setup production tools (prod mode only)."""
        if self.mode != "prod":
            return
        
        print("\n🏭 Setting Up Production Tools...")
        
        # Create Docker files
        dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash oncall
RUN chown -R oncall:oncall /app
USER oncall

# Expose port
EXPOSE 8000

# Start application
CMD ["gunicorn", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "src.main:app"]
"""
        
        dockerfile = self.project_root / "Dockerfile"
        with open(dockerfile, 'w') as f:
            f.write(dockerfile_content)
        print(f"✅ Created: {dockerfile}")
        
        # Create docker-compose.yml
        docker_compose_content = """version: '3.8'

services:
  oncall-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/oncall_agent
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=oncall_agent
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  postgres_data:
  redis_data:
"""
        
        docker_compose_file = self.project_root / "docker-compose.yml"
        with open(docker_compose_file, 'w') as f:
            f.write(docker_compose_content)
        print(f"✅ Created: {docker_compose_file}")
        
        # Create Kubernetes deployment
        k8s_content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: oncall-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oncall-agent
  template:
    metadata:
      labels:
        app: oncall-agent
    spec:
      containers:
      - name: oncall-agent
        image: oncall-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://postgres:password@postgres:5432/oncall_agent"
        - name: REDIS_URL
          value: "redis://redis:6379/0"
---
apiVersion: v1
kind: Service
metadata:
  name: oncall-agent-service
spec:
  selector:
    app: oncall-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
"""
        
        k8s_dir = self.project_root / "k8s"
        k8s_dir.mkdir(exist_ok=True)
        k8s_file = k8s_dir / "deployment.yaml"
        with open(k8s_file, 'w') as f:
            f.write(k8s_content)
        print(f"✅ Created: {k8s_file}")
        
        print("✅ Production tools setup completed")
    
    def create_startup_scripts(self):
        """Create startup scripts."""
        print("\n🚀 Creating Startup Scripts...")
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Development startup script
        dev_script_content = """#!/bin/bash
# Development startup script

echo "🚀 Starting AI On-Call Agent (Development Mode)"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Start Redis if available
if command -v redis-server &> /dev/null; then
    redis-server --daemonize yes
    echo "✅ Redis started"
fi

# Initialize database
python3 -c "
import sys
sys.path.insert(0, 'src')
from database import init_db
init_db()
print('✅ Database initialized')
"

# Start the application
echo "🌟 Starting FastAPI server..."
python3 -m uvicorn src.main:app --host localhost --port 8000 --reload
"""
        
        dev_script = scripts_dir / "start_dev.sh"
        with open(dev_script, 'w') as f:
            f.write(dev_script_content)
        dev_script.chmod(0o755)
        print(f"✅ Created: {dev_script}")
        
        # Production startup script
        prod_script_content = """#!/bin/bash
# Production startup script

echo "🏭 Starting AI On-Call Agent (Production Mode)"

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "🐳 Running in Docker container"
    exec gunicorn --host 0.0.0.0 --port 8000 --workers 4 src.main:app
else
    echo "🖥️  Running on host system"
    
    # Start with Docker Compose
    if command -v docker-compose &> /dev/null; then
        echo "🐳 Starting with Docker Compose..."
        docker-compose up -d
    else
        echo "🏃 Starting directly..."
        python3 -m gunicorn --host 0.0.0.0 --port 8000 --workers 4 src.main:app
    fi
fi
"""
        
        prod_script = scripts_dir / "start_prod.sh"
        with open(prod_script, 'w') as f:
            f.write(prod_script_content)
        prod_script.chmod(0o755)
        print(f"✅ Created: {prod_script}")
        
        # Cross-platform startup script (Python)
        py_script_content = f"""#!/usr/bin/env python3
'''
Cross-platform startup script for AI On-Call Agent
'''

import sys
import subprocess
import os
from pathlib import Path

def start_system(mode="{self.mode}"):
    print(f"🚀 Starting AI On-Call Agent - {{mode.upper()}} Mode")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    if mode == "dev":
        # Development mode
        try:
            print("🌟 Starting development server...")
            subprocess.run([
                sys.executable, "-m", "uvicorn", 
                "src.main:app", 
                "--host", "localhost", 
                "--port", "8000", 
                "--reload"
            ], check=True)
        except KeyboardInterrupt:
            print("\\n👋 Development server stopped")
    
    elif mode == "prod":
        # Production mode
        try:
            print("🏭 Starting production server...")
            subprocess.run([
                sys.executable, "-m", "gunicorn",
                "--host", "0.0.0.0",
                "--port", "8000", 
                "--workers", "4",
                "src.main:app"
            ], check=True)
        except KeyboardInterrupt:
            print("\\n👋 Production server stopped")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start AI On-Call Agent")
    parser.add_argument("--mode", choices=["dev", "prod"], default="{self.mode}",
                      help="Startup mode")
    args = parser.parse_args()
    
    start_system(args.mode)
"""
        
        py_script = scripts_dir / "start.py"
        with open(py_script, 'w') as f:
            f.write(py_script_content)
        py_script.chmod(0o755)
        print(f"✅ Created: {py_script}")
        
        print("✅ Startup scripts created")
    
    def verify_installation(self):
        """Verify the installation."""
        print("\n🔍 Verifying Installation...")
        
        # Check Python imports
        test_imports = [
            "fastapi",
            "uvicorn", 
            "sklearn",
            "pandas",
            "numpy",
            "typer",
            "rich"
        ]
        
        for module in test_imports:
            try:
                __import__(module)
                print(f"✅ {module} imported successfully")
            except ImportError:
                print(f"❌ {module} import failed")
        
        # Check configuration files
        config_files = [
            "config/config.json",
            ".env",
            "requirements.txt"
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                print(f"✅ {config_file} exists")
            else:
                print(f"❌ {config_file} missing")
        
        # Test CLI command
        try:
            result = subprocess.run([
                self.python_executable, "cli.py", "status"
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ CLI command working")
            else:
                print("⚠️  CLI command returned error")
        except Exception as e:
            print(f"⚠️  CLI test failed: {e}")
        
        print("✅ Installation verification completed")
    
    def print_next_steps(self):
        """Print next steps for the user."""
        print("\n🎉 Setup Complete!")
        print("=" * 60)
        
        if self.mode == "dev":
            print("📚 Development Mode Setup Complete")
            print("\n🚀 To start the system:")
            print(f"   bash scripts/start_dev.sh")
            print(f"   # OR")
            print(f"   python3 scripts/start.py --mode dev")
            print(f"   # OR")
            print(f"   python3 -m uvicorn src.main:app --host localhost --port 8000 --reload")
            
            print("\n🧪 To run tests:")
            print(f"   python3 -m pytest")
            print(f"   python3 test_and_train.py")
            print(f"   python3 system_test.py")
            
            print("\n📖 Documentation:")
            print(f"   Open docs/AI_ON_CALL_AGENT_DUMMYS_GUIDE.html in browser")
            
        elif self.mode == "prod":
            print("🏭 Production Mode Setup Complete")
            print("\n🚀 To start the system:")
            print(f"   bash scripts/start_prod.sh")
            print(f"   # OR with Docker:")
            print(f"   docker-compose up -d")
            print(f"   # OR with Kubernetes:")
            print(f"   kubectl apply -f k8s/deployment.yaml")
            
            print("\n📊 Monitoring:")
            print(f"   Prometheus: http://localhost:9090")
            print(f"   Application: http://localhost:8000")
        
        print("\n⚙️  Configuration:")
        print(f"   Edit .env file for environment variables")
        print(f"   Edit config/config.json for application settings")
        
        print("\n🆘 Need help?")
        print(f"   Check docs/AI_ON_CALL_AGENT_DUMMYS_GUIDE.md")
        print(f"   Run: python3 cli.py --help")
        
        print(f"\n🎯 System URL: http://localhost:8000")
        print(f"📚 API Docs: http://localhost:8000/docs")
    
    def run_setup(self):
        """Run the complete setup process."""
        steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Project Structure", self.create_project_structure),
            ("Python Dependencies", self.install_python_dependencies),
            ("Configuration Files", self.create_configuration_files),
            ("Startup Scripts", self.create_startup_scripts),
            ("Installation Verification", self.verify_installation)
        ]
        
        if self.mode == "dev":
            steps.insert(-1, ("Development Tools", self.setup_development_tools))
        elif self.mode == "prod":
            steps.insert(-1, ("Production Tools", self.setup_production_tools))
        
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                print(f"\n{'='*20} {step_name} {'='*20}")
                result = step_func()
                if result is False:
                    failed_steps.append(step_name)
                    print(f"❌ {step_name} failed")
                else:
                    print(f"✅ {step_name} completed")
            except Exception as e:
                print(f"❌ {step_name} failed with error: {e}")
                failed_steps.append(step_name)
        
        print(f"\n{'='*60}")
        if failed_steps:
            print(f"⚠️  Setup completed with warnings. Failed steps: {', '.join(failed_steps)}")
        else:
            print("🎉 Setup completed successfully!")
        
        self.print_next_steps()
        
        return len(failed_steps) == 0

def main():
    parser = argparse.ArgumentParser(description="AI On-Call Agent Setup Script")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev",
                      help="Setup mode: dev (development) or prod (production)")
    parser.add_argument("--skip-deps", action="store_true",
                      help="Skip dependency installation")
    
    args = parser.parse_args()
    
    setup = AIOnCallAgentSetup(mode=args.mode)
    
    if args.skip_deps:
        print("⏭️  Skipping dependency installation")
        setup.install_python_dependencies = lambda: print("⏭️  Skipped dependency installation")
    
    success = setup.run_setup()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
