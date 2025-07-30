#!/bin/bash
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
