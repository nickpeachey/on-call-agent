#!/bin/bash
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
