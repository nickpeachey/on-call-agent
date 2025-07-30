#!/bin/bash
# Quick Setup Script for AI On-Call Agent
# Usage: ./quick_setup.sh [dev|prod]

set -e  # Exit on any error

MODE=${1:-dev}
PYTHON_CMD=${PYTHON_CMD:-python3}

echo "🚀 AI On-Call Agent Quick Setup - $MODE Mode"
echo "============================================================"

# Check if Python is available
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Run the main setup script
echo "🔄 Running main setup script..."
$PYTHON_CMD setup.py --mode $MODE

echo ""
echo "🎉 Quick setup completed!"
echo ""

if [ "$MODE" = "dev" ]; then
    echo "🚀 To start development server:"
    echo "   bash scripts/start_dev.sh"
    echo "   # OR"
    echo "   $PYTHON_CMD scripts/start.py --mode dev"
    echo ""
    echo "🧪 To run tests:"
    echo "   $PYTHON_CMD test_and_train.py"
    echo "   $PYTHON_CMD system_test.py"
else
    echo "🏭 To start production server:"
    echo "   bash scripts/start_prod.sh"
    echo "   # OR with Docker:"
    echo "   docker-compose up -d"
fi

echo ""
echo "📚 Documentation: docs/AI_ON_CALL_AGENT_DUMMYS_GUIDE.html"
echo "🌐 API Docs: http://localhost:8000/docs (after starting)"
echo ""
echo "Happy coding! 🎯"
