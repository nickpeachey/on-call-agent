#!/bin/bash

# AI On-Call Agent Local Development Setup
echo "🚀 Starting AI On-Call Agent Local Development Environment"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start services in Docker
echo "📦 Starting PostgreSQL and Redis services..."
docker-compose -f docker-compose.dev.yml up -d postgres redis

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker ps | grep -q oncall-postgres-dev; then
    echo "✅ PostgreSQL is running on localhost:5432"
else
    echo "❌ PostgreSQL failed to start"
    exit 1
fi

if docker ps | grep -q oncall-redis-dev; then
    echo "✅ Redis is running on localhost:6379"
else
    echo "❌ Redis failed to start"
    exit 1
fi

# Optional: Start Airflow for testing
read -p "🛩️  Do you want to start Airflow for testing? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Starting Airflow services..."
    docker-compose -f docker-compose.dev.yml up -d airflow-postgres
    sleep 5
    docker-compose -f docker-compose.dev.yml run --rm airflow-webserver airflow db init
    docker-compose -f docker-compose.dev.yml up -d airflow-webserver
    echo "✅ Airflow will be available at http://localhost:8080 (admin/admin)"
fi

echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "📋 Next steps:"
echo "1. Copy the local environment file:"
echo "   cp .env.local .env"
echo ""
echo "2. Start the on-call agent locally:"
echo "   .venv/bin/python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8001 --env-file .env.local"
echo ""
echo "3. Test the API:"
echo "   curl -X POST 'http://localhost:8001/test/airflow-dag-failure?dag_id=test_dag&auto_resolve=true'"
echo ""
echo "🛑 To stop services:"
echo "   docker-compose -f docker-compose.dev.yml down"
