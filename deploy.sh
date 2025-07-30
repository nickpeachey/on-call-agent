#!/bin/bash

# AI On-Call Agent Production Deployment Script
set -e

echo "🚀 AI On-Call Agent Production Deployment"
echo "=========================================="

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f ".env" ]; then
        echo "❌ .env file not found. Please copy .env.example and configure it."
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Build and deploy
deploy() {
    echo "🏗️ Building and deploying services..."
    
    # Build the application
    echo "Building AI On-Call Agent..."
    docker-compose build
    
    # Start services
    echo "Starting services..."
    docker-compose up -d
    
    echo "✅ Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    echo "⏳ Waiting for services to be ready..."
    
    # Wait for database
    echo "Waiting for PostgreSQL..."
    while ! docker-compose exec -T postgres pg_isready -U oncall_user -d oncall_agent; do
        sleep 2
    done
    
    # Wait for Redis
    echo "Waiting for Redis..."
    while ! docker-compose exec -T redis redis-cli ping; do
        sleep 2
    done
    
    # Wait for main application
    echo "Waiting for AI On-Call Agent..."
    while ! curl -f http://localhost:8000/health &> /dev/null; do
        sleep 5
    done
    
    echo "✅ All services are ready"
}

# Run tests
run_tests() {
    echo "🧪 Running production validation tests..."
    
    # Run validation inside container
    docker-compose exec oncall-agent python scripts/validate_production.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Production validation passed"
    else
        echo "❌ Production validation failed"
        exit 1
    fi
}

# Show status
show_status() {
    echo "📊 Deployment Status"
    echo "===================="
    
    echo "Services:"
    docker-compose ps
    
    echo ""
    echo "Health Checks:"
    curl -s http://localhost:8000/health | jq '.'
    
    echo ""
    echo "Access URLs:"
    echo "- API Documentation: http://localhost:8000/docs"
    echo "- Health Check: http://localhost:8000/health"
    echo "- Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "- Prometheus: http://localhost:9090"
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "check")
            check_prerequisites
            ;;
        "deploy")
            check_prerequisites
            deploy
            wait_for_services
            show_status
            ;;
        "test")
            run_tests
            ;;
        "status")
            show_status
            ;;
        "stop")
            echo "🛑 Stopping services..."
            docker-compose down
            ;;
        "logs")
            docker-compose logs -f "${2:-oncall-agent}"
            ;;
        "restart")
            echo "🔄 Restarting services..."
            docker-compose restart
            wait_for_services
            show_status
            ;;
        *)
            echo "Usage: $0 {deploy|check|test|status|stop|logs|restart}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Full deployment (default)"
            echo "  check    - Check prerequisites only"
            echo "  test     - Run production validation tests"
            echo "  status   - Show deployment status"
            echo "  stop     - Stop all services"
            echo "  logs     - Show service logs"
            echo "  restart  - Restart all services"
            exit 1
            ;;
    esac
}

main "$@"
