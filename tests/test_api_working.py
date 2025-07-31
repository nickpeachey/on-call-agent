"""Test working API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.main import app


class TestWorkingAPI:
    """Test the API endpoints that actually exist and work."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test the root API endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "AI On-Call Agent API"
    
    def test_health_endpoint(self, client):
        """Test the health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_api_v1_root(self, client):
        """Test the API v1 root endpoint."""
        response = client.get("/api/v1/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_monitoring_status(self, client):
        """Test monitoring status endpoint."""
        response = client.get("/api/v1/monitoring/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_log_polling_status(self, client):
        """Test log polling status endpoint."""
        response = client.get("/api/v1/logs/polling/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_ai_model_status(self, client):
        """Test AI model status endpoint."""
        response = client.get("/api/v1/api/v1/ai/model-status")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data or "status" in data
    
    def test_resolution_metrics(self, client):
        """Test resolution metrics endpoint."""
        response = client.get("/api/v1/resolutions/metrics")
        assert response.status_code == 200
        data = response.json()
        # Should return metrics data
        assert isinstance(data, dict)
    
    def test_enhanced_incidents_list(self, client):
        """Test enhanced incidents list endpoint."""
        response = client.get("/api/v1/enhanced-incidents/")
        assert response.status_code == 200
        data = response.json()
        # Should return a list
        assert isinstance(data, list)
    
    def test_knowledge_base_list(self, client):
        """Test knowledge base list endpoint."""
        response = client.get("/api/v1/knowledge/")
        assert response.status_code == 200
        data = response.json()
        # Should return a list
        assert isinstance(data, list)
    
    def test_actions_list(self, client):
        """Test actions list endpoint."""
        response = client.get("/api/v1/actions/")
        assert response.status_code == 200
        data = response.json()
        # Should return a list
        assert isinstance(data, list)
