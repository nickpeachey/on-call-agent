"""Comprehensive tests for ML service functionality."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.services.ml_service import MLService
from src.core.config import settings


@pytest.fixture
def temp_model_path():
    """Create a temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def ml_service(temp_model_path):
    """Create MLService instance with temporary model path."""
    with patch.object(settings, 'ml_model_path', str(temp_model_path)):
        service = MLService()
        return service


@pytest.mark.asyncio
class TestMLService:
    """Test ML service functionality."""
    
    async def test_ml_service_initialization(self, ml_service):
        """Test ML service initializes correctly."""
        assert ml_service.model_path.exists()
        assert ml_service.incident_classifier is None
        assert ml_service.action_recommender is None
        assert not ml_service.model_metadata["incident_classifier"]["loaded"]
        assert not ml_service.model_metadata["action_recommender"]["loaded"]
    
    async def test_initialize_with_no_existing_models(self, ml_service):
        """Test initialization when no models exist."""
        with patch.object(ml_service, 'train_initial_models', new_callable=AsyncMock) as mock_train:
            mock_train.return_value = {"success": True}
            
            await ml_service.initialize()
            
            mock_train.assert_called_once()
    
    async def test_load_nonexistent_models(self, ml_service):
        """Test loading when no models exist."""
        result = await ml_service.load_models()
        assert result is True  # Should not fail, just not load anything
    
    async def test_save_and_load_models(self, ml_service):
        """Test saving and loading models."""
        # Create and train mock models with sample data
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create sample training data
        sample_texts = ["database error", "network timeout", "memory leak", "disk full"]
        sample_severities = ["high", "medium", "high", "critical"]
        sample_actions = ["restart_service", "check_network", "check_memory", "clean_disk"]
        
        # Initialize and fit vectorizers
        ml_service.text_vectorizer = TfidfVectorizer(max_features=100)
        ml_service.action_vectorizer = TfidfVectorizer(max_features=50)
        
        text_features = ml_service.text_vectorizer.fit_transform(sample_texts)
        action_features = ml_service.action_vectorizer.fit_transform(sample_actions)
        
        # Initialize and train classifiers
        ml_service.incident_classifier = RandomForestClassifier(n_estimators=5)
        ml_service.action_recommender = RandomForestClassifier(n_estimators=5)
        
        ml_service.incident_classifier.fit(text_features, sample_severities)
        ml_service.action_recommender.fit(text_features, sample_actions)
        
        # Save models
        await ml_service.save_models()
        
        # Verify files were created
        assert (ml_service.model_path / "incident_classifier.joblib").exists()
        assert (ml_service.model_path / "action_recommender.joblib").exists()
        assert (ml_service.model_path / "text_vectorizer.joblib").exists()
        assert (ml_service.model_path / "action_vectorizer.joblib").exists()
        assert (ml_service.model_path / "model_metadata.json").exists()
        
        # Clear models and reload
        ml_service.incident_classifier = None
        ml_service.action_recommender = None
        ml_service.text_vectorizer = None
        ml_service.action_vectorizer = None
        
        result = await ml_service.load_models()
        assert result is True
        assert ml_service.incident_classifier is not None
        assert ml_service.action_recommender is not None
    
    async def test_train_initial_models_with_sample_data(self, ml_service):
        """Test training initial models with sample data."""
        with patch('src.services.ml_service.get_db_session') as mock_db:
            # Mock session to return low counts (use sample data)
            mock_session = AsyncMock()
            mock_session.scalar.return_value = 5  # Low count, will use sample data
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            result = await ml_service.train_initial_models()
            
            assert result["success"] is True
            assert "incident_classifier" in result
            assert "action_recommender" in result
            assert ml_service.incident_classifier is not None
            assert ml_service.action_recommender is not None
    
    async def test_predict_incident_severity(self, ml_service):
        """Test incident severity prediction."""
        # Test with no model loaded
        severity, confidence = await ml_service.predict_incident_severity("database error")
        assert severity == "medium"
        assert confidence == 0.5
        
        # Test with trained model (mock database access)
        with patch('src.services.ml_service.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_session.scalar.return_value = 5  # Low count, will use sample data
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            await ml_service.train_initial_models()
            severity, confidence = await ml_service.predict_incident_severity("database connection timeout")
            assert severity in ["low", "medium", "high", "critical"]
            assert 0 <= confidence <= 1
    
    async def test_recommend_action(self, ml_service):
        """Test action recommendation."""
        # Test with no model loaded
        action, confidence = await ml_service.recommend_action("database error")
        assert action == "check_logs"
        assert confidence == 0.5
        
        # Test with trained model (mock database access)
        with patch('src.services.ml_service.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_session.scalar.return_value = 5  # Low count, will use sample data
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            await ml_service.train_initial_models()
            action, confidence = await ml_service.recommend_action("database connection timeout")
            assert action in ["restart_service", "scale_up", "clear_cache", "check_connectivity", 
                             "check_logs", "renew_certificate"]
            assert 0 <= confidence <= 1
    
    async def test_evaluate_models(self, ml_service):
        """Test model evaluation."""
        # Test with no models
        result = await ml_service.evaluate_models()
        assert result == {}
        
        # Test with trained models (mock database access)
        with patch('src.services.ml_service.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_session.scalar.return_value = 5  # Low count, will use sample data
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            await ml_service.train_initial_models()
            result = await ml_service.evaluate_models()
            
            assert "incident_classifier" in result
            assert "action_recommender" in result
            assert result["incident_classifier"]["status"] == "loaded"
            assert result["action_recommender"]["status"] == "loaded"
    
    async def test_get_model_status(self, ml_service):
        """Test getting model status."""
        status = ml_service.get_model_status()
        
        assert "models" in status
        assert "model_path" in status
        assert "service_status" in status
        assert status["service_status"] == "no_models"
        
        # After training (mock database access)
        with patch('src.services.ml_service.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_session.scalar.return_value = 5  # Low count, will use sample data
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            await ml_service.train_initial_models()
            status = ml_service.get_model_status()
            assert status["service_status"] == "running"
    
    async def test_generate_sample_training_data(self, ml_service):
        """Test sample training data generation."""
        sample_data = ml_service._generate_sample_training_data()
        
        assert "incidents" in sample_data
        assert "severities" in sample_data
        assert "action_incidents" in sample_data
        assert "action_types" in sample_data
        
        assert len(sample_data["incidents"]) == 100
        assert len(sample_data["severities"]) == 100
        assert len(sample_data["action_incidents"]) == 100
        assert len(sample_data["action_types"]) == 100
        
        # Check data quality
        assert all(isinstance(incident, str) for incident in sample_data["incidents"])
        assert all(severity in ["low", "medium", "high", "critical"] 
                  for severity in sample_data["severities"])
    
    async def test_train_on_sample_data(self, ml_service):
        """Test training on sample data."""
        sample_data = ml_service._generate_sample_training_data()
        result = await ml_service._train_on_sample_data(sample_data)
        
        assert result["success"] is True
        assert "incident_classifier" in result
        assert "action_recommender" in result
        assert result["training_samples"] == 100
        
        # Verify models were created and trained
        assert ml_service.incident_classifier is not None
        assert ml_service.action_recommender is not None
        assert ml_service.text_vectorizer is not None
        assert ml_service.action_vectorizer is not None
    
    async def test_train_models_integration(self, ml_service):
        """Test full model training integration."""
        result = await ml_service.train_models()
        
        assert "training_started_at" in result
        assert "success" in result
        
        # Should at least have placeholder models
        assert result["incident_classifier"] is not None
        assert result["action_recommender"] is not None
    
    async def test_error_handling_in_prediction(self, ml_service):
        """Test error handling in prediction methods."""
        # Mock a broken model
        ml_service.incident_classifier = Mock()
        ml_service.text_vectorizer = Mock()
        ml_service.incident_classifier.predict.side_effect = Exception("Model error")
        
        severity, confidence = await ml_service.predict_incident_severity("test")
        assert severity == "medium"  # Fallback
        assert confidence == 0.5  # Fallback
    
    async def test_model_metadata_persistence(self, ml_service):
        """Test that model metadata is properly saved and loaded."""
        # Train models (mock database access)
        with patch('src.services.ml_service.get_db_session') as mock_db:
            mock_session = AsyncMock()
            mock_session.scalar.return_value = 5  # Low count, will use sample data
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            await ml_service.train_initial_models()
            
            # Save models
            await ml_service.save_models()
            
            # Create new instance and load
            with patch.object(settings, 'ml_model_path', str(ml_service.model_path)):
                new_service = MLService()
                await new_service.load_models()
                
                # Check metadata was loaded
                assert new_service.model_metadata["incident_classifier"]["loaded"]
                assert new_service.model_metadata["action_recommender"]["loaded"]
                assert new_service.model_metadata["incident_classifier"]["accuracy"] > 0
                assert new_service.model_metadata["action_recommender"]["accuracy"] > 0


@pytest.mark.integration
class TestMLServiceIntegration:
    """Integration tests for ML service with database."""
    
    async def test_full_ml_pipeline(self, ml_service):
        """Test complete ML pipeline from training to prediction."""
        # Initialize service
        await ml_service.initialize()
        
        # Should have trained models
        assert ml_service.incident_classifier is not None
        assert ml_service.action_recommender is not None
        
        # Test predictions
        severity, sev_conf = await ml_service.predict_incident_severity(
            "Critical database connection failure in production"
        )
        action, act_conf = await ml_service.recommend_action(
            "Critical database connection failure in production"
        )
        
        assert severity in ["low", "medium", "high", "critical"]
        assert action in ["restart_service", "scale_up", "clear_cache", "check_connectivity",
                         "check_logs", "renew_certificate"]
        assert 0 <= sev_conf <= 1
        assert 0 <= act_conf <= 1
        
        # Test evaluation
        eval_results = await ml_service.evaluate_models()
        assert "incident_classifier" in eval_results
        assert "action_recommender" in eval_results
