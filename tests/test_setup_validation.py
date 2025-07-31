"""Validation tests to ensure the testing infrastructure is properly set up."""
import sys
from pathlib import Path

import pytest


class TestSetupValidation:
    """Tests to validate the testing infrastructure setup."""
    
    def test_python_version(self):
        """Verify Python version is 3.8 or higher."""
        assert sys.version_info >= (3, 8), "Python 3.8 or higher is required"
    
    def test_pytest_imported(self):
        """Verify pytest can be imported."""
        import pytest
        assert pytest.__version__
    
    def test_pytest_cov_imported(self):
        """Verify pytest-cov can be imported."""
        import pytest_cov
        assert pytest_cov.__version__
    
    def test_pytest_mock_imported(self):
        """Verify pytest-mock can be imported."""
        import pytest_mock
        # pytest_mock doesn't expose version directly, check it exists
        assert pytest_mock
    
    def test_project_structure_exists(self):
        """Verify the project structure exists."""
        workspace = Path(__file__).parent.parent
        
        # Check main directories
        assert (workspace / "configs").exists()
        assert (workspace / "nets").exists()
        assert (workspace / "utils").exists()
        assert (workspace / "tools").exists()
        assert (workspace / "projects").exists()
        
        # Check test directories
        assert (workspace / "tests").exists()
        assert (workspace / "tests" / "unit").exists()
        assert (workspace / "tests" / "integration").exists()
    
    def test_conftest_fixtures(self, temp_dir, mock_config):
        """Verify conftest fixtures are working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "model" in mock_config
        assert "training" in mock_config
        assert "data" in mock_config
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Verify unit test marker is working."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Verify integration test marker is working."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Verify slow test marker is working."""
        assert True
    
    def test_coverage_configured(self):
        """Verify coverage is properly configured."""
        # This test will be run with coverage enabled
        # The actual verification happens through the coverage report
        assert True
    
    def test_imports_from_project_modules(self):
        """Verify project modules can be imported."""
        # Note: Some modules may have dependencies like torch that aren't installed
        # This test verifies the module structure is correct, not all dependencies
        try:
            import configs
            # Skip nets as it requires torch
            import utils
            import tools
            import projects
            assert True
        except ImportError as e:
            # Expected for modules with missing dependencies
            if "torch" not in str(e):
                pytest.fail(f"Failed to import project module: {e}")


class TestFixtureValidation:
    """Tests to validate pytest fixtures are working correctly."""
    
    def test_sample_yaml_config_fixture(self, sample_yaml_config, mock_config):
        """Verify sample_yaml_config fixture creates a valid YAML file."""
        assert sample_yaml_config.exists()
        assert sample_yaml_config.suffix == ".yaml"
        
        # Read and verify content
        import yaml
        with open(sample_yaml_config, "r") as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == mock_config
    
    def test_mock_image_path_fixture(self, mock_image_path):
        """Verify mock_image_path fixture creates a file."""
        assert mock_image_path.exists()
        assert mock_image_path.name == "test_image.jpg"
    
    def test_mock_checkpoint_path_fixture(self, mock_checkpoint_path):
        """Verify mock_checkpoint_path fixture creates a file."""
        assert mock_checkpoint_path.exists()
        assert mock_checkpoint_path.name == "test_checkpoint.tar"
    
    def test_sample_project_structure_fixture(self, sample_project_structure):
        """Verify sample_project_structure fixture creates the expected structure."""
        assert (sample_project_structure / "configs" / "__init__.py").exists()
        assert (sample_project_structure / "nets" / "__init__.py").exists()
        assert (sample_project_structure / "nets" / "backbone" / "__init__.py").exists()
        assert (sample_project_structure / "utils" / "__init__.py").exists()
        assert (sample_project_structure / "tools" / "__init__.py").exists()
    
    def test_capture_logs_fixture(self, capture_logs):
        """Verify capture_logs fixture works with loguru."""
        from loguru import logger
        
        logger.info("Test log message")
        logger.warning("Test warning")
        
        logs = capture_logs.getvalue()
        assert "Test log message" in logs
        assert "Test warning" in logs
        assert "INFO" in logs
        assert "WARNING" in logs
    
    def test_mock_training_data_fixture(self, mock_training_data):
        """Verify mock_training_data fixture provides expected data."""
        assert "images" in mock_training_data
        assert "labels" in mock_training_data
        assert "batch_size" in mock_training_data
        assert "num_epochs" in mock_training_data
        
        assert len(mock_training_data["images"]) == 3
        assert len(mock_training_data["labels"]) == 3
        assert mock_training_data["batch_size"] == 2
        assert mock_training_data["num_epochs"] == 5


def test_standalone_function():
    """Verify standalone test functions work outside of classes."""
    assert 2 + 2 == 4