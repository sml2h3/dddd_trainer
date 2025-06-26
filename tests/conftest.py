"""Shared pytest fixtures and configuration."""
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import yaml


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "model": {
            "name": "test_model",
            "type": "ddddocr",
            "backbone": "mobilenetv2",
            "num_classes": 10,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "save_interval": 1000,
        },
        "data": {
            "train_path": "/path/to/train",
            "val_path": "/path/to/val",
            "test_path": "/path/to/test",
        },
    }


@pytest.fixture
def sample_yaml_config(temp_dir: Path, mock_config: Dict[str, Any]) -> Path:
    """Create a sample YAML configuration file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(mock_config, f)
    return config_path


@pytest.fixture
def mock_image_path(temp_dir: Path) -> Path:
    """Create a mock image file path."""
    image_path = temp_dir / "test_image.jpg"
    # Create an empty file to represent the image
    image_path.touch()
    return image_path


@pytest.fixture
def mock_checkpoint_path(temp_dir: Path) -> Path:
    """Create a mock checkpoint file path."""
    checkpoint_path = temp_dir / "test_checkpoint.tar"
    # Create an empty file to represent the checkpoint
    checkpoint_path.touch()
    return checkpoint_path


@pytest.fixture
def sample_project_structure(temp_dir: Path) -> Path:
    """Create a sample project directory structure."""
    # Create directories
    (temp_dir / "configs").mkdir()
    (temp_dir / "nets" / "backbone").mkdir(parents=True)
    (temp_dir / "utils").mkdir()
    (temp_dir / "tools").mkdir()
    
    # Create __init__.py files
    (temp_dir / "configs" / "__init__.py").touch()
    (temp_dir / "nets" / "__init__.py").touch()
    (temp_dir / "nets" / "backbone" / "__init__.py").touch()
    (temp_dir / "utils" / "__init__.py").touch()
    (temp_dir / "tools" / "__init__.py").touch()
    
    return temp_dir


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module imports between tests to avoid state pollution."""
    import sys
    modules_to_remove = [
        module for module in sys.modules 
        if module.startswith(('configs', 'nets', 'utils', 'tools', 'projects'))
    ]
    for module in modules_to_remove:
        sys.modules.pop(module, None)
    yield


@pytest.fixture
def capture_logs():
    """Capture loguru logs for testing."""
    from loguru import logger
    from io import StringIO
    
    log_capture = StringIO()
    handler_id = logger.add(
        log_capture,
        format="{time} | {level} | {message}",
        level="DEBUG"
    )
    
    yield log_capture
    
    logger.remove(handler_id)


@pytest.fixture
def mock_training_data() -> Dict[str, Any]:
    """Provide mock training data."""
    return {
        "images": ["image1.jpg", "image2.jpg", "image3.jpg"],
        "labels": ["label1", "label2", "label3"],
        "batch_size": 2,
        "num_epochs": 5,
    }