"""Shared pytest fixtures and configuration for the threedgrut test suite."""
import os
import tempfile
from pathlib import Path
from typing import Generator, Any
import shutil

import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file within the temp directory."""
    temp_path = temp_dir / "test_file.txt"
    temp_path.write_text("test content")
    yield temp_path


@pytest.fixture
def sample_config() -> DictConfig:
    """Create a sample configuration for testing."""
    config = {
        "model": {
            "type": "gaussian",
            "num_points": 1000,
            "learning_rate": 0.001,
        },
        "dataset": {
            "type": "nerf",
            "path": "/path/to/dataset",
            "batch_size": 32,
        },
        "training": {
            "max_iterations": 10000,
            "checkpoint_interval": 1000,
            "validation_interval": 500,
        },
        "render": {
            "resolution": [800, 600],
            "samples_per_pixel": 1,
        }
    }
    return OmegaConf.create(config)


@pytest.fixture
def mock_dataset_path(temp_dir: Path) -> Path:
    """Create a mock dataset directory structure."""
    dataset_dir = temp_dir / "mock_dataset"
    dataset_dir.mkdir()
    
    # Create subdirectories
    (dataset_dir / "images").mkdir()
    (dataset_dir / "sparse").mkdir()
    (dataset_dir / "dense").mkdir()
    
    # Create some mock files
    (dataset_dir / "images" / "image_001.jpg").touch()
    (dataset_dir / "images" / "image_002.jpg").touch()
    (dataset_dir / "sparse" / "cameras.bin").touch()
    (dataset_dir / "sparse" / "images.bin").touch()
    (dataset_dir / "sparse" / "points3D.bin").touch()
    
    return dataset_dir


@pytest.fixture
def mock_checkpoint_path(temp_dir: Path) -> Path:
    """Create a mock checkpoint file."""
    checkpoint_path = temp_dir / "checkpoint.ckpt"
    checkpoint_path.write_text("mock checkpoint data")
    return checkpoint_path


@pytest.fixture
def mock_ply_file(temp_dir: Path) -> Path:
    """Create a mock PLY file for testing."""
    ply_path = temp_dir / "pointcloud.ply"
    ply_content = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
"""
    ply_path.write_text(ply_content)
    return ply_path


@pytest.fixture
def mock_camera_params() -> dict[str, Any]:
    """Create mock camera parameters."""
    return {
        "width": 800,
        "height": 600,
        "fx": 500.0,
        "fy": 500.0,
        "cx": 400.0,
        "cy": 300.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


@pytest.fixture
def mock_render_params() -> dict[str, Any]:
    """Create mock render parameters."""
    return {
        "resolution": [800, 600],
        "samples_per_pixel": 1,
        "background_color": [0.0, 0.0, 0.0],
        "near_plane": 0.1,
        "far_plane": 100.0,
    }


@pytest.fixture
def clean_environment(monkeypatch):
    """Clean environment variables for testing."""
    # Remove any existing CUDA-related environment variables
    cuda_vars = ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]
    for var in cuda_vars:
        monkeypatch.delenv(var, raising=False)
    
    # Set testing environment
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb to prevent actual logging during tests."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch, temp_dir):
    """Automatically change to temp directory for each test."""
    monkeypatch.chdir(temp_dir)
    yield
    # No need to change back as monkeypatch handles cleanup


@pytest.fixture
def sample_point_cloud() -> dict[str, list[float]]:
    """Create a sample point cloud data."""
    return {
        "positions": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "colors": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        "scales": [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
        ],
    }


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_gpu(gpu_available):
    """Skip test if GPU is not available."""
    if not gpu_available:
        pytest.skip("GPU not available")


@pytest.fixture
def mock_timer(mocker):
    """Mock timer for performance testing."""
    timer = mocker.MagicMock()
    timer.elapsed_time = 0.1
    timer.average_time = 0.1
    return timer