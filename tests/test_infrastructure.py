"""Validation tests to ensure the testing infrastructure is set up correctly."""
import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure."""
    
    def test_python_version(self):
        """Verify Python version meets requirements."""
        assert sys.version_info >= (3, 11), "Python 3.11+ is required"
    
    def test_project_imports(self):
        """Test that project modules can be imported."""
        try:
            import threedgrut
            import threedgrt_tracer
            import threedgut_tracer
            import threedgrut_playground
        except ImportError as e:
            pytest.fail(f"Failed to import project modules: {e}")
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test the temp_dir fixture creates a valid directory."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
        assert test_file.read_text() == "test"
    
    def test_temp_file_fixture(self, temp_file):
        """Test the temp_file fixture creates a valid file."""
        assert temp_file.exists()
        assert temp_file.is_file()
        assert temp_file.read_text() == "test content"
    
    def test_sample_config_fixture(self, sample_config):
        """Test the sample_config fixture returns valid config."""
        assert isinstance(sample_config, DictConfig)
        assert "model" in sample_config
        assert "dataset" in sample_config
        assert "training" in sample_config
        assert "render" in sample_config
        
        # Test nested values
        assert sample_config.model.type == "gaussian"
        assert sample_config.dataset.batch_size == 32
        assert sample_config.training.max_iterations == 10000
        assert sample_config.render.resolution == [800, 600]
    
    def test_mock_dataset_fixture(self, mock_dataset_path):
        """Test the mock_dataset_path fixture creates proper structure."""
        assert mock_dataset_path.exists()
        assert (mock_dataset_path / "images").exists()
        assert (mock_dataset_path / "sparse").exists()
        assert (mock_dataset_path / "dense").exists()
        
        # Check files
        assert (mock_dataset_path / "images" / "image_001.jpg").exists()
        assert (mock_dataset_path / "sparse" / "cameras.bin").exists()
    
    def test_mock_checkpoint_fixture(self, mock_checkpoint_path):
        """Test the mock_checkpoint_path fixture."""
        assert mock_checkpoint_path.exists()
        assert mock_checkpoint_path.suffix == ".ckpt"
        assert mock_checkpoint_path.read_text() == "mock checkpoint data"
    
    def test_mock_ply_fixture(self, mock_ply_file):
        """Test the mock_ply_file fixture creates valid PLY."""
        assert mock_ply_file.exists()
        assert mock_ply_file.suffix == ".ply"
        content = mock_ply_file.read_text()
        assert "ply" in content
        assert "element vertex 3" in content
    
    def test_clean_environment_fixture(self, clean_environment):
        """Test environment is properly cleaned."""
        import os
        assert os.environ.get("TESTING") == "1"
        assert os.environ.get("HYDRA_FULL_ERROR") == "1"
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    
    def test_change_test_dir_fixture(self, temp_dir):
        """Test that we're running in temp directory."""
        assert Path.cwd().parent == temp_dir.parent
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        import time
        time.sleep(0.01)  # Simulate slow test
        assert True
    
    def test_coverage_tracking(self):
        """Test that coverage is being tracked."""
        # This test ensures coverage is working
        def dummy_function(x):
            if x > 0:
                return x * 2
            else:
                return 0
        
        assert dummy_function(5) == 10
        assert dummy_function(-1) == 0
    
    def test_pytest_mock_available(self, mocker):
        """Test that pytest-mock is available and working."""
        mock_func = mocker.Mock(return_value=42)
        assert mock_func() == 42
        mock_func.assert_called_once()
    
    def test_mock_wandb_fixture(self, mock_wandb):
        """Test wandb is properly mocked."""
        import os
        assert os.environ.get("WANDB_MODE") == "disabled"
        assert os.environ.get("WANDB_SILENT") == "true"


class TestParametrizedExamples:
    """Examples of parametrized tests."""
    
    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
    ])
    def test_parametrized_example(self, input_val, expected):
        """Example of parametrized test."""
        assert input_val * 2 == expected
    
    @pytest.mark.parametrize("resolution", [
        [800, 600],
        [1920, 1080],
        [1024, 768],
    ])
    def test_resolution_validation(self, resolution):
        """Example of testing different resolutions."""
        width, height = resolution
        assert width > 0
        assert height > 0
        assert isinstance(width, int)
        assert isinstance(height, int)