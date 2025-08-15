"""
Tests for the augmentation module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from datasets import Dataset

from synthdoc.augmentation import (
    AugmentationProcessor,
    AugmentationConfig,
    LIGHT_AUGMENTATIONS,
    BALANCED_AUGMENTATIONS,
    AVAILABLE_AUGMENTATIONS,
    detect_image_column,
    calculate_augmentation_ratios,
)
from synthdoc.augmentation.engine import AugmentationEngine


class TestAugmentationConfig:
    """Test augmentation configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = AugmentationConfig()
        assert config.original_ratio == 0.3
        assert "original" in config.augmentations
        assert config.preserve_metadata is True
    
    def test_list_augmentations(self):
        """Test list-based augmentation configuration."""
        config = AugmentationConfig(
            augmentations=["brightness", "folding", "original"]
        )
        assert isinstance(config.augmentations, list)
        assert "brightness" in config.augmentations
        assert "original" in config.augmentations
    
    def test_dict_augmentations(self):
        """Test dict-based augmentation configuration."""
        config = AugmentationConfig(
            augmentations={"brightness": 0.4, "original": 0.6}
        )
        assert isinstance(config.augmentations, dict)
        assert config.augmentations["brightness"] == 0.4
    
    def test_invalid_augmentations(self):
        """Test validation of invalid augmentations."""
        with pytest.raises(ValueError, match="Unsupported augmentations"):
            AugmentationConfig(augmentations=["invalid_aug"])
    
    def test_invalid_ratios(self):
        """Test validation of invalid ratios."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            AugmentationConfig(augmentations={"brightness": 0.5, "original": 0.8})


class TestAugmentationEngine:
    """Test augmentation engine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create augmentation engine instance."""
        return AugmentationEngine()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL image for testing."""
        # Create a white 100x100 image
        image = Image.new('RGB', (100, 100), color='white')
        return image
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine is not None
        assert hasattr(engine, 'apply_augmentation')
    
    def test_original_augmentation(self, engine, sample_image):
        """Test original augmentation with padding."""
        result, success, timing = engine.apply_augmentation(sample_image, "original")
        
        assert success is True
        assert isinstance(result, Image.Image)
        assert timing > 0
        # Should have some padding applied
        assert result.size[0] >= sample_image.size[0]
        assert result.size[1] >= sample_image.size[1]
    
    def test_brightness_augmentation(self, engine, sample_image):
        """Test brightness augmentation."""
        result, success, timing = engine.apply_augmentation(sample_image, "brightness")
        
        assert isinstance(result, Image.Image) 
        assert timing > 0
        # Result should be same size as input
        assert result.size == sample_image.size
    
    def test_invalid_augmentation(self, engine, sample_image):
        """Test handling of invalid augmentation."""
        result, success, timing = engine.apply_augmentation(sample_image, "invalid")
        
        assert success is False
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size
    
    def test_augmentation_plan_generation(self, engine):
        """Test augmentation plan generation."""
        ratios = {"brightness": 0.5, "original": 0.5}
        plan = engine.generate_augmentation_plan(100, ratios)
        
        assert len(plan) == 100
        assert "brightness" in plan
        assert "original" in plan
        
        # Check approximate distribution
        brightness_count = plan.count("brightness")
        original_count = plan.count("original")
        assert 40 <= brightness_count <= 60  # Allow some variance due to rounding
        assert 40 <= original_count <= 60


class TestAugmentationProcessor:
    """Test augmentation processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create augmentation processor instance."""
        return AugmentationProcessor()
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample HuggingFace dataset for testing."""
        # Create sample images
        images = []
        for i in range(5):
            img = Image.new('RGB', (100, 100), color='white')
            images.append(img)
        
        return Dataset.from_dict({
            "image": images,
            "text": [f"Sample text {i}" for i in range(5)],
            "id": [f"doc_{i}" for i in range(5)]
        })
    
    @pytest.fixture
    def temp_image_folder(self):
        """Create temporary folder with test images."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create test images
        for i in range(3):
            img = Image.new('RGB', (100, 100), color='white')
            img.save(temp_path / f"test_image_{i}.png")
        
        yield temp_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_image_column_detection(self, sample_dataset):
        """Test automatic image column detection."""
        column = detect_image_column(sample_dataset)
        assert column == "image"
    
    def test_dataset_processing(self, processor, sample_dataset):
        """Test processing of HuggingFace dataset."""
        config = AugmentationConfig(
            augmentations=["brightness", "original"],
            max_samples=3
        )
        
        result = processor.process_dataset(sample_dataset, config)
        
        assert isinstance(result, Dataset)
        assert len(result) == 3
        assert "augmentation_type" in result.column_names
        assert "augmentation_success" in result.column_names
    
    def test_folder_processing(self, processor, temp_image_folder):
        """Test processing of image folder."""
        output_dir = tempfile.mkdtemp()
        
        try:
            config = AugmentationConfig(
                augmentations=["brightness", "original"],
                max_samples=2
            )
            
            result = processor.process_folder(
                input_folder=temp_image_folder,
                output_folder=output_dir,
                config=config
            )
            
            assert isinstance(result, Dataset)
            assert len(result) == 2
            assert "image" in result.column_names
            assert "augmentation_type" in result.column_names
            
            # Check output files were created
            output_path = Path(output_dir)
            output_files = list(output_path.glob("*.png"))
            assert len(output_files) == 2
            
        finally:
            shutil.rmtree(output_dir)


class TestAugmentationHelpers:
    """Test helper functions."""
    
    def test_calculate_ratios(self):
        """Test augmentation ratio calculation."""
        augmentations = ["brightness", "folding"]
        ratios = calculate_augmentation_ratios(augmentations, original_ratio=0.4)
        
        assert "original" in ratios
        assert "brightness" in ratios
        assert "folding" in ratios
        assert abs(sum(ratios.values()) - 1.0) < 0.001
        assert ratios["original"] == 0.4
    
    def test_calculate_ratios_no_original(self):
        """Test ratio calculation when original not in list."""
        augmentations = ["brightness", "folding"]
        ratios = calculate_augmentation_ratios(augmentations)
        
        assert "original" in ratios
        assert ratios["original"] == 0.3  # default
        assert abs(sum(ratios.values()) - 1.0) < 0.001


class TestIntegration:
    """Test integration with SynthDoc core."""
    
    def test_core_integration(self):
        """Test that augmentation integrates with SynthDoc core."""
        from synthdoc import SynthDoc
        
        # Test that the method exists
        synth = SynthDoc(output_dir="./test_output")
        assert hasattr(synth, 'apply_augmentations')
    
    def test_available_augmentations_list(self):
        """Test that available augmentations list is complete."""
        # Should contain all augmentations from your original script
        expected_augs = [
            "brightness", "folding", "watermark", "colorshift", 
            "bleedthrough", "dirtydrum", "shadowcast"
        ]
        
        for aug in expected_augs:
            assert aug in AVAILABLE_AUGMENTATIONS
    
    def test_presets_exist(self):
        """Test that all presets are properly defined."""
        assert len(LIGHT_AUGMENTATIONS) >= 2
        assert len(BALANCED_AUGMENTATIONS) >= 4
        assert "original" in LIGHT_AUGMENTATIONS
        assert "original" in BALANCED_AUGMENTATIONS


if __name__ == "__main__":
    pytest.main([__file__])