import pytest
import numpy as np
from robomerge.fast_prep import FASTPreprocessor

class TestFASTPreprocessor:
    def test_chunk_creation(self):
        preprocessor = FASTPreprocessor(chunk_size=50, chunk_overlap=10)
        
        # Test with perfect size data
        actions = np.random.randn(100, 7)
        chunks = preprocessor._create_chunks(actions, 50, 10)
        assert chunks.shape == (3, 50, 7)  # Should create 3 chunks
        
        # Test padding of last chunk
        actions = np.random.randn(60, 7)
        chunks = preprocessor._create_chunks(actions, 50, 10)
        assert chunks.shape == (2, 50, 7)
        assert np.all(chunks[1, 40:] == 0)  # Check padding
    
    def test_observation_preparation(self):
        preprocessor = FASTPreprocessor()
        
        # Test with complete data
        data = {
            'images': {
                'wrist': np.random.randn(100, 224, 224, 3),
                'external': np.random.randn(100, 224, 224, 3)
            },
            'states': np.random.randn(100, 7)
        }
        
        obs = preprocessor._prepare_observations(data)
        assert 'image_wrist' in obs
        assert 'image_external' in obs
        assert 'robot_state' in obs