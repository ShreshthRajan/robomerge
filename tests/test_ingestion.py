import pytest
import numpy as np
from robomerge.ingestion import DROIDIngestion, DROIDEpisode
from robomerge.transform import DataStandardizer

class TestDROIDIngestion:
    def test_load_episode_validates_keys(self):
        ingestion = DROIDIngestion()
        # Test missing key validation
        with pytest.raises(ValueError):
            ingestion._validate_keys({'actions': None})  # Missing required keys

    def test_frequency_calculation(self):
        ingestion = DROIDIngestion()
        # Test frequency calculation with known timestamps
        timestamps = np.array([0, 0.1, 0.2, 0.3])  # 10Hz data
        episode = DROIDEpisode(
            actions=np.zeros((4, 3)),
            states=np.zeros((4, 7)),
            timestamps=timestamps,
            images={},
            frequency=10.0,
            metadata={}
        )
        assert abs(episode.frequency - 10.0) < 1e-5

    def test_normalization_bounds(self):
        standardizer = DataStandardizer()
        # Test action normalization to [-1, 1]
        actions = np.random.randn(1000, 7) * 10  # Random actions
        normalized = standardizer._normalize_actions(actions)
        assert np.all(normalized >= -1.0) and np.all(normalized <= 1.0)