# robomerge/robomerge/transform.py

from typing import Dict
import numpy as np

from robomerge.ingestion import DROIDEpisode


class DataStandardizer:
    """Standardizes DROID data for FAST compatibility."""
    
    def __init__(self, target_freq: float = 50.0):
        self.target_freq = target_freq
    
    def standardize_episode(self, episode: DROIDEpisode) -> Dict[str, np.ndarray]:
        """Convert episode to standardized format."""
        # Resample to target frequency
        resampled = self._resample_timeseries(
            episode.actions, 
            episode.timestamps, 
            self.target_freq
        )
        
        # Normalize action space to [-1, 1]
        normalized = self._normalize_actions(resampled)
        
        # Prepare FAST-compatible format
        return {
            'actions': normalized,
            'timestamps': np.arange(len(normalized)) / self.target_freq,
            'states': self._resample_timeseries(
                episode.states,
                episode.timestamps,
                self.target_freq
            )
        }
    
    def _resample_timeseries(self, data: np.ndarray, 
                            timestamps: np.ndarray,
                            target_freq: float) -> np.ndarray:
        """Resample time series data to target frequency."""
        # Create regular timestamp grid at target frequency
        t_start = timestamps[0]
        t_end = timestamps[-1]
        t_new = np.arange(t_start, t_end, 1.0/target_freq)
        
        # Interpolate each dimension
        result = np.zeros((len(t_new), data.shape[1]))
        for i in range(data.shape[1]):
            result[:, i] = np.interp(t_new, timestamps, data[:, i])
        
        return result
    
    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions to [-1, 1] range using quantiles."""
        q_low = np.percentile(actions, 1, axis=0)
        q_high = np.percentile(actions, 99, axis=0)
        
        normalized = 2 * (actions - q_low) / (q_high - q_low) - 1
        return np.clip(normalized, -1, 1)