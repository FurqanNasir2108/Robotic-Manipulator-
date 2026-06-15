"""Simulated sensor interface for edge deployment testing."""

from __future__ import annotations

import numpy as np


class SimulatedSensorStream:
    """Simulated sensor stream that provides task conditions from a test dataset.

    Parameters
    ----------
    test_data : dict
        Loaded test dataset with keys: waypoints, shape_type, etc.
    rate_hz : float
        Simulated sensor rate (for logging, not real-time delay).
    """

    def __init__(self, test_data: dict, rate_hz: float = 10.0):
        self.waypoints = np.asarray(test_data["waypoints"])
        self.shape_types = test_data.get("shape_type", [None] * len(self.waypoints))
        self.rate_hz = rate_hz
        self._index = 0

    def get_next_condition(self) -> dict:
        """Get the next condition from the stream.

        Returns
        -------
        dict
            Keys: waypoints, shape_type, index.
        """
        if self._index >= len(self.waypoints):
            self._index = 0

        result = {
            "waypoints": self.waypoints[self._index],
            "shape_type": self.shape_types[self._index],
            "index": self._index,
        }
        self._index += 1
        return result

    def reset(self):
        """Reset the stream to the beginning."""
        self._index = 0

    def __len__(self):
        return len(self.waypoints)


class SensorPreprocessor:
    """Preprocess raw sensor conditions for model input.

    Parameters
    ----------
    normalizer : Normalizer or None
    """

    def __init__(self, normalizer=None):
        self.normalizer = normalizer

    def preprocess(self, raw_condition: dict) -> np.ndarray:
        """Convert raw condition dict to model-ready numpy array.

        Parameters
        ----------
        raw_condition : dict
            From SimulatedSensorStream.get_next_condition().

        Returns
        -------
        np.ndarray
            Preprocessed condition array.
        """
        waypoints = np.asarray(raw_condition["waypoints"], dtype=np.float32)
        if self.normalizer is not None and hasattr(self.normalizer, "normalize_waypoints"):
            waypoints = self.normalizer.normalize_waypoints(waypoints)
        return waypoints
