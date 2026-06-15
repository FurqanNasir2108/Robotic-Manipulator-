"""Unit tests for dataset construction and validation."""

import os
import tempfile
import numpy as np
import pytest
from src.data.dataset import validate_sample, save_dataset, load_dataset


def make_valid_sample(sample_id=0, N=50):
    """Create a minimal valid sample for testing."""
    return {
        'sample_id': sample_id,
        'shape_type': 'circle',
        'start_pose': np.array([1.5, 0.3, 0.0]),
        'goal_pose': np.array([1.5, -0.3, 0.0]),
        'waypoints': np.random.randn(N, 3),
        'q_sequence': np.random.randn(N, 3),
        'x_sequence': np.random.randn(N),
        'y_sequence': np.random.randn(N),
        'theta_sequence': np.random.randn(N),
        'dq_sequence': np.random.randn(N, 3),
        'ddq_sequence': np.random.randn(N, 3),
        'energy_cost': 1.23,
        'smoothness_cost': 0.45,
    }


class TestValidation:
    def test_valid_sample(self):
        assert validate_sample(make_valid_sample())

    def test_missing_key(self):
        s = make_valid_sample()
        del s['energy_cost']
        assert not validate_sample(s)

    def test_shape_mismatch(self):
        s = make_valid_sample()
        s['q_sequence'] = np.random.randn(10, 3)  # Wrong N
        assert not validate_sample(s)


class TestSaveLoad:
    def test_save_and_load(self):
        samples = [make_valid_sample(i) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.npz')
            save_dataset(samples, path)
            loaded = load_dataset(path)
            assert 'sample_id' in loaded
            assert len(loaded['sample_id']) == 5
            np.testing.assert_array_equal(loaded['sample_id'], [0, 1, 2, 3, 4])