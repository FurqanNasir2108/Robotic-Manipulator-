"""Unit tests for normalization module."""

import os
import tempfile
import numpy as np
import pytest
from src.data.normalization import Normalizer


class TestZScoreNormalization:
    def test_fit_transform_roundtrip(self):
        norm = Normalizer(strategy='z_score')
        data = np.random.randn(100, 3)
        norm.fit(data, 'test')
        transformed = norm.transform(data, 'test')
        recovered = norm.inverse_transform(transformed, 'test')
        np.testing.assert_allclose(recovered, data, atol=1e-10)

    def test_transformed_stats(self):
        norm = Normalizer(strategy='z_score')
        data = np.random.randn(1000, 3) * 5 + 10
        norm.fit(data, 'test')
        transformed = norm.transform(data, 'test')
        np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=0.1)
        np.testing.assert_allclose(transformed.std(axis=0), 1, atol=0.1)


class TestMinMaxNormalization:
    def test_fit_transform_roundtrip(self):
        norm = Normalizer(strategy='min_max')
        data = np.random.randn(100, 3)
        norm.fit(data, 'test')
        transformed = norm.transform(data, 'test')
        recovered = norm.inverse_transform(transformed, 'test')
        np.testing.assert_allclose(recovered, data, atol=1e-10)

    def test_transformed_range(self):
        norm = Normalizer(strategy='min_max')
        data = np.random.randn(1000, 3)
        norm.fit(data, 'test')
        transformed = norm.transform(data, 'test')
        assert transformed.min() >= -0.01
        assert transformed.max() <= 1.01


class TestSaveLoad:
    def test_save_and_load(self):
        norm = Normalizer(strategy='z_score')
        data = np.random.randn(100, 3)
        norm.fit(data, 'test')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'stats.json')
            norm.save(path)
            loaded = Normalizer.load(path)
            assert loaded.strategy == 'z_score'
            transformed_orig = norm.transform(data, 'test')
            transformed_loaded = loaded.transform(data, 'test')
            np.testing.assert_allclose(transformed_orig, transformed_loaded)


class TestInvalidStrategy:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            Normalizer(strategy='unknown')