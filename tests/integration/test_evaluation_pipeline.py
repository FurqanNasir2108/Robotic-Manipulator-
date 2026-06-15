"""Integration tests for the evaluation and plotting pipeline."""

import json
import os

import numpy as np

from src.data.normalization import Normalizer
from src.evaluate.comparisons import evaluate_model_set, load_test_dataset, write_summary_files
from scripts.plot_results import generate_result_plots


def _build_test_dataset(root_dir, n_steps=8):
    data_dir = os.path.join(root_dir, "processed")
    metadata_dir = os.path.join(root_dir, "metadata")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    shape_types = np.array(["circle", "square", "hexagon"] * 3)
    trajectories = []
    for index in range(len(shape_types)):
        t = np.linspace(0.0, 1.0, n_steps)
        q = np.stack(
            [
                0.05 * index + 0.2 * t,
                -0.03 * index + 0.1 * t,
                0.02 * index - 0.05 * t,
            ],
            axis=1,
        ).astype(np.float32)
        trajectories.append(q)

    q_sequence = np.stack(trajectories)
    waypoints = q_sequence.copy()
    start_pose = waypoints[:, 0, :]
    goal_pose = waypoints[:, -1, :]

    np.savez(
        os.path.join(data_dir, "test.npz"),
        waypoints=waypoints,
        q_sequence=q_sequence,
        start_pose=start_pose,
        goal_pose=goal_pose,
        shape_type=shape_types,
    )

    normalizer = Normalizer(strategy="z_score")
    normalizer.fit(q_sequence, "q_sequence")
    normalizer.fit(waypoints, "waypoints")
    normalizer.save(os.path.join(metadata_dir, "normalization_stats.json"))

    return data_dir, metadata_dir


class _DeterministicSpec:
    def __init__(self):
        self.predict_calls = 0
        self.timing_calls = 0

    def predict(self, condition, num_samples=1):
        self.predict_calls += 1
        return np.expand_dims(np.asarray(condition), axis=0)

    def timing(self, condition):
        self.timing_calls += 1
        return condition


class _GenerativeSpec:
    def __init__(self, offset=0.15):
        self.offset = offset
        self.predict_calls = 0
        self.timing_calls = 0

    def predict(self, condition, num_samples=2):
        self.predict_calls += 1
        condition = np.asarray(condition)
        samples = np.stack([condition, condition + self.offset], axis=0)
        return samples[:num_samples]

    def timing(self, condition):
        self.timing_calls += 1
        return condition


class TestEvaluationPipeline:
    def test_load_test_dataset_supports_random_and_stratified_subsetting(self, tmp_path):
        data_dir, metadata_dir = _build_test_dataset(tmp_path)

        random_a, _ = load_test_dataset(
            data_dir,
            metadata_dir,
            max_samples=6,
            seed=11,
            subset_strategy="random",
        )
        random_b, _ = load_test_dataset(
            data_dir,
            metadata_dir,
            max_samples=6,
            seed=11,
            subset_strategy="random",
        )
        stratified, _ = load_test_dataset(
            data_dir,
            metadata_dir,
            max_samples=6,
            seed=11,
            subset_strategy="stratified",
        )

        assert np.array_equal(random_a.selected_indices, random_b.selected_indices)
        assert not np.array_equal(random_a.selected_indices, np.arange(6))
        assert stratified.shape_counts == {"circle": 2, "hexagon": 2, "square": 2}

    def test_evaluation_writes_summary_plots_and_caches_full_k_samples(self, tmp_path):
        data_dir, metadata_dir = _build_test_dataset(tmp_path)
        dataset, normalizer = load_test_dataset(
            data_dir,
            metadata_dir,
            max_samples=6,
            seed=5,
            subset_strategy="stratified",
        )

        deterministic = _DeterministicSpec()
        generative = _GenerativeSpec(offset=0.2)
        model_specs = {
            "deterministic": {
                "kind": "deterministic",
                "predict": deterministic.predict,
                "timing": deterministic.timing,
                "uses_raw_waypoints": False,
                "outputs_normalized": True,
            },
            "generative": {
                "kind": "generative",
                "predict": generative.predict,
                "timing": generative.timing,
                "uses_raw_waypoints": False,
                "outputs_normalized": True,
            },
        }

        cache_dir = os.path.join(tmp_path, "cache")
        summary_json = os.path.join(tmp_path, "summary.json")
        summary_md = os.path.join(tmp_path, "summary.md")
        plots_dir = os.path.join(tmp_path, "plots")
        config = {
            "evaluation": {
                "seed": 7,
                "generative_samples": 2,
                "inference_runs": 3,
                "latency_num_conditions": 2,
                "latency_subset_strategy": "stratified",
                "cache_trajectories_dir": cache_dir,
                "cache_full_generative_samples": True,
            },
            "metrics": {
                "dt": 1.0,
                "path_method": "dtw",
                "diversity_method": "dtw",
            },
        }

        summary = evaluate_model_set(model_specs, dataset, normalizer, config)
        write_summary_files(summary, summary_json, summary_md)
        generate_result_plots(summary, plots_dir)

        assert summary["num_test_samples"] == 6
        assert summary["dataset"]["shape_counts"] == {"circle": 2, "hexagon": 2, "square": 2}
        assert summary["overall"]["generative"]["mean_joint_rmse"] > summary["overall"]["generative"]["oracle_joint_rmse"]

        expected_timing_calls = config["evaluation"]["latency_num_conditions"] * config["evaluation"]["inference_runs"]
        assert deterministic.timing_calls == expected_timing_calls
        assert generative.timing_calls == expected_timing_calls
        assert deterministic.predict_calls == 6
        assert generative.predict_calls == 6

        with np.load(os.path.join(cache_dir, "generative.npz"), allow_pickle=True) as cached:
            assert cached["q_predicted"].shape == (6, 2, 8, 3)
            assert cached["q_oracle_predicted"].shape == (6, 8, 3)
            assert cached["q_reference"].shape == (6, 8, 3)
            assert cached["waypoints"].shape == (6, 8, 3)

        with open(summary_json, "r") as f:
            persisted = json.load(f)
        assert persisted["dataset"]["subset_strategy"] == "stratified"
        assert os.path.exists(summary_md)
        assert os.path.exists(os.path.join(plots_dir, "metric_grid.png"))
        assert os.path.exists(os.path.join(plots_dir, "accuracy_latency_tradeoff.png"))
        assert os.path.exists(os.path.join(plots_dir, "per_shape_joint_rmse_heatmap.png"))
        assert os.path.exists(os.path.join(plots_dir, "generative_oracle_gap.png"))
