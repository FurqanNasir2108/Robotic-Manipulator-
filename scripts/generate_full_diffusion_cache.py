"""Generate the full 4,500-sample diffusion cache with batched DDIM sampling."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate.comparisons import load_model_specs, load_test_dataset
from src.utils.config import load_config


def main():
    cfg = load_config("configs/evaluation.yaml")
    dataset, normalizer = load_test_dataset(
        cfg["paths"]["data_dir"],
        cfg["paths"]["metadata_dir"],
        max_samples=None,
        seed=cfg["evaluation"].get("seed", 42),
        subset_strategy="stratified",
    )
    spec = load_model_specs(
        device="cpu",
        include=["diffusion_ddim"],
        include_diffusion_ddpm=False,
    )["diffusion_ddim"]
    model = spec["predict"].__defaults__[1]

    num_conditions = len(dataset)
    num_samples = cfg["evaluation"].get("generative_samples", 10)
    seq_len = cfg["sampling"].get("diffusion_seq_len", 100)
    ddim_steps = cfg["sampling"].get("diffusion_ddim_steps", 50)
    guidance_scale = cfg["sampling"].get("diffusion_guidance_scale", 2.0)
    batch_size = 8

    all_predictions = np.empty((num_conditions, num_samples, seq_len, 3), dtype=np.float32)
    all_oracle = np.empty((num_conditions, seq_len, 3), dtype=np.float32)
    all_reference = np.empty((num_conditions, seq_len, 3), dtype=np.float32)
    all_waypoints = np.empty((num_conditions, seq_len, 3), dtype=np.float32)
    all_shapes = np.asarray(dataset.shape_type)

    start = time.time()
    for batch_start in range(0, num_conditions, batch_size):
        batch_end = min(batch_start + batch_size, num_conditions)
        batch_conditions = []
        batch_targets = []
        batch_waypoints = []
        for index in range(batch_start, batch_end):
            condition_norm, target_norm = dataset[index]
            condition_norm = condition_norm.numpy()
            target_norm = target_norm.numpy()
            batch_conditions.append(condition_norm)
            batch_targets.append(normalizer.inverse_transform(target_norm, "q_sequence").astype(np.float32))
            batch_waypoints.append(normalizer.inverse_transform(condition_norm, "waypoints").astype(np.float32))

        condition_array = np.stack(batch_conditions, axis=0)
        target_array = np.stack(batch_targets, axis=0)
        waypoint_array = np.stack(batch_waypoints, axis=0)

        with torch.no_grad():
            generated = model.generate(
                torch.from_numpy(condition_array).to(dtype=torch.float32),
                num_samples=num_samples,
                method="ddim",
                num_steps=ddim_steps,
                guidance_scale=guidance_scale,
                seq_len=seq_len,
            ).cpu().numpy()

        generated = generated.reshape(batch_end - batch_start, num_samples, seq_len, 3)
        generated = normalizer.inverse_transform(generated, "q_sequence").astype(np.float32)

        rmse = np.sqrt(np.mean((generated - target_array[:, None, :, :]) ** 2, axis=(2, 3)))
        oracle_indices = np.argmin(rmse, axis=1)

        all_predictions[batch_start:batch_end] = generated
        all_reference[batch_start:batch_end] = target_array
        all_waypoints[batch_start:batch_end] = waypoint_array
        for row, oracle_idx in enumerate(oracle_indices):
            all_oracle[batch_start + row] = generated[row, oracle_idx]

        elapsed = time.time() - start
        processed = batch_end
        rate = processed / max(elapsed, 1e-6)
        remaining = (num_conditions - processed) / max(rate, 1e-6)
        print(
            f"Processed {processed}/{num_conditions} conditions "
            f"({elapsed / 60.0:.2f} min elapsed, {remaining / 60.0:.2f} min remaining)",
            flush=True,
        )

    output_path = os.path.join("results", "generated_trajectories", "diffusion_ddim.npz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        q_predicted=all_predictions,
        q_oracle_predicted=all_oracle,
        q_reference=all_reference,
        shape_type=all_shapes,
        waypoints=all_waypoints,
    )
    print(f"Saved full diffusion cache to {output_path}", flush=True)


if __name__ == "__main__":
    main()
