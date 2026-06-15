"""
Dataset generation script for 3-link manipulator trajectories.

Reads configs/data.yaml, generates trajectory samples for each shape type,
splits into train/val/test, saves .npz files, and computes normalization statistics.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --config configs/data.yaml
"""

import os
import sys
import argparse
import logging

import numpy as np
import yaml
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.trajectory_generator import TrajectoryGenerator
from src.data.dataset import validate_sample, save_dataset
from src.data.normalization import Normalizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load the data generation config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def sample_shape_params(shape_type, shape_params_cfg, rng):
    """
    Randomly sample generation parameters for a given shape type from the config ranges.

    Parameters
    ----------
    shape_type : str
    shape_params_cfg : dict
    rng : np.random.Generator

    Returns
    -------
    dict of kwargs for the trajectory generator
    """
    cfg = shape_params_cfg[shape_type]
    kwargs = {}

    if shape_type == 'circle':
        cx, cy = cfg['center']
        cx += rng.uniform(-0.3, 0.3)
        cy += rng.uniform(-0.3, 0.3)
        kwargs['center'] = (cx, cy)
        kwargs['radius'] = rng.uniform(*cfg['radius'])
        kwargs['theta_orient'] = rng.uniform(*cfg['theta_orient'])

    elif shape_type == 'square':
        cx, cy = cfg['center']
        cx += rng.uniform(-0.3, 0.3)
        cy += rng.uniform(-0.3, 0.3)
        kwargs['center'] = (cx, cy)
        kwargs['side_length'] = rng.uniform(*cfg['side_length'])
        kwargs['theta_orient'] = rng.uniform(*cfg['theta_orient'])

    elif shape_type in ('pentagon', 'hexagon'):
        cx, cy = cfg['center']
        cx += rng.uniform(-0.3, 0.3)
        cy += rng.uniform(-0.3, 0.3)
        kwargs['center'] = (cx, cy)
        kwargs['radius'] = rng.uniform(*cfg['radius'])
        kwargs['theta_orient'] = rng.uniform(*cfg['theta_orient'])

    elif shape_type == 'line':
        x1 = rng.uniform(*cfg['x_range'])
        y1 = rng.uniform(*cfg['y_range'])
        x2 = rng.uniform(*cfg['x_range'])
        y2 = rng.uniform(*cfg['y_range'])
        kwargs['start'] = (x1, y1)
        kwargs['end'] = (x2, y2)
        kwargs['theta_orient'] = rng.uniform(*cfg['theta_orient'])

    elif shape_type == 'random_smooth':
        kwargs['workspace_center'] = tuple(cfg['workspace_center'])
        kwargs['workspace_radius'] = cfg['workspace_radius']
        kwargs['num_control_points'] = rng.integers(cfg['num_control_points'][0],
                                                     cfg['num_control_points'][1] + 1)
        kwargs['theta_orient'] = rng.uniform(*cfg['theta_orient'])
        kwargs['rng'] = rng

    return kwargs


def generate_samples(shape_type, num_samples, gen, shape_params_cfg, rng, start_id=0):
    """
    Generate trajectory samples for a given shape type.

    Returns
    -------
    list of dict
        Valid trajectory samples.
    """
    samples = []
    attempts = 0
    max_attempts = num_samples * 3  # Allow retries for infeasible trajectories

    pbar = tqdm(total=num_samples, desc=f"  {shape_type}", leave=False)
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        kwargs = sample_shape_params(shape_type, shape_params_cfg, rng)
        result = gen.generate(shape_type, **kwargs)
        if result is None:
            continue
        sample = {
            'sample_id': start_id + len(samples),
            **result,
        }
        if validate_sample(sample):
            samples.append(sample)
            pbar.update(1)
    pbar.close()

    if len(samples) < num_samples:
        logger.warning(f"Only generated {len(samples)}/{num_samples} samples for '{shape_type}' "
                       f"after {attempts} attempts")
    return samples


def split_samples(samples, ratios, rng):
    """
    Split samples into train/val/test sets.

    Parameters
    ----------
    samples : list of dict
    ratios : dict with keys 'train', 'val', 'test'
    rng : np.random.Generator

    Returns
    -------
    dict with keys 'train', 'val', 'test'
    """
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_train = int(len(samples) * ratios['train'])
    n_val = int(len(samples) * ratios['val'])
    return {
        'train': [samples[i] for i in indices[:n_train]],
        'val': [samples[i] for i in indices[n_train:n_train + n_val]],
        'test': [samples[i] for i in indices[n_train + n_val:]],
    }


def main():
    parser = argparse.ArgumentParser(description='Generate trajectory datasets')
    parser.add_argument('--config', default='configs/data.yaml', help='Path to data config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg['random_seed'])
    gen = TrajectoryGenerator(num_waypoints=cfg['num_waypoints'])

    output_dir = cfg['output_dir']
    metadata_dir = cfg['metadata_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    all_train = []
    all_val = []
    all_test = []
    sample_id_counter = 0

    # Generate samples for standard shapes (train/val/test split)
    logger.info("Generating samples for standard shapes...")
    for shape_type, num_samples in cfg['samples_per_shape'].items():
        logger.info(f"Shape: {shape_type} ({num_samples} samples)")
        samples = generate_samples(shape_type, num_samples, gen,
                                   cfg['shape_params'], rng, start_id=sample_id_counter)
        sample_id_counter += len(samples)
        splits = split_samples(samples, cfg['split_ratios'], rng)
        all_train.extend(splits['train'])
        all_val.extend(splits['val'])
        all_test.extend(splits['test'])

    # Generate test-only shapes (e.g., hexagon for generalization)
    logger.info("Generating test-only shapes...")
    for shape_type, num_samples in cfg['test_only_shapes'].items():
        logger.info(f"Shape (test-only): {shape_type} ({num_samples} samples)")
        samples = generate_samples(shape_type, num_samples, gen,
                                   cfg['shape_params'], rng, start_id=sample_id_counter)
        sample_id_counter += len(samples)
        all_test.extend(samples)

    # Save datasets
    logger.info(f"Train: {len(all_train)}, Val: {len(all_val)}, Test: {len(all_test)}")
    save_dataset(all_train, os.path.join(output_dir, 'train.npz'))
    save_dataset(all_val, os.path.join(output_dir, 'val.npz'))
    save_dataset(all_test, os.path.join(output_dir, 'test.npz'))

    # Compute and save normalization statistics from training set
    logger.info("Computing normalization statistics...")
    normalizer = Normalizer(strategy=cfg['normalization'])
    train_q = np.array([s['q_sequence'] for s in all_train])
    train_wp = np.array([s['waypoints'] for s in all_train])
    train_dq = np.array([s['dq_sequence'] for s in all_train])
    train_ddq = np.array([s['ddq_sequence'] for s in all_train])

    normalizer.fit(train_q, 'q_sequence')
    normalizer.fit(train_wp, 'waypoints')
    normalizer.fit(train_dq, 'dq_sequence')
    normalizer.fit(train_ddq, 'ddq_sequence')
    normalizer.save(os.path.join(metadata_dir, 'normalization_stats.json'))

    logger.info("Dataset generation complete.")


if __name__ == '__main__':
    main()