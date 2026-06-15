"""Evaluate trained models and write comparison summaries."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluate.comparisons import (
	evaluate_model_set,
	load_model_specs,
	load_test_dataset,
	plot_main_comparison,
	plot_trajectory_overlay,
	write_summary_files,
)
from src.utils.config import load_config
from scripts.plot_results import generate_result_plots


def main():
	parser = argparse.ArgumentParser(description='Evaluate trained models')
	parser.add_argument('--config', type=str, default='configs/evaluation.yaml')
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--max-samples', type=int, default=None)
	parser.add_argument('--subset-strategy', type=str, default=None,
						choices=['stratified', 'random', 'ordered'])
	args = parser.parse_args()

	cfg = load_config(args.config)
	max_samples = args.max_samples
	if max_samples is None:
		max_samples = cfg['evaluation'].get('max_test_samples')
	subset_strategy = args.subset_strategy
	if subset_strategy is None:
		subset_strategy = cfg['evaluation'].get('subset_strategy', 'stratified')

	dataset, normalizer = load_test_dataset(
		cfg['paths']['data_dir'],
		cfg['paths']['metadata_dir'],
		max_samples=max_samples,
		seed=cfg['evaluation'].get('seed', 42),
		subset_strategy=subset_strategy,
	)

	include = list(cfg['models']['include'])
	model_specs = load_model_specs(
		device=args.device,
		include=include,
		include_diffusion_ddpm=cfg['models'].get('include_diffusion_ddpm', False),
	)

	summary = evaluate_model_set(model_specs, dataset, normalizer, cfg)

	write_summary_files(
		summary,
		cfg['paths']['summary_json'],
		cfg['paths']['summary_md'],
	)
	plot_main_comparison(summary, cfg['paths']['comparison_figure'])
	plot_trajectory_overlay(summary, cfg['paths']['overlay_figure'])

	plots_dir = cfg['paths']['plots_dir']
	os.makedirs(plots_dir, exist_ok=True)
	plot_main_comparison(summary, os.path.join(plots_dir, 'main_comparison.png'))
	plot_trajectory_overlay(summary, os.path.join(plots_dir, 'trajectory_overlay.png'))
	generate_result_plots(summary, plots_dir)

	print(f"Evaluated {len(model_specs)} models on {summary['num_test_samples']} test samples")
	print(f"Summary JSON: {cfg['paths']['summary_json']}")
	print(f"Summary Markdown: {cfg['paths']['summary_md']}")


if __name__ == '__main__':
	main()
