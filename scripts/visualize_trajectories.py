"""Generate sample trajectory visualizations for all shape types."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.trajectory_generator import TrajectoryGenerator
from src.simulation.visualization import plot_sample_trajectories

if __name__ == '__main__':
    gen = TrajectoryGenerator(num_waypoints=100)
    plot_sample_trajectories(gen, save_dir='figures/simulation')
