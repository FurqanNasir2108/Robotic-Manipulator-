# Phase 1: Foundation and Simulation

> **Prerequisite:** Read `00_main_orchestrator.instructions.md` first.

---

## Goal

Set up the repository structure, implement core 3-link manipulator simulation, and generate trajectory datasets.

---

## Tasks

### 1.1 Environment Setup
- Create `requirements.txt` with pinned versions:
  ```
  torch>=2.0
  numpy>=1.24
  scipy>=1.10
  matplotlib>=3.7
  seaborn>=0.12
  pyyaml>=6.0
  h5py>=3.8
  tensorboard>=2.13
  pytest>=7.3
  tqdm>=4.65
  ```
- Create `environment.yml` as conda alternative
- Create `setup.py` / `pyproject.toml` making `src/` installable via `pip install -e .`
- Create `.gitignore` (Python, PyTorch, data, checkpoints, IDE files)
- Initialize all `__init__.py` files

### 1.2 3-Link Manipulator Kinematics
- **Primary (no external deps):** Pure-Python implementation using DH parameters
  - Forward kinematics: `(q1, q2, q3)` → `(x, y, theta)`
  - Inverse kinematics: `(x, y, theta)` → `(q1, q2, q3)` with all solution branches (elbow-up/down)
  - Jacobian computation for velocity/force mapping
  - Joint limit checking
- **Fallback:** PyBullet wrapper in `physics_engine.py` for dynamics/collision if needed later
- Robot parameters must be configurable via `configs/simulation.yaml`:
  ```yaml
  link_lengths: [1.0, 1.0, 0.5]
  joint_limits:
    q1: [-pi, pi]
    q2: [-pi, pi]
    q3: [-pi, pi]
  base_position: [0.0, 0.0]
  ```

### 1.3 Trajectory Generation
Implement trajectory generators for:
- Circle (configurable center, radius)
- Square (configurable center, side length)
- Pentagon (configurable center, radius)
- Hexagon (configurable center, radius)
- Straight line (start point, end point)
- Random smooth curves (Bezier or spline-based)

Each generator must:
- Output `N` waypoints evenly spaced in time
- Return both task-space `(x, y, theta)` and joint-space `(q1, q2, q3)` sequences
- Support configurable time horizon and resolution
- Handle multiple IK solutions (prefer lowest energy)

### 1.4 Dataset Construction
- Store datasets as `.npz` files with structure:
  ```python
  {
      "sample_id": int,
      "shape_type": str,
      "start_pose": (x, y, theta),
      "goal_pose": (x, y, theta),
      "waypoints": (N, 3),         # task-space
      "q_sequence": (N, 3),        # joint-space
      "x_sequence": (N,),
      "y_sequence": (N,),
      "theta_sequence": (N,),
      "dq_sequence": (N, 3),       # joint velocities
      "ddq_sequence": (N, 3),      # joint accelerations
      "energy_cost": float,
      "smoothness_cost": float,
  }
  ```
- Create `configs/data.yaml` controlling:
  - Number of samples per shape
  - Time steps per trajectory
  - Train/val/test split ratios
  - Random seed
  - Normalization strategy
- Reserve at least one shape entirely for generalization testing (e.g., hexagon for test-only)

### 1.5 Data Normalization
- Implement min-max and z-score normalization
- Save normalization statistics with the dataset (in `data/metadata/`)
- All downstream models load normalization stats from metadata

---

## Deliverables

- [ ] Working FK/IK for 3-link arm with tests passing
- [ ] Trajectory generators for all 6 shape types
- [ ] Dataset generation script (`scripts/generate_data.py`)
- [ ] At least 5000 samples per shape in train set
- [ ] `README.md` with project summary and setup instructions
- [ ] Visualization of sample trajectories in `figures/simulation/`

---

## Go/No-Go Checks

| Check | Action if Failed |
|-------|-----------------|
| FK(IK(pose)) ≠ pose | Debug IK solver; check quadrant handling and joint limits |
| Trajectories have discontinuities | Verify IK branch consistency along trajectory; use closest-solution tracking |
| Dataset files corrupt or inconsistent | Add schema validation in `src/data/dataset.py` |
| Normalization produces NaN/Inf | Clamp extreme values; check for zero-variance features |

---

## Fallbacks

- If PyBullet/MuJoCo are needed later, the `physics_engine.py` wrapper is ready but **not required for Phase 1**
- If random Bezier curves produce infeasible poses, filter by workspace reachability before saving
- If IK has no solution for a waypoint, log a warning and skip that sample (do not silently produce garbage)
