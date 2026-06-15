# Copilot Instruction Pack: Generative Trajectory Generation for a 3-Link Planar Manipulator

This folder is the only documentation Copilot should rely on to build the full repository. It defines the project scope, code structure, phase-wise tasks, implementation rules, and expected outputs.

---

# 1) MAIN INSTRUCTION FILE

## File: `COPILOT_MAIN_INSTRUCTIONS.md`

```md
# Copilot Main Instructions

You are building a full research repository for a master's thesis project titled:

**Generative Models for Trajectory Generation in a 3-Link Planar Robotic Manipulator**

## Project Goal
Create a complete, reproducible, and well-documented research codebase that:
1. Simulates a 3-link planar manipulator.
2. Generates trajectory datasets in joint space and task space.
3. Implements prior supervised baselines already studied by the student.
4. Implements conditional VAE and diffusion-based trajectory generation.
5. Evaluates methods on accuracy, energy, smoothness, diversity, and inference time.
6. Produces figures, logs, results, and thesis/paper-ready outputs.

## Context You Must Preserve
The student has already worked on:
- 3-link planar manipulator trajectory generation
- CNN-based trajectory prediction
- CNN + LSTM trajectory prediction
- CNN + GRU trajectory prediction
- Multi-input CNN / adaptive kernel CNN experiments
- Shape trajectories such as circle, square, pentagon, hexagon
- Input: end-effector pose `(x, y, theta)`
- Output: joint angles `(q1, q2, q3)`
- Comparison against analytical inverse kinematics
- Focus on energy efficiency, accuracy, and execution time

The next stage is generative modeling:
- Conditional VAE for trajectory generation
- Diffusion model for trajectory generation
- Conditioning on waypoints, goal pose, or shape descriptors

## Implementation Rules
- Write clean, modular, reproducible code.
- Use Python.
- Prefer PyTorch for deep learning.
- Keep the code simulation-first and research-friendly.
- Avoid unnecessary complexity.
- Every component must be configurable.
- Save every experiment result in a structured way.
- Use clear names and docstrings.
- Add comments where the logic is non-obvious.
- Support easy thesis and presentation reuse.

## Expected Repo Quality
The repository must include:
- `README.md` with project overview
- `requirements.txt` or `environment.yml`
- source code for simulation, data generation, models, training, evaluation, and plotting
- experiment configs
- saved logs/checkpoints structure
- results/figures folder
- thesis/paper notes
- reusable scripts

## Non-Negotiable Output Requirements
The repo should be able to:
- Generate or load 3-link manipulator trajectory datasets
- Train baseline supervised models
- Train a conditional VAE
- Train a diffusion model
- Evaluate and compare all methods
- Produce plots and summary metrics
- Save artifacts cleanly

## Engineering Priorities
1. Reproducibility
2. Clarity
3. Scientific correctness
4. Modularity
5. Extensibility to higher-DOF manipulators later

## Scientific Priorities
The code should support experiments for:
- trajectory accuracy
- energy usage
- smoothness/jerk
- inference latency
- diversity of generated trajectories
- constraint satisfaction

## Repository Structure to Create
Create a structure like:

```text
project_root/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
├── data/
├── docs/
├── figures/
├── notebooks/
├── results/
├── scripts/
├── src/
├── tests/
└── thesis_notes/
```

## Core Functional Modules
Implement or prepare these modules:
- `src/simulation/`
- `src/data/`
- `src/models/`
- `src/train/`
- `src/evaluate/`
- `src/visualization/`
- `src/utils/`

## Required Model Families
### Baselines
- Analytical IK baseline
- CNN baseline
- CNN + LSTM baseline
- CNN + GRU baseline

### Generative Models
- Conditional VAE
- Diffusion model

## Required Evaluation Metrics
- Joint-space RMSE
- End-effector position/orientation error
- Path tracking error
- Energy proxy or torque-squared cost
- Smoothness / jerk
- Inference time
- Diversity across repeated generations
- Constraint violation rate

## Required Documentation
Generate or maintain docs for:
- problem statement
- data generation process
- model design
- training pipeline
- evaluation protocol
- thesis outline
- paper outline
- experiment log format

## Phase-Wise Development Style
Work in phases:
1. Simulation and data generation
2. Baseline supervised modeling
3. Conditional VAE implementation
4. Diffusion implementation
5. Evaluation and comparison
6. Documentation and thesis/paper packaging

## Important Behavior
- Do not invent results.
- Do not hide missing pieces.
- If something is not implemented yet, mark it clearly.
- Keep the repository usable for future expansion.
- When unsure, choose a simpler and reliable implementation.

## Final Deliverables
The completed repo should allow the student to:
- demonstrate the work in seminars and reviews
- write a master's thesis
- extract a conference paper from the experiments
- extend the work later to higher-DOF manipulators
```

---

# 2) SUPPORTING INSTRUCTION FILES

## File: `docs/01_project_overview.md`

```md
# Project Overview

## Title
Generative Models for Trajectory Generation in a 3-Link Planar Robotic Manipulator

## Research Goal
Build a learning-based trajectory generation pipeline for a 3-link planar manipulator that can generate joint trajectories from task-level conditions such as waypoints, goal pose, and shape descriptors.

## What Has Already Been Done
- Studied the 3-link manipulator setting
- Generated trajectories for geometric shapes
- Used CNN, CNN+LSTM, CNN+GRU, and related supervised models
- Compared predicted trajectories against inverse kinematics baselines
- Evaluated energy, accuracy, and execution time

## What Comes Next
- Conditional VAE for stochastic trajectory generation
- Diffusion model for high-quality generative trajectory synthesis
- Comparative study of baseline vs generative methods
- Thesis and paper preparation

## Thesis Angle
This project should be framed as a deep learning and generative AI thesis for robotics, centered on learning distributions over valid manipulator trajectories instead of relying only on deterministic mappings.
```

---

## File: `docs/02_problem_formulation.md`

```md
# Problem Formulation

## Robot Model
A 3-link planar manipulator operating in 2D workspace with orientation.

## Inputs
Possible conditioning inputs:
- end-effector pose: `(x, y, theta)`
- waypoint sequences
- shape descriptors: circle radius, line endpoints, polygon parameters
- start and goal joint states

## Outputs
- joint trajectory sequence: `(q1(t), q2(t), q3(t))`

## Learning Objective
Given a desired task-level condition, generate a feasible, smooth, energy-efficient joint-space trajectory.

## Key Challenge
This is a one-to-many problem. For the same task condition, multiple valid joint trajectories may exist. Deterministic networks often average solutions. Generative models are better suited for this setting.

## Why Generative Models
- capture multi-modal solutions
- allow sampling diverse trajectories
- support latent-space interpolation
- can be guided toward smoothness and energy efficiency
```

---

## File: `docs/03_data_and_simulation.md`

```md
# Data and Simulation Plan

## Simulation Environment
Use one of the following:
- PyBullet
- MuJoCo

Prefer a lightweight setup for the 3-link planar arm and keep the simulator modular so the code can later extend to higher-DOF manipulators.

## Dataset Construction
Generate trajectories for:
- circle
- square
- pentagon
- hexagon
- straight line
- random smooth curves

For each sample store:
- task condition
- waypoints or goal pose
- time steps
- joint angles `(q1, q2, q3)`
- optional velocities and accelerations
- energy proxy values

## Suggested Sample Structure
```text
sample_id
shape_type
start_pose
goal_pose
waypoints
q_sequence
x_sequence
y_sequence
theta_sequence
energy_cost
smoothness_cost
```

## Dataset Split
Use:
- train
- validation
- test

Keep some shapes or parameter ranges completely unseen during training for generalization tests.

## Data Quality Requirements
- consistent time horizon or padded sequences
- normalized coordinates and angles
- joint limit checks
- reproducible random seeds
- saved metadata for each sample
```

---

## File: `docs/04_baselines.md`

```md
# Baseline Models

## Why Baselines Matter
They provide a reference point for evaluating whether generative models actually improve trajectory quality, diversity, or energy efficiency.

## Required Baselines
1. Analytical inverse kinematics
2. CNN trajectory regressor
3. CNN + LSTM trajectory regressor
4. CNN + GRU trajectory regressor
5. Optional: Transformer regressor

## Baseline Outputs
Each baseline should predict joint trajectories or joint states from task-level input.

## Baseline Metrics
- joint RMSE
- end-effector error
- energy proxy
- smoothness
- inference time

## Documentation Requirement
Each baseline must have:
- model description
- training config
- evaluation results
- example trajectory plots
- comparison table
```

---

## File: `docs/05_vae_method.md`

```md
# Conditional VAE Method

## Purpose
Learn a latent distribution over valid manipulator trajectories.

## Input Conditioning Options
- waypoint sequence
- goal pose `(x, y, theta)`
- shape parameters

## Model Structure
- encoder: condition + trajectory -> latent distribution
- latent sampling: `z ~ N(mu, sigma)`
- decoder: condition + z -> predicted joint trajectory

## Decoder Choices
- LSTM decoder
- GRU decoder
- Transformer decoder

## Loss Terms
1. reconstruction loss on joint trajectory
2. KL divergence
3. optional energy penalty
4. optional smoothness/jerk penalty
5. optional constraint penalties

## Experimental Goals
- reconstruct known trajectories well
- generate diverse but valid trajectories
- compare against deterministic regressors

## Desired Thesis Insight
Show how stochastic latent generation improves flexibility over direct regression.
```

---

## File: `docs/06_diffusion_method.md`

```md
# Diffusion Method

## Purpose
Generate trajectories by learning a denoising process over joint-space sequences.

## Conditioning Options
- waypoint sequence
- goal pose
- shape class
- start state

## Model Structure
- forward process adds noise to clean trajectories
- reverse process learns to denoise iteratively
- denoiser can be a 1D U-Net or Transformer

## Training Objective
- standard diffusion noise prediction loss

## Sampling Strategy
- start from Gaussian noise
- iteratively denoise for a fixed number of steps
- optionally apply guidance toward energy efficiency or smoothness

## Expected Advantages
- better multimodal modeling
- stronger sample quality
- more expressive than deterministic models

## Expected Trade-Off
- slower inference than VAE
- more compute required during sampling
```

---

## File: `docs/07_metrics_and_evaluation.md`

```md
# Evaluation Metrics and Protocol

## Core Metrics
1. Joint-space RMSE
2. End-effector position error
3. End-effector orientation error
4. Path tracking error
5. Energy proxy
6. Smoothness / jerk
7. Inference latency
8. Diversity of generated trajectories
9. Constraint violation rate

## Energy Proxy
Use a consistent proxy such as:
- sum of squared joint velocities
- sum of squared torques
- torque-velocity product if torque is available

## Smoothness Proxy
Use jerk-based or acceleration-based metrics.

## Evaluation Protocol
- train/val/test split
- compare all models on same dataset
- test on unseen shapes and unseen parameters
- report mean and standard deviation
- include visual trajectory comparisons

## Required Figures
- trajectory overlays
- error curves
- loss curves
- metric bar charts
- latent space visualizations for VAE
- sampling quality plots for diffusion
```

---

## File: `docs/08_experiments.md`

```md
# Experiment Plan

## Experiment 1: Baseline Reproduction
Reproduce the earlier supervised models and verify the pipeline works end-to-end.

## Experiment 2: Conditional VAE
Train cVAE on trajectory dataset and test reconstruction + sampling.

## Experiment 3: Diffusion Model
Train diffusion model and test generation quality + diversity.

## Experiment 4: Ablation Studies
- with vs without energy term
- with vs without smoothness term
- different latent sizes
- different diffusion steps
- different conditioning schemes

## Experiment 5: Generalization Tests
- unseen shapes
- altered radii
- different waypoint densities
- longer trajectories

## Experiment 6: Constraint Tests
- joint limits
- path feasibility
- optional obstacle handling
```

---

## File: `docs/09_thesis_outline.md`

```md
# Thesis Outline

## Chapter 1
Introduction, motivation, problem statement, contributions

## Chapter 2
Literature review
- manipulator trajectory generation
- inverse kinematics
- deep learning baselines
- generative models
- diffusion models in robotics

## Chapter 3
Robot model and data generation

## Chapter 4
Baseline supervised models

## Chapter 5
Conditional VAE for trajectory generation

## Chapter 6
Diffusion-based trajectory generation

## Chapter 7
Evaluation and discussion

## Chapter 8
Conclusion and future work

## Thesis Contribution Framing
- learning-based trajectory generation for a 3-link manipulator
- generative AI for multi-solution motion synthesis
- energy-aware comparison between deterministic and generative methods
```

---

## File: `docs/10_paper_outline.md`

```md
# Research Paper Outline

## Possible Paper Title
Generative Trajectory Synthesis for a 3-Link Planar Manipulator Using Conditional VAE and Diffusion Models

## Sections
1. Abstract
2. Introduction
3. Related Work
4. Problem Formulation
5. Methods
6. Experiments
7. Results
8. Discussion
9. Conclusion

## Paper Angle
Focus on one strong novelty:
- either conditional VAE for trajectory generation
- or diffusion-based trajectory generation
- or a direct comparison paper showing why generative models outperform deterministic regressors in diversity and task feasibility

## Recommended Submission Targets
- ICRA
- IROS
- CoRL
- RSS workshop or main conference
- NeurIPS workshop if emphasizing generative modeling
```

---

## File: `docs/11_visualization_and_reporting.md`

```md
# Visualization and Reporting

## Tools
- Matplotlib
- Seaborn
- TensorBoard
- Weights & Biases if needed
- LaTeX / Overleaf for thesis and paper writing

## Required Visual Outputs
- 2D manipulator path plots
- joint angle trajectories over time
- loss curves
- metric comparison plots
- latent space plots for VAE
- diffusion sampling step illustrations
- qualitative success/failure examples

## Reporting Style
- clean figures
- consistent axis labels
- clear legends
- publication-quality outputs
- saved in a dedicated figures directory
```

---

## File: `docs/12_risk_and_limitations.md`

```md
# Risks and Limitations

## Technical Risks
- data may be too simple and models may overfit
- diffusion may be too slow for interactive use
- VAE may produce averaged or blurry trajectories
- energy proxy may not perfectly reflect real hardware energy
- generalization to higher-DOF arms may require redesign

## Mitigation Strategies
- strong baselines
- ablation studies
- unseen-shape testing
- simple but consistent metrics
- modular code so models can be replaced easily

## Thesis Risk Management
If diffusion is too heavy, keep VAE as the primary generative contribution and diffusion as a comparative extension.
```

---

# 3) PHASE-WISE TASK FILE

## File: `docs/13_phase_wise_tasks.md`

```md
# Phase-Wise Tasks for Project Completion

## Phase 1: Foundation and Setup
### Goal
Create the repo structure and simulation backbone.

### Tasks
- set up environment
- define folder structure
- implement 3-link manipulator kinematics
- build trajectory generators for basic shapes
- save datasets in a clean format

### Deliverables
- working simulation
- dataset generation scripts
- README with project summary

### Go/No-Go Check
If joint-space and task-space trajectories are not reproducible, fix simulation before model training.

---

## Phase 2: Baseline Reproduction
### Goal
Reproduce the student’s earlier supervised work.

### Tasks
- implement CNN baseline
- implement CNN+LSTM baseline
- implement CNN+GRU baseline
- evaluate against inverse kinematics baseline
- generate result plots and tables

### Deliverables
- baseline training scripts
- baseline comparison report
- sample trajectory visualizations

### Go/No-Go Check
If baselines do not outperform trivial heuristics, verify data preprocessing and model input format.

---

## Phase 3: Conditional VAE
### Goal
Build the first generative model.

### Tasks
- define conditioning scheme
- implement encoder and decoder
- train cVAE
- tune latent size and loss weights
- evaluate reconstruction and sampling diversity

### Deliverables
- cVAE training and inference code
- latent plots
- quality comparison vs baselines

### Go/No-Go Check
If reconstruction is poor, simplify the decoder and confirm sequence alignment.

---

## Phase 4: Diffusion Model
### Goal
Implement the second generative model.

### Tasks
- implement forward noising process
- implement denoiser network
- train diffusion model
- test different sampling steps
- compare speed and quality

### Deliverables
- diffusion training/inference code
- denoising visualizations
- sampling quality report

### Go/No-Go Check
If inference is too slow, reduce steps or simplify the model.

---

## Phase 5: Evaluation and Comparison
### Goal
Create the thesis-grade comparison.

### Tasks
- evaluate all models on the same split
- compute metrics
- run ablation studies
- compare generalization to unseen shapes
- summarize strengths and failures

### Deliverables
- final result tables
- publication-grade graphs
- discussion notes

### Go/No-Go Check
If comparisons are unclear, standardize the evaluation protocol and rerun experiments.

---

## Phase 6: Thesis and Paper Packaging
### Goal
Convert the work into thesis and paper form.

### Tasks
- write thesis outline
- write paper outline
- prepare figures and tables
- prepare slides
- clean repository and add usage instructions

### Deliverables
- thesis draft structure
- paper draft structure
- presentation assets
- polished repository
```

---

# 4) IMPLEMENTATION TASK FILES

## File: `scripts/README_FOR_COPILOT.md`

```md
# Scripts Guide

Copilot should create scripts for:
- dataset generation
- model training
- model evaluation
- plotting and comparison
- experiment logging

All scripts should be runnable from the command line and use config files where possible.
```

---

## File: `src/README_FOR_COPILOT.md`

```md
# Source Code Guide

## Required Modules
- `src/simulation/`
- `src/data/`
- `src/models/`
- `src/train/`
- `src/evaluate/`
- `src/visualization/`
- `src/utils/`

## Suggested Responsibilities
### simulation
3-link robot kinematics, workspace, trajectory generation, rendering

### data
Dataset creation, normalization, splitting, loaders

### models
CNN, CNN+LSTM, CNN+GRU, cVAE, diffusion models

### train
Training loops, checkpointing, loss tracking

### evaluate
Metrics, comparisons, ablation summaries

### visualization
Trajectory plots, training plots, latent plots, result figures

### utils
Random seeds, config loading, serialization, math helpers
```

---

# 5) REPO TASK CHECKLIST FILE

## File: `docs/14_repo_completion_checklist.md`

```md
# Repo Completion Checklist

- [ ] project structure created
- [ ] 3-link manipulator simulation implemented
- [ ] trajectory datasets generated
- [ ] baseline supervised models implemented
- [ ] conditional VAE implemented
- [ ] diffusion model implemented
- [ ] evaluation metrics implemented
- [ ] comparison tables generated
- [ ] figures exported
- [ ] README written
- [ ] thesis outline documented
- [ ] paper outline documented
- [ ] experiment logs saved
- [ ] code is reproducible
- [ ] final repo is presentation-ready
```

---

# 6) DEFAULT REPO TREE COPILOT SHOULD CREATE

```text
project_root/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   ├── data.yaml
│   ├── baseline.yaml
│   ├── vae.yaml
│   └── diffusion.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── docs/
│   ├── 01_project_overview.md
│   ├── 02_problem_formulation.md
│   ├── 03_data_and_simulation.md
│   ├── 04_baselines.md
│   ├── 05_vae_method.md
│   ├── 06_diffusion_method.md
│   ├── 07_metrics_and_evaluation.md
│   ├── 08_experiments.md
│   ├── 09_thesis_outline.md
│   ├── 10_paper_outline.md
│   ├── 11_visualization_and_reporting.md
│   ├── 12_risk_and_limitations.md
│   ├── 13_phase_wise_tasks.md
│   └── 14_repo_completion_checklist.md
├── figures/
├── notebooks/
├── results/
│   ├── logs/
│   ├── metrics/
│   ├── checkpoints/
│   └── plots/
├── scripts/
│   ├── generate_data.py
│   ├── train_baselines.py
│   ├── train_vae.py
│   ├── train_diffusion.py
│   ├── evaluate_models.py
│   └── plot_results.py
├── src/
│   ├── simulation/
│   ├── data/
│   ├── models/
│   ├── train/
│   ├── evaluate/
│   ├── visualization/
│   └── utils/
└── tests/
```

---

# 7) HOW COPILOT SHOULD USE THESE FILES

1. Read `COPILOT_MAIN_INSTRUCTIONS.md` first.
2. Read the matching `docs/*.md` file before creating each module.
3. Build the repo phase by phase.
4. Keep every implementation aligned with the problem formulation.
5. Do not skip documentation.
6. Do not overwrite the student’s earlier work; extend it.
7. Preserve the 3-link manipulator as the main experimental testbed.

---

# 8) SHORT HUMAN-FACING SUMMARY

This instruction pack is designed to make Copilot behave like a research assistant building a thesis-grade robotics repo from scratch. It preserves the student’s previous supervised trajectory work and extends it into VAE and diffusion-based generative trajectory modeling, with clear phases, metrics, documentation, and experiment structure.

