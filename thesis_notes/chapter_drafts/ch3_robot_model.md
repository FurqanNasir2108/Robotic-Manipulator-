# Chapter 3: Robot Model and Data Generation

## 3.1 Introduction

This chapter describes the robotic manipulator used throughout this study, the mathematical formulation of its kinematics, the design of the trajectory dataset, and the data preprocessing pipeline. The 3-link planar manipulator serves as a tractable yet nontrivial testbed for evaluating generative trajectory models, offering a well-defined one-to-many mapping between task-space paths and joint-space trajectories.

---

## 3.2 3-Link Planar Manipulator

### 3.2.1 Physical Description

The robot considered in this work is a 3-link planar revolute manipulator operating in the $xy$-plane. The manipulator consists of three rigid links connected by revolute joints, with the base fixed at the origin $(0, 0)$. The link lengths are:

| Link | Length (m) |
|------|-----------|
| $l_1$ | 1.0 |
| $l_2$ | 1.0 |
| $l_3$ | 0.5 |

Each joint $q_i$ has symmetric limits $q_i \in [-\pi, \pi]$, providing full rotational freedom. The total reach of the manipulator is $l_1 + l_2 + l_3 = 2.5$ m. The configuration is illustrated in Figure 3.1.

> **Figure 3.1:** 3-Link planar manipulator with labeled joints and link lengths.
> See `figures/simulation/arm_diagram.png`

### 3.2.2 Denavit-Hartenberg Parameters

The kinematics are formulated using the standard Denavit-Hartenberg (DH) convention. For a planar manipulator with all revolute joints, the DH parameter table simplifies to:

| Link $i$ | $a_i$ | $\alpha_i$ | $d_i$ | $\theta_i$ |
|-----------|--------|------------|--------|------------|
| 1 | $l_1 = 1.0$ | 0 | 0 | $q_1$ |
| 2 | $l_2 = 1.0$ | 0 | 0 | $q_2$ |
| 3 | $l_3 = 0.5$ | 0 | 0 | $q_3$ |

where $a_i$ is the link length, $\alpha_i$ is the link twist (zero for planar case), $d_i$ is the link offset (zero for revolute joints), and $\theta_i$ is the joint variable.

---

## 3.3 Forward Kinematics

### 3.3.1 Derivation

The forward kinematics maps joint angles $\mathbf{q} = [q_1, q_2, q_3]^T$ to the end-effector pose $\mathbf{p} = [x, y, \theta]^T$ in task space. For the 3-link planar manipulator, the homogeneous transformation matrix for each link is:

$$
T_i = \begin{bmatrix}
\cos(\theta_i) & -\sin(\theta_i) & a_i \cos(\theta_i) \\
\sin(\theta_i) & \cos(\theta_i) & a_i \sin(\theta_i) \\
0 & 0 & 1
\end{bmatrix}
$$

The end-effector position is obtained by composing the individual transformations $T = T_1 T_2 T_3$, yielding:

$$
x = l_1 \cos(q_1) + l_2 \cos(q_1 + q_2) + l_3 \cos(q_1 + q_2 + q_3)
$$

$$
y = l_1 \sin(q_1) + l_2 \sin(q_1 + q_2) + l_3 \sin(q_1 + q_2 + q_3)
$$

$$
\theta = q_1 + q_2 + q_3
$$

where $\theta$ is the end-effector orientation angle measured from the positive $x$-axis.

### 3.3.2 Jacobian Matrix

The geometric Jacobian relates joint velocities to end-effector velocity:

$$
\dot{\mathbf{p}} = J(\mathbf{q}) \dot{\mathbf{q}}
$$

For the 3-link planar manipulator, the Jacobian is a $3 \times 3$ matrix:

$$
J(\mathbf{q}) = \begin{bmatrix}
-l_1 s_1 - l_2 s_{12} - l_3 s_{123} & -l_2 s_{12} - l_3 s_{123} & -l_3 s_{123} \\
l_1 c_1 + l_2 c_{12} + l_3 c_{123} & l_2 c_{12} + l_3 c_{123} & l_3 c_{123} \\
1 & 1 & 1
\end{bmatrix}
$$

where we use the shorthand $s_1 = \sin(q_1)$, $c_1 = \cos(q_1)$, $s_{12} = \sin(q_1 + q_2)$, $c_{12} = \cos(q_1 + q_2)$, $s_{123} = \sin(q_1 + q_2 + q_3)$, and $c_{123} = \cos(q_1 + q_2 + q_3)$.

The Jacobian is used to compute joint velocities $\dot{\mathbf{q}}$ and accelerations $\ddot{\mathbf{q}}$ from task-space trajectory derivatives via numerical differentiation.

---

## 3.4 Inverse Kinematics

### 3.4.1 Analytical Solution

The inverse kinematics (IK) problem solves for joint angles $\mathbf{q}$ given a desired end-effector pose $\mathbf{p} = [x, y, \theta]^T$. Since the end-effector orientation $\theta = q_1 + q_2 + q_3$ provides a constraint, the problem can be decomposed into a 2-link subproblem.

**Step 1: Wrist position.** Given the desired end-effector pose, compute the wrist position (joint 2 to joint 3 intersection):

$$
w_x = x - l_3 \cos(\theta), \quad w_y = y - l_3 \sin(\theta)
$$

**Step 2: 2-link IK.** The wrist position $(w_x, w_y)$ must be reached by the first two links ($l_1, l_2$). Using the cosine rule:

$$
D = \frac{w_x^2 + w_y^2 - l_1^2 - l_2^2}{2 l_1 l_2}
$$

If $|D| > 1$, the wrist position is unreachable. Otherwise:

$$
q_2 = \text{atan2}\left(\pm\sqrt{1 - D^2},\ D\right)
$$

The $\pm$ sign yields two solution branches — **elbow-up** ($+$) and **elbow-down** ($-$).

**Step 3: Solve for $q_1$ and $q_3$:**

$$
q_1 = \text{atan2}(w_y, w_x) - \text{atan2}(l_2 \sin(q_2),\ l_1 + l_2 \cos(q_2))
$$

$$
q_3 = \theta - q_1 - q_2
$$

### 3.4.2 Solution Multiplicity

A key property of this manipulator is that for most reachable poses, there exist **two valid IK solutions** corresponding to elbow-up and elbow-down configurations. This one-to-many mapping is fundamental to the motivation of this thesis: given a task-space trajectory, there are exponentially many valid joint-space trajectories (choosing between branches at each waypoint). This motivates the use of generative models that can capture and sample from the distribution of valid trajectories, rather than deterministic models that produce only a single mapping.

> **Figure 3.2:** Elbow-up and elbow-down configurations for the same end-effector pose.
> See `figures/simulation/ik_branches.png`

### 3.4.3 Joint Limit Checking

After computing IK solutions, each joint angle is verified against the configured limits $q_i \in [-\pi, \pi]$. Solutions violating any joint limit are discarded. The implementation wraps angles to $[-\pi, \pi]$ before checking.

---

## 3.5 Workspace Analysis

The workspace of the manipulator is the set of all reachable end-effector positions. For the 3-link planar manipulator with full joint rotation, the workspace is an annular region (or full disk when the minimum reach approaches zero).

- **Maximum reach:** $r_{\max} = l_1 + l_2 + l_3 = 2.5$ m
- **Minimum reach:** $r_{\min} = |l_1 - l_2| - l_3 = |1.0 - 1.0| - 0.5 = -0.5$ m → effectively 0

Since $r_{\min} \leq 0$, the workspace includes the origin and forms a full disk of radius 2.5 m. This is confirmed by random sampling of 50,000 joint configurations, shown in Figure 3.3.

> **Figure 3.3:** Workspace of the 3-link planar manipulator obtained by random joint sampling.
> See `figures/simulation/workspace.png`

---

## 3.6 Trajectory Design

### 3.6.1 Shape Types

Six distinct trajectory shape types are defined in task space to ensure diversity in the training data:

| Shape | Parameters | Description |
|-------|-----------|-------------|
| **Circle** | center, radius | Closed circular path |
| **Square** | center, side length | Closed rectangular path with 4 vertices |
| **Pentagon** | center, radius | Closed regular polygon with 5 vertices |
| **Hexagon** | center, radius | Closed regular polygon with 6 vertices (test only) |
| **Line** | start point, end point | Open linear trajectory |
| **Random Smooth** | control points, spline | Smooth open curve via cubic B-spline |

Each shape is parameterized with randomized parameters (e.g., varying center, radius, orientation) to ensure variability within each shape class. All shapes are generated within the reachable workspace centered approximately at $(1.5, 0.0)$.

> **Figure 3.4:** The six trajectory shape types in task space.
> See `figures/simulation/shape_trajectories_grid.png`

### 3.6.2 Trajectory Representation

Each trajectory sample consists of:

| Field | Shape | Description |
|-------|-------|-------------|
| `waypoints` | $(T, 3)$ | Task-space poses $(x, y, \theta)$ at each timestep |
| `q_sequence` | $(T, 3)$ | Joint angle trajectory $[q_1, q_2, q_3]$ |
| `dq_sequence` | $(T, 3)$ | Joint velocity (numerical derivative) |
| `ddq_sequence` | $(T, 3)$ | Joint acceleration (second derivative) |
| `shape_type` | string | Shape label |

where $T = 100$ is the number of waypoints per trajectory.

### 3.6.3 Joint Trajectory Resolution

Converting a task-space trajectory to a joint-space trajectory requires solving the IK at each waypoint. Since the IK is multi-valued, we adopt a **closest-solution tracking** strategy: at each waypoint, we evaluate all valid IK solutions and select the one closest (in joint-space Euclidean distance) to the previous waypoint's solution. This produces smooth, branch-consistent joint trajectories without discontinuous jumps.

---

## 3.7 Dataset Construction

### 3.7.1 Generation Parameters

The dataset is generated using the following configuration:

| Parameter | Value |
|-----------|-------|
| Samples per shape (train) | 5,000 |
| Training shapes | Circle, Square, Pentagon, Line, Random Smooth |
| Test-only shape | Hexagon (2,000 samples) |
| Waypoints per trajectory | 100 |
| Random seed | 42 |

### 3.7.2 Train/Validation/Test Split

The data is split as follows:

| Split | Ratio | Total Samples | Source |
|-------|-------|---------------|--------|
| Train | 80% | 20,000 | 5 shapes × 5,000 × 0.8 |
| Validation | 10% | 2,500 | 5 shapes × 5,000 × 0.1 |
| Test | 10% | 2,500 | 5 shapes × 5,000 × 0.1 |
| Test (Hexagon) | 100% | 2,000 | Hexagon only |

The hexagon shape is reserved exclusively for generalization testing — it does not appear in the training or validation sets. This evaluates the model's ability to generalize to unseen geometric patterns.

### 3.7.3 Data Quality Checks

Each generated sample undergoes validation:
- Joint angles within limits
- Forward kinematics consistency (reconstructed waypoints match original within tolerance)
- Finite values (no NaN or Inf)
- Correct dimensions

Samples failing validation are discarded and regenerated.

---

## 3.8 Data Normalization

### 3.8.1 Z-Score Normalization

All trajectory data is normalized using z-score standardization:

$$
\hat{x} = \frac{x - \mu}{\sigma}
$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed from the **training set only**. The same statistics are applied to the validation and test sets to prevent data leakage.

Normalization is applied independently to each feature dimension (e.g., $q_1$, $q_2$, $q_3$). The statistics are saved in `data/metadata/normalization_stats.json` for use during inference.

### 3.8.2 Motivation

Z-score normalization is preferred over min-max normalization for this application because:
1. Joint angle distributions are roughly symmetric around zero
2. It preserves the relative scale of angular quantities
3. It is robust to outliers in the random smooth trajectories
4. Neural network training generally converges faster with zero-mean, unit-variance inputs

---

## 3.9 Summary

This chapter presented the 3-link planar manipulator model, including its forward and inverse kinematics, workspace analysis, and the trajectory dataset construction pipeline. The key design choices — closest-solution tracking for IK, hexagon shape holdout for generalization, and z-score normalization — are justified by the requirements of the learning-based trajectory generation methods described in subsequent chapters.

> **Figure 3.5:** Dataset distribution showing samples per shape and joint angle histograms.
> See `figures/simulation/dataset_distribution.png`

**Key figures for this chapter:**
| Figure | File |
|--------|------|
| 3.1 Arm Diagram | `figures/simulation/arm_diagram.png` |
| 3.2 IK Branches | `figures/simulation/ik_branches.png` |
| 3.3 Workspace | `figures/simulation/workspace.png` |
| 3.4 Shape Types | `figures/simulation/shape_trajectories_grid.png` |
| 3.5 Dataset Distribution | `figures/simulation/dataset_distribution.png` |
| 3.6 Circle Trajectory | `figures/simulation/circle_trajectory.png` |
| 3.7 Circle Joint Angles | `figures/simulation/circle_joints.png` |
