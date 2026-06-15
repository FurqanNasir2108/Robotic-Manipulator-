# Presentation Guide

This document provides a structured template for creating presentations about this project — for thesis defense, seminar reviews, lab meetings, or conference talks.

---

## Presentation Types

| Type | Duration | Slides | Audience |
|------|----------|--------|----------|
| Thesis Defense | 30–45 min | 35–50 | Committee, faculty |
| Seminar Review | 15–20 min | 15–25 | Lab group, advisor |
| Conference Talk | 12–15 min | 12–18 | Researchers, peers |
| Poster | N/A | 1 poster | Conference attendees |
| Lab Demo | 5–10 min | 5–10 | Visitors, collaborators |

---

## Thesis Defense Slide Structure (35–50 slides)

### Section 1: Introduction (5–7 slides)
1. **Title Slide** — Project title, student name, advisor, date, university
2. **Motivation** — Why trajectory generation matters for robotics
3. **Problem Statement** — One-to-many mapping challenge, limitations of deterministic methods
4. **Research Questions** — Can generative models produce diverse, energy-efficient trajectories?
5. **Contributions** — 3–4 bullet points summarizing novelty
6. **Outline** — Roadmap of the presentation

### Section 2: Background (5–7 slides)
7. **3-Link Planar Manipulator** — Robot diagram, DH parameters, workspace
8. **Trajectory Generation Problem** — Input/output formulation
9. **Supervised Baselines** — CNN, CNN+LSTM, CNN+GRU overview
10. **Generative Models Overview** — VAE, Diffusion, why they suit this problem
11. **Related Work** — Key references with positioning of this work

### Section 3: Robot Model & Data (3–5 slides)
12. **Kinematics** — FK/IK equations, joint limits, workspace boundary
13. **Trajectory Shapes** — Circle, square, pentagon, hexagon, line, random curves
14. **Dataset** — Size, split, sample structure, normalization
15. **Visualization** — Sample trajectories in task space and joint space

### Section 4: Baseline Methods (3–5 slides)
16. **Analytical IK** — Reference solution
17. **CNN Baseline** — Architecture diagram, training setup
18. **CNN+LSTM / CNN+GRU** — Architecture diagrams
19. **Baseline Results** — Table + sample trajectory plots

### Section 5: Conditional VAE (5–7 slides)
20. **VAE Overview** — Encoder-decoder, latent space, reparameterization trick
21. **Conditioning Strategy** — How task conditions are fed to the model
22. **Architecture** — Encoder + decoder diagram
23. **Training** — Loss function, KL annealing, hyperparameters
24. **Results** — Reconstruction quality, diversity, latent space visualization
25. **Comparison vs Baselines** — Key metrics table

### Section 6: Diffusion Model (5–7 slides)
26. **Diffusion Overview** — Forward/reverse process diagram
27. **Conditioning** — Classifier-free guidance
28. **Architecture** — Denoiser network diagram
29. **Training** — Noise prediction loss, training schedule
30. **Sampling** — DDPM vs DDIM vs DPM-Solver comparison
31. **Results** — Sample quality, diversity, denoising visualization
32. **Speed vs Quality** — Inference time comparison

### Section 7: Evaluation (5–7 slides)
33. **Metrics Overview** — All 9 metrics explained
34. **Main Comparison Table** — All models side by side
35. **Ablation Studies** — Key findings (latent size, guidance scale, etc.)
36. **Generalization** — Unseen shapes, altered parameters
37. **Qualitative Comparison** — Best/worst trajectory plots per model
38. **Discussion** — Strengths and limitations of each approach

### Section 8: Conclusion (3–4 slides)
39. **Summary** — Key findings in 3–5 bullet points
40. **Contributions** — Restate thesis contributions
41. **Limitations & Future Work** — Higher-DOF, real hardware, obstacle avoidance
42. **Thank You / Q&A**

---

## Key Figures to Include

| Figure | Source | Slide |
|--------|--------|-------|
| 3-link arm diagram | `figures/simulation/arm_diagram.png` | Slide 7 |
| Workspace boundary | `figures/simulation/workspace.png` | Slide 12 |
| Sample shape trajectories | `figures/simulation/shape_trajectories.png` | Slide 15 |
| Baseline architecture diagrams | Create manually or `figures/baselines/` | Slides 17–18 |
| VAE architecture diagram | Create manually or `figures/vae/architecture.png` | Slide 22 |
| Latent space t-SNE | `figures/vae/latent_tsne.png` | Slide 24 |
| Diffusion denoising steps | `figures/diffusion/denoising_steps.png` | Slide 31 |
| Main comparison table | `results/metrics/summary.md` | Slide 34 |
| Metric bar charts | `figures/comparison/metric_bars.png` | Slide 34 |
| Ablation line plots | `figures/comparison/ablation_*.png` | Slide 35 |
| Trajectory overlays | `figures/comparison/trajectory_overlay.png` | Slide 37 |

---

## Shorter Presentation Variants

### Seminar Review (15–20 min)
- Keep Sections 1, 3, 5 or 6 (whichever is more complete), 7, 8
- Drop detailed background; assume audience knows basics
- Focus on results and discussion

### Conference Talk (12–15 min)
- Title → Problem (1 slide) → Method (3–4 slides) → Results (3–4 slides) → Conclusion (1 slide)
- Focus on the single strongest contribution
- Heavy on figures, light on text

---

## Visual Style Guidelines

- **Font:** Sans-serif (Arial, Helvetica, or Calibri), minimum 18pt for body text
- **Colors:** Consistent color scheme; use colorblind-friendly palette
- **Figures:** All figures at 300 DPI minimum; use vector (SVG/PDF) when possible
- **Equations:** Use LaTeX-rendered equations, not screenshots
- **Animations:** Use sparingly; only for denoising process or trajectory evolution
- **Template:** Use university-provided template if available

---

## How to Generate Presentation Assets

```bash
# Export all figures at presentation quality
python scripts/export_figures.py --dpi 300 --format png

# Generate comparison table as image
python scripts/plot_results.py --output figures/comparison/

# Generate individual model result plots
python scripts/plot_results.py --model vae --output figures/vae/
python scripts/plot_results.py --model diffusion --output figures/diffusion/
```

---

## Preparation Checklist

- [ ] All figures exported at 300 DPI
- [ ] Architecture diagrams created (draw.io, TikZ, or PowerPoint)
- [ ] Results tables finalized
- [ ] Slide text reviewed for clarity
- [ ] Timing practiced (mark check at slide 10, 20, 30)
- [ ] Backup slides prepared for anticipated questions
- [ ] Demo ready (if live demo planned)
