# GitHub Repo Recommendations

This file captures the recommended GitHub-side settings for the repository.

## Suggested Repository Description

Research codebase for trajectory generation, evaluation, and edge deployment in a 3-link planar robotic manipulator using CNN baselines, conditional VAE, and diffusion models.

## Suggested Topics

Use a focused set of topics such as:

- `robotics`
- `robot-manipulator`
- `trajectory-generation`
- `inverse-kinematics`
- `pytorch`
- `variational-autoencoder`
- `diffusion-model`
- `onnx`
- `edge-deployment`
- `research-code`

## Recommended Branch Protection for `main`

Enable protection on `main` with these settings:

1. Require a pull request before merging.
2. Require at least 1 approval.
3. Dismiss stale approvals when new commits are pushed.
4. Require conversation resolution before merge.
5. Block force pushes.
6. Block branch deletion.
7. Prefer squash merge or rebase merge to keep history clean.

## Status Checks

Do not require status checks until a CI workflow is in place and verified.

Once CI exists, require at least:

- lint or formatting checks, if added later
- `python -m pytest`

## Release Hygiene

For milestone tracking, consider tagging major repository states such as:

- `v0.1-data-and-baselines`
- `v0.2-cvae`
- `v0.3-diffusion`
- `v1.0-thesis-submission`

## License Reminder

Before advertising the repo publicly for reuse, add an explicit license file. That choice has legal consequences, so it should be selected deliberately rather than guessed.
