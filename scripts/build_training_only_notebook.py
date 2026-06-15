import json
import textwrap
from pathlib import Path


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Neural Network Training Notebook

        This notebook is intentionally focused on **training only**.

        It does not run evaluation or plotting. Use it to launch training for:

        - `cnn`
        - `cnn_lstm`
        - `cnn_gru`
        - `cvae`
        - `diffusion`

        The notebook calls the existing training scripts so the workflow stays aligned with the repo.
        """
    ),
    code(
        """
        %pip install -q -r requirements.txt
        """
    ),
    code(
        """
        import os
        import shlex
        import subprocess
        import sys
        import time
        from pathlib import Path

        REPO_ROOT = Path.cwd()
        PYTHON = sys.executable

        TRAIN_BASELINES = False
        BASELINE_MODELS = ["cnn", "cnn_lstm", "cnn_gru"]
        TRAIN_CVAE = False
        TRAIN_DIFFUSION = False
        FORCE_CPU = False

        ENV = os.environ.copy()
        if FORCE_CPU:
            ENV["CUDA_VISIBLE_DEVICES"] = ""

        print("Python:", PYTHON)
        print("Repo root:", REPO_ROOT)
        print("FORCE_CPU:", FORCE_CPU)
        """
    ),
    code(
        """
        def run_command(args):
            command = " ".join(shlex.quote(str(arg)) for arg in args)
            print(f"\\n>>> {command}\\n")
            start = time.time()
            completed = subprocess.run(
                [str(arg) for arg in args],
                cwd=str(REPO_ROOT),
                env=ENV,
                check=True,
            )
            elapsed = time.time() - start
            print(f"Completed in {elapsed / 60.0:.2f} min")
            return completed
        """
    ),
    code(
        """
        if TRAIN_BASELINES:
            for model_name in BASELINE_MODELS:
                run_command([PYTHON, "scripts/train_baselines.py", "--model", model_name])
        else:
            print("Baseline training skipped. Set TRAIN_BASELINES = True to enable it.")
        """
    ),
    code(
        """
        if TRAIN_CVAE:
            run_command([PYTHON, "scripts/train_vae.py", "--config", "configs/vae.yaml"])
        else:
            print("cVAE training skipped. Set TRAIN_CVAE = True to enable it.")
        """
    ),
    code(
        """
        if TRAIN_DIFFUSION:
            run_command([PYTHON, "scripts/train_diffusion.py", "--config", "configs/diffusion.yaml"])
        else:
            print("Diffusion training skipped. Set TRAIN_DIFFUSION = True to enable it.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path("notebooks/neural_network_training_only.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {output_path}")
