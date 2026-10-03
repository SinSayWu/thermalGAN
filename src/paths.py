"""Central path configuration for the thermal GAN code.

Every script builds its file locations from this module. The defaults are folders
inside the repository; override any of them with an environment variable, e.g.

    export THERMAL_GAN_DATA=/path/to/my/datasets
    export THERMAL_GAN_RUNS=/scratch/me/thermal_gan_runs

Layout expected under each location:

    THERMAL_GAN_DATA/simdata{1,2,4,8}/          FEM datasets (thermal_XXXX.npz, struct_XXXX.png/.npy)
    THERMAL_GAN_RUNS/training_outputs_strongD_smoothgen08{N}/   training checkpoints + logs
    THERMAL_GAN_WEIGHTS/smoothgen08{N}/G_<epoch>.pth            released generators
    THERMAL_GAN_OUTPUTS/<script name>/                         evaluation results and figures
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(var, default):
    return Path(os.environ.get(var, default)).expanduser().resolve()


DATA_DIR = _env_path("THERMAL_GAN_DATA", REPO_ROOT / "data")
RUNS_DIR = _env_path("THERMAL_GAN_RUNS", REPO_ROOT / "runs")
WEIGHTS_DIR = _env_path("THERMAL_GAN_WEIGHTS", REPO_ROOT / "weights")
OUTPUT_DIR = _env_path("THERMAL_GAN_OUTPUTS", REPO_ROOT / "outputs")

# Released one-shot generators used in the paper: pore count -> (folder, checkpoint).
KEEPERS = {1: ("smoothgen081", "G_540"), 2: ("smoothgen082", "G_420"),
           4: ("smoothgen084", "G_240"), 8: ("smoothgen088", "G_180")}


def simdata_dir(n_pores):
    """Dataset folder for an n-pore microstructure set."""
    return DATA_DIR / f"simdata{n_pores}"


def run_dir(n_pores):
    """Training-output folder (checkpoints every 10 epochs) for an n-pore run."""
    return RUNS_DIR / f"training_outputs_strongD_smoothgen08{n_pores}"


def keeper_path(n_pores):
    """Path to the released generator weights for an n-pore model."""
    folder, ckpt = KEEPERS[int(n_pores)]
    return WEIGHTS_DIR / folder / f"{ckpt}.pth"


def output_dir(name):
    """Create and return OUTPUT_DIR/<name>."""
    d = OUTPUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d
