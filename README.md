# FEM-Validated Generative Inverse Design of Thermally Anisotropic Microstructures

Code and trained models for a regression-based conditional Wasserstein GAN with gradient
penalty (RCWGAN-GP) that generates 128 × 128 porous microstructures from target directional
thermal conductivities (k<sub>x</sub>, k<sub>y</sub>). Generated designs are checked by
re-simulating them with the finite element method (FEniCS). One model is trained per pore
count (1, 2, 4 and 8 elliptical pores).

## Repository layout

```
data_generation/thermal_data_generation.ipynb   FEM dataset generation (pore placement + FEniCS)
src/ResDCGAN_GP_thermal_strongD_smoothgen08.py   RCWGAN-GP model and training
src/certify_all_pores.py                         closed-loop FEM validation + geometry metrics
src/eval_diversity.py                            within-label diversity (Hamming distance)
src/screen_smoothgen08.py                        one-shot checkpoint screening (keeper selection)
src/_strongd_lib.py                              data loaders and FEM conductivity solver
src/geometry_metrics.py                          pore count, ellipticity, orientation metrics
src/_orig_big_gen.py                             large-capacity generator (ablation only)
src/paths.py                                     all file locations (see "Paths" below)
jobs/                                            SLURM scripts for training and validation
weights/smoothgen08{1,2,4,8}/                    trained generators used in the paper
results/all_pores_metrics.json                   validation results for the released models
```

## Paths

All locations come from `src/paths.py`. By default everything lives inside the repository;
override any location with an environment variable:

| Variable | Default | Contents |
|---|---|---|
| `THERMAL_GAN_DATA` | `data/` | datasets `simdata1/`, `simdata2/`, `simdata4/`, `simdata8/` |
| `THERMAL_GAN_RUNS` | `runs/` | training checkpoints (`training_outputs_strongD_smoothgen08<N>/`) |
| `THERMAL_GAN_WEIGHTS` | `weights/` | released generators |
| `THERMAL_GAN_OUTPUTS` | `outputs/` | evaluation results and figures |

`PORES` (1, 2, 4 or 8, default 8) selects the pore count for data generation, training and
checkpoint screening.

## Usage

The FEM steps need legacy FEniCS (`dolfin`) and `mshr`; the GAN needs PyTorch. Other
dependencies: numpy, scipy, pandas, matplotlib, scikit-image, opencv-python, Pillow, torchvision.

1. **Generate a dataset.** Open `data_generation/thermal_data_generation.ipynb` with `PORES`
   set and run it. Samples are written to `THERMAL_GAN_DATA/simdata<PORES>/`.
2. **Train.** `PORES=8 python src/ResDCGAN_GP_thermal_strongD_smoothgen08.py`
   (or `sbatch --export=ALL,PORES=8 jobs/jobs_smoothgen08_pores.sh`). Checkpoints are saved
   every 10 epochs to `THERMAL_GAN_RUNS/training_outputs_strongD_smoothgen08<PORES>/`.
3. **Select a checkpoint.** `PORES=8 python src/screen_smoothgen08.py` ranks checkpoints by
   one-shot pore-count accuracy and ellipticity.
4. **Validate.** `python src/certify_all_pores.py` evaluates the released generators in
   `weights/`: FEM re-simulation error by anisotropy regime, pore count, ellipticity and
   orientation consistency. `python src/eval_diversity.py` computes within-label diversity.

Run scripts from the repository root.
