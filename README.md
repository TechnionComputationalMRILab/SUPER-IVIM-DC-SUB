# SUPER-IVIM-DC-SUB

Official code for the paper **"SUPER-IVIM-DC-SUB: A Physics-Informed Subset Ensembling Framework for Robust Placental IVIM Analysis in Uncontrolled Maternal Diabetes."**

This repository extends [SUPER-IVIM-DC](https://github.com/TechnionComputationalMRILab/SUPER-IVIM-DC) with a subset-ensembling framework for more robust IVIM parameter estimation in placental imaging.

## Table of Contents

- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Model Variants](#model-variants)
- [References](#references)

## Installation

```bash
conda create -n ivim_env python=3.10
conda activate ivim_env
pip install -r requirements.txt
```

## Training

Train a model and generate the corresponding `.pt` weights with:

```bash
python run_train.py
```

Select the model variant by setting the `arg` parameter (see [Model Variants](#model-variants) below):

| Variant | `arg` value |
|---|---|
| SUPER-IVIM-DC-SUB | `hp('sim_boot')` |
| SUPER-IVIM-DC-BVAL | `hp('sim_bvalue')` |
| SUPER-IVIM-DC | `hp('sim')` |

> **Note:** If training the `sim_bvalue` variant, update the b-values array from:
> ```python
> bvalues = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000])
> ```
> to:
> ```python
> bvalues = np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000])
> ```

Running `run_train.py` produces two main output files for each trained model:

- `<super_ivim_dc_filename>_init.json` - the initial values used during training
- `<super_ivim_dc_filename>.pt` - the trained PyTorch model weights

`<super_ivim_dc_filename>` is a name you choose to identify the model being trained.

## Inference

To generate a noisy synthetic signal and extract IVIM parameters from it, run:

```bash
python run_infer.py
```

Set `<super_ivim_dc_filename>` and the `arg` value (e.g. `hp('sim_boot')`) to match the model variant you want to run inference with.

## Model Variants

This repository supports three related models, selected via the `arg` parameter at training/inference time:

- **SUPER-IVIM-DC-SUB** (`sim_boot`) - the subset-ensembling model introduced in this paper
- **SUPER-IVIM-DC-BVAL** (`sim_bvalue`) - The variant using b-value concatenation without subset ensembling.
- **SUPER-IVIM-DC** (`sim`) - the base model from the original paper

## References

This code builds on SUPER-IVIM-DC:

- Code: https://github.com/TechnionComputationalMRILab/SUPER-IVIM-DC
- Paper: https://arxiv.org/abs/2206.03820

```bibtex
@inproceedings{korngut2022super,
  title={SUPER-IVIM-DC: Intra-voxel incoherent motion based Fetal lung maturity assessment from limited DWI data using supervised learning coupled with data-consistency},
  author={Korngut, Noam and Rotman, Elad and Afacan, Onur and Kurugol, Sila and Zaffrani-Reznikov, Yael and Nemirovsky-Rotman, Shira and Warfield, Simon and Freiman, Moti},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={743--752},
  year={2022},
  organization={Springer}
}
```
