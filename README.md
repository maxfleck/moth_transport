# moth-transport

Repository of MotH - Model of the Henrys

### Notebooks

#### 00_MotH_base_demo

This notebook introduces MotH and showns how to fit viscosity and self-diffusion from scratch, fine-tuning or both properties together.

#### 01_MotH_thermal_cond_qualitative

Qualitative analysis of the relation between a free volume model and thermal conductivity.

#### 03_MotH_demo_train_parallel

Parallel training of viscosity, thermal conductivity and self-diffusion.

#### 04_free_volume_energy

Plots of free volume and activation energy for different molecules includin MD water results.


### Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then sync and launch:

```bash
uv sync        # install all dependencies into .venv
uv run jupyter-lab
```

Or open in VS Code (`code .`) and select the `.venv` interpreter.

---



## Cite



If you find MotH useful for your own scientific studies, consider citing our publication accompanying this library.

```
@article{fleck2026moth,
author = {Maximilian Fleck  and Marcelle B M Spera },
title = {Model of the Henrys: Eyring, Frank, and the Foundations for Predicting Transport Coefficients},
journal = {ChemRxiv},
volume = {2026},
number = {0419},
pages = {},
year = {2026},
doi = {10.26434/chemrxiv.15001840/v2},
URL = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15001840/v2},
}
```
If you use the experimental data provided here, consider citing it too. Sources are available in the SI of our publication.

If you use the code, please also cite FeOs as we are using the code as well as the PC-SAFT parameters from this package.

```
@article{rehner2023feos,
  author = {Rehner, Philipp and Bauer, Gernot and Gross, Joachim},
  title = {FeOs: An Open-Source Framework for Equations of State and Classical Density Functional Theory},
  journal = {Industrial \& Engineering Chemistry Research},
  volume = {62},
  number = {12},
  pages = {5347-5357},
  year = {2023},
}
```

