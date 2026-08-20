<!-- markdownlint-disable line-length no-inline-html -->

<!-- logo:header:start -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/FZJ-IEK3-VSA/README_assets/blob/v.1.0.0/software_logos/fine/fine_logo_v19_dark.svg">
    <img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/v.1.0.0/software_logos/fine/fine_logo_v19_no_overlap.svg" alt="ETHOS.FINE logo" height="80">
  </picture>
  &nbsp;&nbsp;
  <a href="https://www.fz-juelich.de/en/ice/ice-2">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header-dark.svg">
      <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header.svg" alt="Jülich Systems Analysis" height="80">
    </picture>
  </a>
</p>
<!-- logo:header:end -->

# ETHOS.FINE - Framework for Integrated Energy System Assessment

**Multi-region, multi-commodity energy system optimization — from a single node to a whole transformation pathway.**

[![PyPI version](https://img.shields.io/pypi/v/FINE.svg)](https://pypi.python.org/pypi/FINE)
[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/fine.svg)](https://anaconda.org/conda-forge/fine)
[![Tests](https://github.com/FZJ-IEK3-VSA/FINE/actions/workflows/test_push.yml/badge.svg)](https://github.com/FZJ-IEK3-VSA/FINE/actions/workflows/test_push.yml)
[![Coverage](https://codecov.io/gh/FZJ-IEK3-VSA/FINE/branch/develop/graph/badge.svg)](https://codecov.io/gh/FZJ-IEK3-VSA/FINE)
[![Documentation](https://readthedocs.org/projects/vsa-fine/badge/?version=develop)](https://vsa-fine.readthedocs.io/en/develop/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06274/status.svg)](https://doi.org/10.21105/joss.06274)
[![License](https://img.shields.io/pypi/l/FINE)](https://github.com/FZJ-IEK3-VSA/FINE/blob/develop/LICENSE.txt)

<!-- readme-only:start -->
📖 **Read the full documentation at [vsa-fine.readthedocs.io](https://vsa-fine.readthedocs.io/).**
<!-- readme-only:end -->

The ETHOS.FINE python package provides a framework for modeling, optimizing and assessing energy systems. With the provided framework, systems with multiple regions, commodities, time steps and investment periods can be modeled. Target of the optimization is the minimization of the systems net present value (NPV) while considering technical and environmental constraints. If only one investment period is considered, the net present value is equal to the total annual costs (TAC). Besides using the full temporal resolution, an interconnected typical period storage formulation can be applied, that reduces the complexity and computational time of the model.

ETHOS.FINE is part of the [Energy Transformation PatHway Optimization Suite (ETHOS) at ICE-2](https://www.fz-juelich.de/de/ice/ice-2/leistungen/model-services), where it is used for the modelling of a diverse group of optimization problems. It uses [ETHOS.TSAM](https://github.com/FZJ-IEK3-VSA/tsam) to reduce the temporal complexity of its models.

ETHOS.FINE was the [JuRSE Code of the Month for February 2026](https://www.fz-juelich.de/en/rse/community-initiatives/jurse-code-of-the-month/february-2026).

## Features

- Multiple regions, commodities, time steps and investment periods in one model
- Component library covering sources and sinks, conversion, storage and transmission
- Cost-optimal design and operation against technical and environmental constraints
- Temporal complexity reduction via typical periods, including an interconnected storage formulation
- Spatial and technology aggregation to cut model size
- Transformation pathways with perfect foresight, stochastic optimization and partload behaviour
- Model setup from Excel and reproducible storage of inputs and results as netCDF

## Installation

There are several options for the installation of ETHOS.FINE. You can install it via PyPI or from conda-forge. In all of the following variants it is recommended to install your dependencies from conda-forge as the ecosystem is better tested and maintained. For more information on installation specifics and comparison between the different options, have a look at the [installation documentation](https://vsa-fine.readthedocs.io/en/develop/installation.html). In the following first the [prerequisites](#prerequisites) for the installation are presented. Then the recommended [installation](#installation-from-conda-forge-recommended) is shown. If you want to work on the source code of ETHOS.FINE, see [Development installation](#development-installation).

### Prerequisites

You either need a mamba or conda installation (recommended), but any Python installation will do. However, if you are unfamiliar with using environment manager (like mamba), you should [consider using one](https://realpython.com/python-virtual-environments-a-primer/). Fine has many dependencies and will likely interfere with other software projects on your machine in case you don't isolate them with an environment

You need conda or mamba installer on your machine, which are mostly interchangeable. The code is tested for Linux, Windows and Macos. We recommend to use [mamba](https://mamba.readthedocs.io/en/latest/) and install it using the [miniforge installer](https://github.com/conda-forge/miniforge). Please be aware that having multiple conda and mamba installation (for example from the [miniforge installer](https://github.com/conda-forge/miniforge) and the [anaconda installer](https://www.anaconda.com/download)) can cause serious problems during the installation. Please remove the other installer and any old environment on your machine if you decide to switch. 

### Installation from conda-forge (Recommended)

```bash
mamba create --name fine_env --channel conda-forge fine
```

### Installation from PyPI

Create venv environment
```bash
python -m venv .venv
```

Activate venv environment on Linux

```bash
source .venv/bin/activate
```

Activate venv environment on Windows

```bash
.venv\Scripts\activate
```

Find more information on creating virtual environments with venv [here](https://vsa-fine.readthedocs.io/en/develop/installation.html#installation-from-pypi).  

```bash
python -m pip install fine
```

### Solver 

At its core, ETHOS.FINE creates an optimisation problem via the Pyomo interface. A Mixed Integer Linear Programming (MILP) solver is required, and theoretically any solver supported by Pyomo can be used with ETHOS.FINE. ETHOS.FINE is tested with [GUROBI](https://www.gurobi.com/), [HiGHS](https://highs.dev/) and [GLPK](https://www.gnu.org/software/glpk/) preinstalled. If you want to solve large problems, it is highly recommended that you use [GUROBI](https://www.gurobi.com/) due to its superior performance. However, a proprietary licence is required to use GUROBI for larger optimisation problems, but this is available free of charge to academics. See the [installation documentation](https://vsa-fine.readthedocs.io/en/develop/installation.html#optimization-solver) for full details. If you do not want or cannot use a GUROBI licence, you can use HiGHS (which is slower than GUROBI but faster than GLPK) or GLPK, which do not require paid licences.

#### Conda Solver Installation 
The Conda installation of ETHOS.FINE comes with [GUROBI](https://www.gurobi.com/), [HiGHS](https://highs.dev/) and [GLPK](https://www.gnu.org/software/glpk/) preinstalled. 

#### PyPi Solver Installation 
If you use the PyPi installation, it comes with a reduced version of Pyomo called 'GurobiPy'. However, if you require the full Gurobi software or another solver, please check the solver provider's homepage. Alternatively, consider using Conda/Mamba.

### Development installation

#### Editable install from conda-forge

It is recommended to create a clean environment with conda to use ETHOS.FINE because it requires many dependencies.

```bash
mamba env create --name fine_env --file requirements_dev.yml
mamba activate fine_env
```

Install ETHOS.FINE as editable install and without checking the dependencies from pypi with

```bash
python -m pip install --no-deps --editable .
```

#### Editable install from pypi

If you do not want to use conda-forge consider the steps in section [Installation from PyPI](#installation-from-pypi) and install ETHOS.FINE as editable install and with developer dependencies with

```bash
python -m venv .venv
```

Activate venv environment on Linux

```bash
source .venv/bin/activate
```

Activate venv environment on Windows

```bash
.venv\Scripts\activate
```

```bash
python -m pip install --editable .[develop]
```

## Getting Started

A number of [examples](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples) shows the capabilities of ETHOS.FINE.

- [00_Tutorial](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/00_Tutorial)
  - In this application, an energy supply system, consisting of two regions, is modeled and optimized. Recommended as starting point to get to know to ETHOS.FINE.
- [01_1node_Energy_System_Workflow](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/01_1node_Energy_System_Workflow)
  - In this application, a single region energy system is modeled and optimized. The system includes only a few technologies.
- [02_EnergyLand](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/02_EnergyLand)
  - In this application, a single region energy system is modeled and optimized. Compared to the previous examples, this example includes a lot more technologies considered in the system.
- [03_Multi-regional_Energy_System_Workflow](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/03_Multi-regional_Energy_System_Workflow)
  - In this application, an energy supply system, consisting of eight regions, is modeled and optimized. The example shows how to model multi-regional energy systems. The example also includes a notebook to get to know the optional performance summary. The summary shows how the optimization performed.
- [04_District_Optimization](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/04_District_Optimization)
  - In this application, a small district is modeled and optimized. This example also includes binary decision variables.
- [05_Water_Supply_System](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/05_Water_Supply_System)
  - The application cases of ETHOS.FINE are not limited. This application shows how to model the water supply system.
- [06_NetCDF_to_save_and_set_up_model_instance](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/06_NetCDF_to_save_and_set_up_model_instance)
  - This example shows how to save the input and optimized results of an energy system Model instance to netCDF files to allow reproducibility.
- [07_Spatial_and_technology_aggregation](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/07_Spatial_and_technology_aggregation)
  - These two examples show how to reduce the model complexity. Model regions can be aggregated to reduce the number of regions (spatial aggregation). Input parameters are automatically adapted. Furthermore, technologies can be aggregated to reduce complexity, e.g. reducing the number of different PV components (technology aggregation). Input parameters are automatically adapted.
- [08_Stochastic_Optimization](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/08_Stochastic_Optimization)
  - In this application, a stochastic optimization is performed. It is possible to perform the optimization of an energy system model with different input parameter sets to receive a more robust solution.
- [09_PerfectForesight](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/09_PerfectForesight)
  - In this application, a transformation pathway of an energy system is modeled and optimized showing how to handle several investment periods with time-dependent assumptions for costs and operation.
- [10_Partload](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/10_Partload)
  - In this application, a hydrogen system is modeled and optimized considering partload behavior of the electrolyzer.

## Citation

If you want to use ETHOS.FINE in a published work, **please kindly cite**:

> Klütz, T., Knosala, K., Behrens, J., Maier, R., Hoffmann, M., Pflugradt, N., & Stolten, D. (2025). ETHOS.FINE: A Framework for Integrated Energy System Assessment. *Journal of Open Source Software*, 10(105), 6274. https://doi.org/10.21105/joss.06274

```bibtex
@article{Kluetz2025,
  title   = {{ETHOS.FINE}: A Framework for Integrated Energy System Assessment},
  author  = {Kl{\"u}tz, Theresa and Knosala, Kevin and Behrens, Johannes and Maier, Rachel and Hoffmann, Maximilian and Pflugradt, Noah and Stolten, Detlef},
  journal = {Journal of Open Source Software},
  volume  = {10},
  number  = {105},
  pages   = {6274},
  year    = {2025},
  doi     = {10.21105/joss.06274}
}
```

## Contributions and Support
All contributions are welcome:
- If you have a question, you can start a [Discussion](https://github.com/FZJ-IEK3-VSA/FINE/discussions). You will get a response as soon as possible.
- If you want to report a bug, please open an [Issue](https://github.com/FZJ-IEK3-VSA/FINE/issues/new). We will then take care of the issue as soon as possible.
- If you want to contribute with additional features or code improvements, open a [Pull request](https://github.com/FZJ-IEK3-VSA/FINE/pulls).

### Good coding style

We use [ruff](https://docs.astral.sh/ruff) to ensure good coding style. Make
sure to use it before contributing to the code base with

```bash
ruff check --config=pyproject.toml
ruff format --diff --config=pyproject.toml
```

## License

MIT License

Copyright (C) 2016-2025 FZJ-ICE-2

Active Developers: Johannes Behrens, Theresa Klütz, Noah Pflugradt,  Julian Belina, Arne Burdack, Toni Busch, Philipp Dunkel, David Franzmann, Maike Gnirß, Thomas Grube, Lars Hadidi, Heidi Heinrichs, Shitab Ishmam, Sebastian Kebrich, Jochen Linßen, Nils Ludwig, Lilly Madeisky, Drin Marmullaku, Gian Müller, Kenneth Okosun, Olalekan Omoyele, Shruthi Patil, Kai Schulze, Julian Schönau, Maximilian Stargardt, Lana Söltzer, Henrik Wenzel, Bernhard Wortmann, Lovindu Wijesinghe, Christoph Winkler, Detlef Stolten

Alumni: Robin Beer, Henrik Büsing, Dilara Caglayan, Patrick Freitag, Maximilian Hoffmann, Jason Hu, Timo Kannengießer, Kevin Knosala, Leander Kotzur, Felix Kullmann, Stefan Kraus, Rachel Maier, Peter Markewitz, Lars Nolting, Jan Priesmann, Stanley Risch, Martin Robinius, Bismark Singh, Andreas Smolenko, Peter Stenzel, Chloi Syranidou, Johannes Thürauf, Lara Welder, Michael Zier

You should have received a copy of the MIT License along with this program.
If not, see https://opensource.org/licenses/MIT

## About Us

We are the <a href="https://www.fz-juelich.de/en/ice/ice-2">Institute of Climate and Energy Systems – Jülich Systems Analysis (ICE-2)</a> at the <a href="https://www.fz-juelich.de/en"> Forschungszentrum Jülich</a>.
Our work focuses on independent, interdisciplinary research in energy, bioeconomy, infrastructure, and sustainability. We support a just, greenhouse gas–neutral transformation through open models and policy-relevant science.

## Code of Conduct
Please respect our [code of conduct](https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/CODE_CONDUCT.md).

## Acknowledgments

This work received primary support from the Helmholtz Association through the Joint Initiative ["Energy System 2050: A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/) and the program ["Energy System Design"](https://www.helmholtz.de/en/research/research-fields/energy/energy-system-design/). 

<p align="left">
  <!-- logo:helmholtz:start -->
  <a href="https://www.helmholtz.de/en/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-White-RGB.svg">
      <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-Dark-Blue-RGB.svg" alt="Helmholtz Association" width="200">
    </picture>
  </a>
  <!-- logo:helmholtz:end -->
</p>

The authors also gratefully acknowledge financial support by the Federal Ministry for Economic Affairs and Energy of Germany as part of the [project METIS (project number 03ET4064, 2018-2022)](https://www.fz-juelich.de/de/ice/ice-2/projekte/metis).
