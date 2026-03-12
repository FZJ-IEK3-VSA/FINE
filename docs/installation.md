# Installation

There are several options for the installation of ETHOS.FINE. You can install it via PyPI or from conda-forge.
The provided framework enables you to create an optimization program based on your model constraints.
The optimization program is built by using [PYOMO](https://pyomo.readthedocs.io/en/stable/index.html).
To solve the program, ETHOS.FINE requires an MILP solver which can be accessed using [PYOMO](https://pyomo.readthedocs.io/en/stable/index.html).

In the following, you find information on:

- how to install the package from conda-forge ([Installation from conda-forge](#installation-from-conda-forge))
- how to install the package from PyPI ([Installation from PyPI](#installation-from-pypi))
- how to install a solver ([Installation of an optimization solver](#installation-of-an-optimization-solver))

## Installation from conda-forge

If you would like to run ETHOS.FINE for your analysis we recommend to install it directly from conda-forge into a new Python environment with

```bash
mamba create --name fine_env --channel conda-forge fine
```

!!! note "Mamba vs. Conda"
    `mamba` commands can be substituted with `conda`. We highly recommend using
    [Mamba](https://mamba.readthedocs.io/en/latest/) instead of Conda. The recommended way to use Mamba on
    your system is to install the [Miniforge distribution](https://github.com/conda-forge/miniforge#miniforge3).
    They offer installers for Windows, Linux and OS X. In principle, Conda and Mamba are interchangeable.
    The commands and concepts are the same.

!!! note "On the solver"
    The mamba/conda installation comes with [GLPK](https://www.gnu.org/software/glpk/) as Mixed Integer Linear
    Programming (MILP) solver. If you want to solve large problems it is highly recommended to install
    [GUROBI](http://www.gurobi.com/). See [Installation of an optimization solver](#installation-of-an-optimization-solver) for more information.

To install an editable version of the code, it is recommended to create a clean environment, e.g., with conda to use ETHOS.FINE because it requires many dependencies.

```bash
mamba env create --name fine_env --file requirements_dev.yml
mamba activate fine_env
```

Install ETHOS.FINE as editable install and without checking the dependencies from PyPI with

```bash
python -m pip install --no-deps --editable .
```

Installation from conda-forge is also recommended because conda-forge provides
[Repodata patching](https://prefix.dev/blog/repodata_patching). This means that any known issues with
dependency constraints are fed back to the automatic installation procedure, providing the best possible
out-of-the-box installation experience.

## Installation from PyPI

The functionality of ETHOS.FINE depends on the following C libraries that need to be installed on your system.
If you do not know how to install those, consider installing from conda-forge.

- [GLPK](https://www.gnu.org/software/glpk/)
- [GDAL](https://gdal.org/index.html)

It is recommended to create a virtual environment. Create the venv environment:

```bash
python -m venv .venv
```

Activate venv environment on Linux:

```bash
source .venv/bin/activate
```

Activate venv environment on Windows:

```bash
.venv\Scripts\activate
```

Find more information on creating virtual environments with venv [here](https://docs.python.org/3/library/venv.html#how-venvs-work).

Install ETHOS.FINE with:

```bash
python -m pip install fine
```

To install an editable version of the code, install ETHOS.FINE with:

```bash
python -m pip install --editable .[develop]
```

## Installation of an optimization solver

ETHOS.FINE requires an MILP solver which can be accessed using [PYOMO](https://pyomo.readthedocs.io/).
It searches for the following solvers in this order:

### GUROBI

The solver [GUROBI](http://www.gurobi.com/) is recommended due to better performance but requires a license
(free academic version available). It is set as the default solver.

The installation requires the following three components:

- **Gurobi Optimizer** — In order to [download](https://www.gurobi.com/downloads/) the
  software you need to create an account and obtain a license.
- **Gurobi license** — The license needs to be installed according to the instructions in the registration process.
- **Gurobi python api** — The python api comes automatically with the fine installation.

### GLPK

The solver [GLPK](https://sourceforge.net/projects/winglpk/files/latest/download) is installed with the
ETHOS.FINE environment.

### CBC

Installation procedure for the solver [CBC](https://projects.coin-or.org/Cbc) can be found
[here](https://projects.coin-or.org/Cbc). Please note that the CBC solver is no longer actively tested. Results may differ from those of the GLPK or Gurobi solvers.