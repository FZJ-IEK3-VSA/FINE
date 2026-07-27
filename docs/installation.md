# Installation

There are several options for the installation of ETHOS.FINE. You can install it via PyPI or from conda-forge.
The provided framework enables you to create an optimization program based on your model constraints.
The optimization program is built by using [PYOMO](https://pyomo.readthedocs.io/en/stable/index.html).
To solve the program, ETHOS.FINE requires an MILP solver which can be accessed using [PYOMO](https://pyomo.readthedocs.io/en/stable/index.html). Depending on your solver choice you might need to obtain and activate a license. 

In the following, you find information on:

- how to install the package from conda-forge ([Installation from conda-forge](#installation-from-conda-forge))
- how to install the package from PyPI ([Installation from PyPI](#installation-from-pypi))
- how to ([select and activate a solver](#optimization-solver))

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

To install ETHOS.FINE it is recommended to create a virtual environment. Create the venv environment:

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

## Optimization solver

### Selection of an Optimization Solver
At its core, ETHOS.FINE creates an optimisation problem via the [PYOMO](https://pyomo.readthedocs.io/) interface. In theory, any MILP solver supported by Pyomo can be used with ETHOS.FINE, but it has only been tested with [GUROBI](http://www.gurobi.com/), [HiGHS](https://highs.dev/) and [GLPK](https://www.gnu.org/software/glpk/). If you want to solve large problems, it is highly recommended that you use [GUROBI](http://www.gurobi.com/) due to its superior performance. However, a proprietary licence is required to use GUROBI for larger optimisation problems, but this is available free of charge to academics. If you do not want or cannot use a GUROBI licence, you can use HiGHS (which is slower than GUROBI but faster than GLPK) or GLPK, which do not require paid licences.

### GUROBI

A full Gurobi installation comes with the Conda Forge installation, while a reduced installation (called gurobipy) comes with the PyPI installation. Small problems can be solved using the preinstalled test licence. However, for larger problems, you need to activate a Gurobi licence by following these steps:

1. Create a [free Gurobi account](https://www.gurobi.com/downloads/end-user-license-agreement-academic/)
   and request a named-user academic license from the [Gurobi user portal](https://portal.gurobi.com/).
2. Copy the license key shown in the portal (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
3. activate your fine environment

    ```bash
    conda activate fine_env
    ```

4. Run the activation command once (internet required):

    ```bash
    grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    ```

    This downloads the license and saves it to `~/gurobi.lic`. 

5. (Optional and only with conda forge) Check whether the activation has succeeded by the following command from an activated environment:

    ```bash
    gurobi_cl --license
    ```

    Example output for a valid named-user license:

    ```
    Set parameter Username
    Set parameter LicenseID to value 2793634
    Set parameter LogFile to value "gurobi.log"
    Using license file C:\Users\j.belina\gurobi.lic
    Academic license - for non-commercial use only - expires 2027-03-17
    ```

    If the check fails, the function prints the detected license type and actionable hints to resolve
    the issue.

### HiGHS and GLPK

The conda-forge installation of Fine comes with the [HiGHS](https://highs.dev/) and [GLPK](https://www.gnu.org/software/glpk/) solvers preinstalled, which can be used without any further steps. If you installed it successfully using PyPI, you will need to search for binaries compatible with your operating system, or compile the solver yourself. This can be a complex task. If you don't feel ready for that, please consider switching to the Conda installation.

