# Installation

There are several options for the installation of ETHOS.FINE. You can install it via PyPI or from conda-forge.
The provided framework enables you to create an optimization program based on your model constraints.
The optimization program is built by using [PYOMO](https://pyomo.readthedocs.io/en/stable/index.html).
To solve the program, ETHOS.FINE requires an MILP solver which is be accessed using [PYOMO](https://pyomo.readthedocs.io/en/stable/index.html). Depending on your solver choice you might need to obtain and activate a license. 

In the following, you find information on:

- how to install the package from conda-forge ([Installation from conda-forge](#installation-from-conda-forge))
- how to install the package from PyPI ([Installation from PyPI](#installation-from-pypi))
- how to ([select and activate a solver](#installation-of-an-optimization-solver))

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

The functionality of ETHOS.FINE depends on a MILP solver that cannot be easily installed using PyPi alone. Please refer to the homepage of the solver provider in the next section, or consider installing it from Conda Forge.

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

ETHOS FINE requires a MILP solver, which can be accessed via [PYOMO](https://pyomo.readthedocs.io/). In theory, any MILP solver supported by Pyomo can be used with ETHOS.FINE, but it has only been tested with Gurobi and GLPK. Both of these solvers are pre-installed with the Conda Forge installation. If you do not want to install from Conda Forge, please refer to the homepage of the solver of interest for installation instructions.

### GUROBI

The solver [GUROBI](http://www.gurobi.com/) is recommended due to better performance but requires a license (free academic version available). It is set as the default solver. In order to activate gurobi please follow these steps:
1. Create a [free Gurobi account](https://www.gurobi.com/downloads/end-user-license-agreement-academic/)
   and request a named-user academic license from the [Gurobi user portal](https://portal.gurobi.com/).
2. Copy the license key shown in the portal (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
3. activate your fine environment

  ```bash
  conda activate fine_env
  ```
3. Run the activation command once (internet required):

    ```bash
    grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    ```

    This downloads the license and saves it to `~/gurobi.lic`. 
4. You can optionally check whether the activation has succeeded by running a file containing the following from an activated environment:




```python
import fine as fn
fn.check_gurobi_license()
```

Example output for a valid named-user license:

```
Checking Gurobi license ...
  gurobipy version : 11.0.0
  License source   : Named-user license file (/home/user/gurobi.lic)
  License type     : named-user
  [OK]  License is valid.
```

Example output for a valid WLS license:

```
Checking Gurobi license ...
  gurobipy version : 11.0.0
  License source   : WLS credentials in environment variables
  License type     : wls-env
  [OK]  License is valid.
```

If the check fails, the function prints the detected license type and actionable hints to resolve
the issue.

### GLPK

The solver [GLPK](https://sourceforge.net/projects/winglpk/files/latest/download) is installed with the
ETHOS.FINE environment and can be used without any further steps.
