Installation
############

There are several options for the installation of ETHOS.FINE. You can install it via PyPI or from conda-forge.
The provided framework enables you to create an optimization program based on your model constraints. 
The optmizitation program is built by using `PYOMO <https://pyomo.readthedocs.io/en/stable/index.html>`_.
To solve the program, ETHOS.FINE requires an MILP solver which can be accessed using `PYOMO <https://pyomo.readthedocs.io/en/stable/index.html>`_.

In the following, you find information on 

* how to install the package from conda-forge (:ref:`Installation from conda-forge`)
* how to install the package from PyPI (:ref:`Installation from PyPI`)
* how to install a solver (:ref:`Installation of an optimization solver`)

Installation from conda-forge
*****************************

If you would like to run ETHOS.FINE for your analysis we recommend to install it directly from conda-forge into a new Python environment with

.. code-block:: bash

    mamba create --name fine --channel conda-forge fine


**Note on Mamba vs.Conda:** `mamba` commands can be substitued with `conda`. We highly recommend using `(Micro-)Mamba <https://mamba.readthedocs.io/en/latest/>`_ instead of Conda. The recommended way to use Mamba on your system is to install the `Miniforge distribution <https://github.com/conda-forge/miniforge#miniforge3>`_ . They offer installers for Windows, Linux and OS X. In principle, Conda and Mamba are interchangeable. The commands and concepts are the same. The distributions differ in the methodology for determining dependencies when installing Python packages. Mamba relies on a more modern methodology, which (with the same result) leads to very significant time savings during the installation of ETHOS.FINE. Switching to Mamba usually does not lead to any problems, as it is virtually identical to Conda in terms of operation.

**Note on the solver:** The mamba/conda installation comes with `GLPK <https://www.gnu.org/software/glpk/>`_  as Mixed Integer Linear Programming (MILP) solver. If you want to solve large problems it is highly recommended to install `GUROBI <http://www.gurobi.com/>`_ . See :ref:`Installation of an optimization solver<Installation of an optimization solver>` for more information.

To install an editable version of the code, it is recommended to create a clean environment, e.g., with conda to use ETHOS.FINE because it requires many dependencies.

.. code-block:: bash

    mamba env create --name fine --file requirements_dev.yml
    mamba activate fine


Install ETHOS.FINE as editable install and without checking the dependencies from PyPI with

.. code-block:: bash

    python -m pip install --no-deps --editable .


Installation from PyPI
**********************

The functionality of ETHOS.FINE depends on the following C libraries that need to be installed on your system. If you do not know how to install those, consider installing from conda-forge.

- `GLPK <https://www.gnu.org/software/glpk/>`_
- `GDAL <https://gdal.org/index.html>`_

It is recommended to create a virtual environment. To do so, you can create a virtual environment with venv in the ETHOS.FINE folder

.. code-block:: bash

    python -m venv .venv

Find more information on creating virtual environments with venv `here <https://docs.python.org/3/library/venv.html#how-venvs-work>`_ .  

Install ETHOS.FINE with

.. code-block:: bash

    python -m pip install fine

To install an editable version of the code, install ETHOS.FINE with

.. code-block:: bash

    python -m pip install --editable .[develop]

Installation of an optimization solver
**************************************

ETHOS.FINE requires an MILP solver which can be accessed using `PYOMO <https://pyomo.readthedocs.io/en/stable/index.html>`_. It searches for the following solvers in this order:

GUROBI
======

The solver `GUROBI <http://www.gurobi.com/>`_ is recommended due to better performance but requires license (free academic version available). It is set as the default solver.

The installation requires the following three components:

- Gurobi Optimizer
  - In order to `download <https://www.gurobi.com/downloads/gurobi-optimizer-eula/>`_ the software you need to create an account and obtain a license.
- Gurobi license
  - The license needs to be installed according to the instructions in the registration process.
- Gurobi python api
  - The python api can be installed according to `this instruction <https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python->`_ .

GLPK
====

The solver `GLPK <https://sourceforge.net/projects/winglpk/files/latest/download>`_ is installed with the ETHOS.FINE environment. A complete installation instruction for Windows can be found `here <http://winglpk.sourceforge.net/>`_ .

CBC
===

Installation procedure for the solver `CBC <https://projects.coin-or.org/Cbc>`_ can be found `here <https://projects.coin-or.org/Cbc>`_ .


