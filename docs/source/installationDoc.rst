############
Installation
############

In the following, instructions for installing and using the FINE framework on Windows are given. The installation
instructions for installing and using FINE on Linux/macOS systems are however quite similar and can be, hopefully
easily, derived from the instructions below.

**Python installation**

FINE runs on Python 3 platforms (i.e. Anaconda). Currently, it is advised not to use a Python version exceeding
Python 3.6. Note: When installing the Python platform Anaconda the options

    $ Add Anaconda to the system PATH environment variable

is available in the advanced installation option. When selecting this options, the environment variables for Python,
pip, jupyter etc. are remotely set and do not have to be manually set.

A development platform which can be used to work with/on the code and which comes with Anaconda is Spyder.
Another option for a development platform is PyCharm.

**FINE installation**

Install via pip by typing

    $ pip install FINE

into the command prompt. Alternatively, download or clone a local copy of the repository to your computer

    $ git clone https://github.com/FZJ-IEK3-VSA/FINE.git

and install FINE in the folder where the setup.py is located with

    $ pip install -e .

or install directly via python as

    $ python setup.py install

**Installation of additional packages**

The Python packages `tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_ and `PYOMO <http://www.pyomo.org/>`_ should be
installed by pip alongside FINE. Some plots in FINE require the GeoPandas package to be installed (nice-to-have).
Installation instructions are given `here <http://geopandas.org/install.html>`_. In some cases, the dependencies of
the GeoPandas package have to be installed manually before the package itself can be installed.

**Installation of an optimization solver**

In theory many solvers can be used (e.g. `GUROBI <http://www.gurobi.com/>`_  or
`GLPK <https://sourceforge.net/projects/winglpk/files/latest/download>`_). For the installation of GUROBI, follow
the instructions on the solver's website. GUROBI has, if applicable, an academic license option. For installation
of GLPK, move the downloaded folder to a desired location. Then, manually append the Environment Variable *Path*
with the absolute path leading to the folder in which the glpsol.exe is located (c.f. w32/w64 folder, depending on
operating system type).