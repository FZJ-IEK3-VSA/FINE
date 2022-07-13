###############
Getting started
###############

**************************
Purpose and Vision of FINE
**************************

FINE is a framework for generating energy system optimization models. This might sound difficult, but the puporse is easy to understand:
FINE is designed to answer pressing questions on future energy systems which include affordability, a high share of renewable energy
sources and - most importantly - low CO\ :sub:`2` emissions.

The concept of FINE is that scientists, programmers and anyone who is interested all around the world can use FINE to answer their
individual questions. Therefore, FINE is open source available and completely for free. Once FINE is installed, you can start
implementing an energy system model that you want to investigate.

************
Installation
************

In the following, instructions for installing and using the FINE framework on Windows are given. The installation
instructions for installing and using FINE on Linux/macOS systems are however quite similar and can be, hopefully
easily, derived from the instructions below.

Prepare and install required software
=====================================

1. Install anaconda [by choosing your operating system here] (https://docs.anaconda.com/anaconda/install/). If you are a Windows 10 user, remember to tick "Add Anaconda to my PATH environment variable" during installation under "Advanced installations options".
2. Install git from https://git-scm.com/downloads

Prepare folder
==============

1. Open a prompt e.g. "anaconda prompt" or "cmd" from the windows start menu
2. Make a folder where you want to work, for example C:\Users\<your username>\work with "mkdir C:\Users\<your username>\work"
3. Go to that directory with "cd C:\Users\<your username>\work" at the command line

Get source code via GIT
=========================

Clone public repository or repository of your choice first

.. code-block:: console

    git clone https://github.com/FZJ-IEK3-VSA/FINE.git 

Move into the FINE folder with

.. code-block:: console

    cd fine

Installation for users
======================

It is recommended to create a clean environment with conda to use FINE because it requires many dependencies. 

.. code-block:: console

    conda env create -f requirements.yml

This directly installs FINE and its dependencies in the `FINE` conda environment. Activate the created environment with:

.. code-block:: console

    activate FINE

Installation for developers
===========================

Create a development environment if you want to modify it.
Install the requirements in a clean conda environment:

.. code-block:: console

     conda env create -f requirements_dev.yml
     activate FINE_dev

This installs FINE and its requirements for development (testing, formatting). Further changes in the current folder are reflected in package installation through the installation with `pip -e`.

Run the test suite with:

.. code-block:: console 

    pytest --cov=FINE test/

A development platform which can be used to work with/on the code and which comes with Anaconda is Spyder.
Other development platforms are PyCharm or Visua Studio Code.

The Python packages `tsam <https://github.com/FZJ-IEK3-VSA/tsam>`_ and `PYOMO <http://www.pyomo.org/>`_ are
installed by pip alongside FINE. Some plots in FINE require the GeoPandas package to be installed (nice-to-have).
Installation instructions are given `here <http://geopandas.org/install.html>`_. In some cases, the dependencies of
the GeoPandas package have to be installed manually before the package itself can be installed.

Installation of an optimization solver
======================================

FINE requires an MILP solver which can be accessed using `PYOMO <https://pyomo.readthedocs.io/en/stable/index.html>`_. There are three standard solvers defined:

* `GUROBI <http://www.gurobi.com/>`_

   * Recommended due to better performance but requires license (free academic version available)
   * Set as standard solver

* `GLPK <https://sourceforge.net/projects/winglpk/files/latest/download>`_

  * Free version available 

* `CBC <https://projects.coin-or.org/Cbc>`_

  * Free version available

Gurobi installation
-------------------

The installation requires the following three components:

* Gurobi Optimizer
    * In order to `download <https://www.gurobi.com/downloads/gurobi-optimizer-eula/>`_ the software you need to create an Account and obtain a license.
* Gurobi license
    * The license needs to be installed according to the instructions in the registration process.
* Gurobi python api
    * The python api can be installed according to `this instruction <https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python->`_.

GLPK installation
-----------------

A complete installation instruction for Windows can be found `here <http://winglpk.sourceforge.net/>`_.

CBC
---

Installation procedure can be found `here <https://projects.coin-or.org/Cbc>`_.

********
About Us
********

.. image:: https://www.fz-juelich.de/iek/iek-3/DE/_Documents/Pictures/IEK-3Team_2019-02-04.jpg?__blob=poster
    :target: https://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html
    :alt: Abteilung TSA
    :align: center

We are the `Institute of Energy and Climate Research - Techno-economic Systems Analysis (IEK-3) <https://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html>`_ 
belonging to the `Forschungszentrum Jülich <www.fz-juelich.de/>`_. Our interdisciplinary institute's research is 
focusing on energy-related process and systems analyses. Data searches and system simulations are used to 
determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. 
The results are used for performing comparative assessment studies between the various systems. Our current priorities 
include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction 
targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis 
studies for integrating new technologies into future energy market frameworks.


**Contributions and Users**

Within the BMWi funded project `METIS <http://www.metis-platform.net/>`_ we develop together with the RWTH-Aachen 
`(Prof. Aaron Praktiknjo) <http://www.wiwi.rwth-aachen.de/cms/Wirtschaftswissenschaften/Die-Fakultaet/Institute-und-Lehrstuehle/Professoren/~jgfr/Praktiknjo-Aaron/?allou=1&lidx=1>`_,
the EDOM Team at FAU `(PD Bismark Singh) <https://www.math.fau.de/wirtschaftsmathematik/team/bismark-singh/>`_ and the 
`Jülich Supercomputing Centre (JSC) <http://www.fz-juelich.de/ias/jsc/DE/Home/home_node.html>`_ new methods and models within FINE.

.. image:: http://www.metis-platform.net/metis-platform/DE/_Documents/Pictures/projectTeamAtKickOffMeeting_640x338.jpg?__blob=normal
    :target: http://www.metis-platform.net
    :alt: METIS Team
    :align: center
