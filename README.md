[![Build Status](https://travis-ci.com/FZJ-IEK3-VSA/FINE.svg?branch=master)](https://travis-ci.com/FZJ-IEK3-VSA/FINE)
[![Version](https://img.shields.io/pypi/v/FINE.svg)](https://pypi.python.org/pypi/FINE)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/fine.svg)](https://anaconda.org/conda-forge/fine)
[![Documentation Status](https://readthedocs.org/projects/vsa-fine/badge/?version=latest)](https://vsa-fine.readthedocs.io/en/latest/)
[![PyPI - License](https://img.shields.io/pypi/l/FINE)]((https://github.com/FZJ-IEK3-VSA/FINE/blob/master/LICENSE.txt))
[![codecov](https://codecov.io/gh/FZJ-IEK3-VSA/FINE/branch/master/graph/badge.svg)](https://codecov.io/gh/FZJ-IEK3-VSA/FINE)


<a href="https://www.fz-juelich.de/en/iek/iek-3"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/FJZ_IEK-3_logo.svg" alt="Forschungszentrum Juelich Logo" width="300px"></a> 

# ETHOS.FINE - Framework for Integrated Energy System Assessment

The ETHOS.FINE python package provides a framework for modeling, optimizing and assessing energy systems. With the provided framework, systems with multiple regions, commodities and time steps can be modeled. Target of the optimization is the minimization of the total annual cost while considering technical and enviromental constraints. Besides using the full temporal resolution, an interconnected typical period storage formulation can be applied, that reduces the complexity and computational time of the model.

ETHOS.FINE is used for the modelling of a diverse group of optimization problems within the [Energy Transformation PatHway Optimization Suite (ETHOS) at IEK-3](https://www.fz-juelich.de/de/iek/iek-3/leistungen/model-services).  

If you want to use ETHOS.FINE in a published work, please [**kindly cite following publication**](https://www.sciencedirect.com/science/article/pii/S036054421830879X) which gives a description of the first stages of the framework. The python package which provides the time series aggregation module and its corresponding literatur can be found [here](https://github.com/FZJ-IEK3-VSA/tsam).

## Features
* representation of an energy system by multiple locations, commodities and time steps
* complexity reducing storage formulation based on typical periods

## Documentation
A "Read the Docs" documentation of ETHOS.FINE can be found [here](https://vsa-fine.readthedocs.io/en/latest/).

## Requirements
The installation process uses a Conda-based Python package manager. We highly recommend using [(Micro-)Mamba](https://mamba.readthedocs.io/en/latest/) instead of Anaconda. The recommended way to use Mamba on your system is to install the [Miniforge distribution](https://github.com/conda-forge/miniforge#miniforge3). They offer installers for Windows, Linux and OS X. Have a look at the Miniforge Readme for further details.

The project environment includes [GLPK](https://sourceforge.net/projects/winglpk/files/latest/download) as Mixed Integer Linear Programming (MILP) solver. If you want to solve large problems it is highly recommended to install [GUROBI](http://www.gurobi.com/). See ["Installation of an optimization solver"](#installation-of-an-optimization-solver) for more information.

## Installation

###  Installation via conda-forge
The simplest way ist to install FINE into a fresh environment from `conda-forge` with:
```bash
mamba create -n fine -c conda-forge fine
```

### Installation from local folder
Alternatively you can first clone the content of this repository and perform the installation from there: 

1. Clone the content of this repository 
```bash
git clone https://github.com/FZJ-IEK3-VSA/FINE.git 
```
2. Move into the FINE folder with
```bash
cd fine
```
3. It is recommended to create a clean environment with conda to use FINE because it requires many dependencies. 
```bash
mamba env create -f requirements.yml
```
5. Activate the new enviroment. You should see `(fine)` in front of your command prompt to indicate that you are now in the virtual environment.
```bash
mamba activate fine
```

### Installation for developers
I you want to work on the FINE codebase you need to run. 
```bash
mamba env create -f requirements_dev.yml
```
This installs additional dependencies such as `pytest` and installs FINE from the folder in editable mode with `pip -e`. Changes in the folder are then reflected in the package installation.

You can run the following command in the project root folder:
```
pytest
```

## Installation of an optimization solver

FINE requires an MILP solver which can be accessed using [PYOMO](https://pyomo.readthedocs.io/en/stable/index.html). It searches for the following solvers in this order:
- [GUROBI](http://www.gurobi.com/)
   - Recommended due to better performance but requires license (free academic version available)
   - Set as standard solver
- [GLPK](https://sourceforge.net/projects/winglpk/files/latest/download)
  - This solver is installed with the FINE environment.
  - Free version available 
- [CBC](https://projects.coin-or.org/Cbc)
  - Free version available

### Gurobi installation
The installation requires the following three components:
- Gurobi Optimizer
    - In order to [download](https://www.gurobi.com/downloads/gurobi-optimizer-eula/) the software you need to create an account and obtain a license.
- Gurobi license
    - The license needs to be installed according to the instructions in the registration process.
- Gurobi python api
    - The python api can be installed according to [this instruction](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-).

### GLPK installation
A complete installation instruction for Windows can be found [here](http://winglpk.sourceforge.net/).

### CBC
Installation procedure can be found [here](https://projects.coin-or.org/Cbc).

## Examples

A number of [examples](examples/) shows the capabilities of FINE.

## License

MIT License

Copyright (C) 2016-2022 FZJ-IEK-3

Active Developers: Theresa Groß, Kevin Knosala, Noah Pflugradt, Johannes Behrens, Julian Belina, Arne Burdack, Toni Busch, Philipp Dunkel, Patrick Freitag, Thomas Grube, Heidi Heinrichs, Maximilian Hoffmann, Shitab Ishmam, Stefan Kraus, Felix Kullmann, Jochen Linßen, Rachel Maier, Peter Markewitz, Lars Nolting, Shruthi Patil, Jan Priesmann, Stanley Risch, Julian Schönau, Bismark Singh, Maximilian Stargardt, Christoph Winkler, Michael Zier, Detlef Stolten

Alumni: Robin Beer, Henrik Büsing, Dilara Caglayan, Timo Kannengießer, Leander Kotzur, Martin Robinius, Andreas Smolenko, Peter Stenzel, Chloi Syranidou, Johannes Thürauf, Lara Welder

You should have received a copy of the MIT License along with this program.
If not, see https://opensource.org/licenses/MIT


## About Us 

<a href="https://www.fz-juelich.de/en/iek/iek-3"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/iek3-square.png?raw=True" alt="Institute image IEK-3" width="280" align="right" style="margin:0px 10px"/></a>

We are the <a href="https://www.fz-juelich.de/en/iek/iek-3">Institute of Energy and Climate Research - Techno-economic Systems Analysis (IEK-3)</a> belonging to the <a href="https://www.fz-juelich.de/en">Forschungszentrum Jülich</a>. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

## Contributions and Users

From 2018 to 2022  we developed  together with the RWTH-Aachen ([Prof. Aaron Praktiknjo](http://www.wiwi.rwth-aachen.de/cms/Wirtschaftswissenschaften/Die-Fakultaet/Institute-und-Lehrstuehle/Professoren/~jgfr/Praktiknjo-Aaron/?allou=1&lidx=1)), the [EDOM Team at FAU](https://www.math.fau.de/wirtschaftsmathematik/) and the [Jülich Supercomputing Centre](http://www.fz-juelich.de/ias/jsc/DE/Home/home_node.html) new methods and models for ETHOS.FINE within the BMWi funded project [METIS](http://www.metis-platform.net/).

<p float="left">
<a href="https://www.rwth-aachen.de/go/id/a/"> <img src="https://upload.wikimedia.org/wikipedia/commons/1/1e/RWTH_Logo_3.svg" width="230" /> </a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<a href="https://www.fau.de/"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Friedrich-Alexander-Universit%C3%A4t_Erlangen-N%C3%BCrnberg_logo.svg/2000px-Friedrich-Alexander-Universit%C3%A4t_Erlangen-N%C3%BCrnberg_logo.svg.png" width="230" /> </a>
</p>

## Acknowledgement

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>
