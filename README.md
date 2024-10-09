<!-- markdownlint-disable line-length no-inline-html -->
# ETHOS.FINE - Framework for Integrated Energy System Assessment

[![Build Status](https://travis-ci.com/FZJ-IEK3-VSA/FINE.svg?branch=master)](https://travis-ci.com/FZJ-IEK3-VSA/FINE)
[![Version](https://img.shields.io/pypi/v/FINE.svg)](https://pypi.python.org/pypi/FINE)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/fine.svg)](https://anaconda.org/conda-forge/fine)
[![Documentation Status](https://readthedocs.org/projects/vsa-fine/badge/?version=latest)](https://vsa-fine.readthedocs.io/en/latest/)
[![PyPI - License](https://img.shields.io/pypi/l/FINE)](https://github.com/FZJ-IEK3-VSA/FINE/blob/master/LICENSE.txt)
[![codecov](https://codecov.io/gh/FZJ-IEK3-VSA/FINE/branch/master/graph/badge.svg)](https://codecov.io/gh/FZJ-IEK3-VSA/FINE)

<a href="https://www.fz-juelich.de/en/ice/ice-2"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/JSA-Header.svg?raw=True" alt="Forschungszentrum Juelich Logo" width="300px"></a>

The ETHOS.FINE python package provides a framework for modeling, optimizing and assessing energy systems. With the provided framework, systems with multiple regions, commodities, time steps and investment periods can be modeled. Target of the optimization is the minimization of the systems net present value (NPV) while considering technical and environmental constraints. If only one investment period is considered, the net present value is equal to the total annual costs (TAC). Besides using the full temporal resolution, an interconnected typical period storage formulation can be applied, that reduces the complexity and computational time of the model.

This Readme provides information on the installation of the package. For further information have a look at the [documentation](https://vsa-fine.readthedocs.io/en/latest/).

ETHOS.FINE is used for the modelling of a diverse group of optimization problems within the [Energy Transformation PatHway Optimization Suite (ETHOS) at ICE-2](https://www.fz-juelich.de/de/ice/ice-2/leistungen/model-services).  

If you want to use ETHOS.FINE in a published work, please [**kindly cite following publication**](https://www.sciencedirect.com/science/article/pii/S036054421830879X) which gives a description of the first stages of the framework. The python package which provides the time series aggregation module and its corresponding literature can be found [here](https://github.com/FZJ-IEK3-VSA/tsam).

## Installation 
There are several options for the installation of ETHOS.FINE. You can install it via PyPI or from conda-forge.
For detailed information, have a look at the [installation documentation](https://vsa-fine.readthedocs.io/en/latest/installationDoc.html).

NOTE: If you want to work on the source code of FINE, see [Editable install from conda-forge](#editable-install-from-conda-forge).

If you would like to run ETHOS.FINE for your analysis we recommend to install it directly from conda-forge into a new Python environment with

```bash
mamba create --name fine --channel conda-forge fine
```

**Note on Mamba vs.Conda:** `mamba` commands can be substitued with `conda`. We highly recommend using [(Micro-)Mamba](https://mamba.readthedocs.io/en/latest/) instead of Conda. The recommended way to use Mamba on your system is to install the [Miniforge distribution](https://github.com/conda-forge/miniforge#miniforge3). They offer installers for Windows, Linux and OS X. In principle, Conda and Mamba are interchangeable. The commands and concepts are the same. The distributions differ in the methodology for determining dependencies when installing Python packages. Mamba relies on a more modern methodology, which (with the same result) leads to very significant time savings during the installation of ETHOS.FINE. Switching to Mamba usually does not lead to any problems, as it is virtually identical to Conda in terms of operation.

**Note on the solver:** 
The functionality of ETHOS.FINE depends on the following C libraries that need to be installed on your system. If you do not know how to install those, consider installing from conda-forge. The mamba/conda installation comes with [GLPK](https://www.gnu.org/software/glpk/) [(installation for Windows)](https://sourceforge.net/projects/winglpk/files/latest/download) as Mixed Integer Linear Programming (MILP) solver. If you want to solve large problems it is highly recommended to install [GUROBI](http://www.gurobi.com/). See ["Installation of an optimization solver"](https://vsa-fine.readthedocs.io/en/latest/installationDoc.html#installation-of-an-optimization-solver) in the documentation for more information.


## Examples

A number of [examples](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples) shows the capabilities of ETHOS.FINE.

- [00_Tutorial](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/00_Tutorial)
  - In this application, an energy supply system, consisting of two regions, is modeled and optimized. Recommended as starting point to get to know to ETHOS.FINE.
- [01_1node_Energy_System_Workflow](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/01_1node_Energy_System_Workflow)
  - In this application, a single region energy system is modeled and optimized. The system includes only a few technologies.
- [02_EnergyLand](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/02_EnergyLand)
  - In this application, a single region energy system is modeled and optimized. Compared to the previous examples, this example includes a lot more technologies considered in the system.
- [03_Multi-regional_Energy_System_Workflow](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/03_Multi-regional_Energy_System_Workflow)
  - In this application, an energy supply system, consisting of eight regions, is modeled and optimized. The example shows how to model multi-regional energy systems. The example also includes a notebook to get to know the optional performance summary. The summary shows how the optimization performed.
- [04_Model_Run_from_Excel](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/04_Model_Run_from_Excel)
  - ETHOS.FINE can also be run by excel. This example shows how to read and run a model using excel files.
- [05_District_Optimization](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/05_District_Optimization)
  - In this application, a small district is modeled and optimized. This example also includes binary decision variables.
- [06_Water_Supply_System](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/06_Water_Supply_System)
  - The application cases of ETHOS.FINE are not limited. This application shows how to model the water supply system.
- [07_NetCDF_to_save_and_set_up_model_instance](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/07_NetCDF_to_save_and_set_up_model_instance)
  - This example shows how to save the input and optimized results of an energy system Model instance to netCDF files to allow reproducibility.
- [08_Spatial_and_technology_aggregation](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/08_Spatial_and_technology_aggregation)
  - These two examples show how to reduce the model complexity. Model regions can be aggregated to reduce the number of regions (spatial aggregation). Input parameters are automatically adapted. Furthermore, technologies can be aggregated to reduce complexity, e.g. reducing the number of different PV components (technology aggregation). Input parameters are automatically adapted.
- [09_Stochastic_Optimization](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/9_Stochastic_Optimizatio)
  - In this application, a stochastic optimization is performed. It is possible to perform the optimization of an energy system model with different input parameter sets to receive a more robust solution.
- [10_PerfectForesight](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/10_PerfectForesight)
  - In this application, a transformation pathway of an energy system is modeled and optimized showing how to handle several investment periods with time-dependent assumptions for costs and operation.
- [11_Partload](https://github.com/FZJ-IEK3-VSA/FINE/tree/develop/examples/11_Partload)
  - In this application, a hydrogen system is modeled and optimized considering partload behavior of the electrolyzer.

## Notes for developers

### Editable install from conda-forge

It is recommended to create a clean environment with conda to use ETHOS.FINE because it requires many dependencies.

```bash
mamba env create --name fine --file requirements_dev.yml
mamba activate fine
```

Install ETHOS.FINE as editable install and without checking the dependencies from pypi with

```bash
python -m pip install --no-deps --editable .
```

### Editable install from pypi

If you do not want to use conda-forge consider the steps in section [Installation from pipy](#Installation-from-pipy) and install ETHOS.FINE as editable install and with developer dependencies with

```bash
python -m pip install --editable .[develop]
```

### Good coding style

We use [ruff](https://docs.astral.sh/ruff) to ensure good coding style. Make
sure to use it before contributing to the code base with

```bash
ruff check fine
```

## License

MIT License

Copyright (C) 2016-2024 FZJ-ICE-2

Active Developers: Theresa Groß, Kevin Knosala, Noah Pflugradt, Johannes Behrens, Julian Belina, Arne Burdack, Toni Busch, Philipp Dunkel, David Franzmann, Patrick Freitag, Thomas Grube, Heidi Heinrichs, Maximilian Hoffmann, Jason Hu, Shitab Ishmam, Sebastian Kebrich, Felix Kullmann, Jochen Linßen, Rachel Maier, Shruthi Patil, Jan Priesmann, Julian Schönau, Maximilian Stargardt, Lovindu Wijesinghe, Christoph Winkler, Detlef Stolten

Alumni: Robin Beer, Henrik Büsing, Dilara Caglayan, Timo Kannengießer, Leander Kotzur, Stefan Kraus, Peter Markewitz, Lars Nolting,Stanley Risch, Martin Robinius, Bismark Singh, Andreas Smolenko, Peter Stenzel, Chloi Syranidou, Johannes Thürauf, Lara Welder, Michael Zier

You should have received a copy of the MIT License along with this program.
If not, see https://opensource.org/licenses/MIT


## About Us 

<a href="https://www.fz-juelich.de/en/ice/ice-2"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/iek3-square.png?raw=True" alt="Institute image ICE-2" width="280" align="right" style="margin:0px 10px"/></a>

We are the <a href="https://www.fz-juelich.de/en/ice/ice-2">Institute of Climate and Energy Systems (ICE) - Jülich Systems Analysis</a> belonging to the <a href="https://www.fz-juelich.de/en">Forschungszentrum Jülich</a>. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

## Contributions and Support
Every contributions are welcome:
- If you have a question, you can start a [Discussion](https://github.com/FZJ-IEK3-VSA/FINE/discussions). You will get a response as soon as possible.
- If you want to report a bug, please open an [Issue](https://github.com/FZJ-IEK3-VSA/FINE/issues/new). We will then take care of the issue as soon as possible.
- If you want to contribute with additional features or code improvements, open a [Pull request](https://github.com/FZJ-IEK3-VSA/FINE/pulls).

## Code of Conduct
Please respect our [code of conduct](CODE_OF_CONDUCT.md).

## Acknowledgement
This work was initially supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/). 

The authors also gratefully acknowledge financial support by the Federal Ministry for Economic Affairs and Energy of Germany as part of the project [METIS](http://www.metis-platform.net/) (project number 03ET4064, 2018-2022).

This work was supported by the Helmholtz Association under the program ["Energy System Design"](https://www.helmholtz.de/en/research/research-fields/energy/energy-system-design/).

<p float="left">
<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px"></a>
</p>
