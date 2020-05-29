# Getting started

## Purpose and Vision of SPAGAT

As energy system analyses and required energy system optimizations become more and more complex, computational tractability becomes an issue. 

Therefore, **SPAGAT** was created to find an optimal compromise between accurately representing the (energy system) data for the model regions and reducing the resulting computational complexity to ensure computational tractability.

In German we might refer to this "balancing act" as ["Spagat"](https://www.google.com/search?client=firefox-b-d&q=google+translate+spagat), which might have inspired the developers naming decision.

## About Us

![IEK3](https://www.fz-juelich.de/iek/iek-3/DE/_Documents/Pictures/IEK-3Team_2019-02-04.jpg?__blob=poster)

We are the [Techno-Economic Energy Systems Analysis](http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html)
department at the [Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)](http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the Forschungszentrum Jülich GmbH. Our
interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and
system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and
costs of energy systems. The results are used for performing comparative assessment studies between the various systems.
Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s
greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and
by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

### Contributions and Users

Within the BMWi funded project [METIS](http://www.metis-platform.net/) we develop together with the [RWTH-Aachen
(Prof. Aaron Praktiknjo)](http://www.wiwi.rwth-aachen.de/cms/Wirtschaftswissenschaften/Die-Fakultaet/Institute-und-Lehrstuehle/Professoren/~jgfr/Praktiknjo-Aaron/?allou=1&lidx=1>),
the [EDOM Team at FAU (PD Lars Schewe)](http://www.mso.math.fau.de/de/edom/team/schewe-lars/dr-lars-schewe) and the
[Jülich Supercomputing Centre (JSC)](http://www.fz-juelich.de/ias/jsc/DE/Home/home_node.html) new methods and models
within FINE, TSAM and SPAGAT.

![METIS Team](http://www.metis-platform.net/metis-platform/DE/_Documents/Pictures/projectTeamAtKickOffMeeting_640x338.jpg?__blob=normal)

## Installation

In the following, instructions for installing and using the **SPAGAT** framework on Windows are given. The installation
instructions for installing and using **SPAGAT** on Linux/macOS systems are however quite similar and can be, hopefully
easily, derived from the instructions below.

### Python installation

SPAGAT runs on Python 3 platforms (i.e. Anaconda). Currently, it is advised not to use a Python version exceeding
Python 3.6. Note: When installing the Python platform Anaconda the options

```
Add Anaconda to the system PATH environment variable
```

is available in the advanced installation option. When selecting this options, the environment variables for Python,
`pip`, `jupyter` etc. are remotely set and do not have to be manually set.

A development platform which can be used to work with/on the code and which comes with Anaconda is Spyder.
Other development platforms are PyCharm or [Visual Studio Code](https://code.visualstudio.com/).

### **SPAGAT** installation

Download or clone a local copy of the repository to your computer

```bash
git clone https://gitlab.version.fz-juelich.de/metis/spagat
```

and install the required dependencies via `conda`:

```bash
conda env create -f environment.yml
```

Finally, install **SPAGAT** in the folder where the `setup.py` is located using

```bash
pip install -e .
```

### Installation of additional packages

To conduct energy system optimizations, you can [install FINE](https://github.com/FZJ-IEK3-VSA/FINE).

## Examples

### High-level introduction

Define where the input dataset can be found and where to store the aggregated output dataset:

```Python
sds_folder_path_in = pathlib.Path("tests/data/input")
sds_folder_path_out = pathlib.Path("tests/data/output/aggregated/33")
spu.create_dir(sds_folder_path_out)
```

Initialize the spagat manager that handles the spatial aggregation procedure and read the input dataset:

```Python
spagat_manager = spm.SpagatManager()
spagat_manager.analysis_path = sds_folder_path_out
spagat_manager.read_data(sds_folder_path=sds_folder_path_in)
```

Group the model regions from n initial regions into a reduced number of m regions (m < n).
In this example, we choose hierarchical clustering such that the clustering is performed for all numbers of regions (1 < m < n):

```Python
spagat_manager.grouping()
```

Represent the initial dataset for a specified number of regions:
```Python
n_regions = 42
spagat_manager.representation(number_of_regions=n_regions)
```

Save the aggregated dataset to the aforementioned path:

```Python
spagat_manager.save_data(
    sds_folder_path_out,
    eligibility_variable="AC_cable_incidence",
    eligibility_component=None,
)
```

### Further examples

<!-- TODO: add further examples with different grouping and representation methods -->

## Documentation

In order to contribute to the documentation you can simply install **SPAGAT** using `conda` and `pip` and `conda activate spagat` as described above.

Then, you can make the HTML documentation locally by changing the directory from the **SPAGAT** root directory to `spagat/doc`:

```bash
cd doc
```

Finally, run `sphinx` to create the HTML files in `spagat/doc/build/html`:

```bash
sphinx-build -d build/doctrees source build/html
```

You can then inspect the documentation for example by running it locally using the [VS Code Extension Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer).

Therefore, install VS code and the extension, right-click on the `index.html` and select `Open with Live Server`.

Voilà, your documentation should appear on your web browser.

PS: This documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/), with the extensions [sphinx-autodoc-typehints](https://pypi.org/project/sphinx-autodoc-typehints/) (to derive types from type annotations) and [sphinx.ext.napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) (to enable NumPy style).

PPS: Thanks to [recommonmark](https://recommonmark.readthedocs.io/en/latest/index.html) the documentation is mostly written in markdown.