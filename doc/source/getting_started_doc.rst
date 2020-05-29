###############
Getting started
###############

******************************
Purpose and Vision of SPAGAT
******************************

As energy system analyses and required energy system optimizations become more and more complex, computational tractability becomes an issue. 

Therefore, SPAGAT was created to find an optimal compromise between accurately representing the (energy system) data for the model regions and reducing the resulting computational complexity to ensure computational tractability.

In German we might refer to this "balancing act" as "Spagat", hence the name.

********
About Us
********

.. image:: https://www.fz-juelich.de/iek/iek-3/DE/_Documents/Pictures/IEK-3Team_2019-02-04.jpg?__blob=poster
    :target: https://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html
    :alt: Abteilung TSA
    :align: center

We are the `Techno-Economic Energy Systems Analysis <http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html>`_
department at the `Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)
<http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html>`_ belonging to the Forschungszentrum Jülich. Our
interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and
system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and
costs of energy systems. The results are used for performing comparative assessment studies between the various systems.
Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s
greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and
by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

**Contributions and Users**

Within the BMWi funded project `METIS <http://www.metis-platform.net/>`_ we develop together with the RWTH-Aachen
`(Prof. Aaron Praktiknjo) <http://www.wiwi.rwth-aachen.de/cms/Wirtschaftswissenschaften/Die-Fakultaet/Institute-und-Lehrstuehle/Professoren/~jgfr/Praktiknjo-Aaron/?allou=1&lidx=1>`_,
the EDOM Team at FAU `(PD Lars Schewe) <http://www.mso.math.fau.de/de/edom/team/schewe-lars/dr-lars-schewe>`_ and the
`Jülich Supercomputing Centre (JSC) <http://www.fz-juelich.de/ias/jsc/DE/Home/home_node.html>`_ new methods and models
within FINE.

.. image:: http://www.metis-platform.net/metis-platform/DE/_Documents/Pictures/projectTeamAtKickOffMeeting_640x338.jpg?__blob=normal
    :target: http://www.metis-platform.net
    :alt: METIS Team
    :align: center

Dr. Martin Robinius is teaching a `course <https://www.campus-elgouna.tu-berlin.de/energy/v_menu/msc_business_engineering_energy/modules_and_curricula/project_market_coupling/>`_
at TU Berlin in which he is introducing FINE to students.

************
Installation
************

In the following, instructions for installing and using the SPAGAT framework on Windows are given. The installation
instructions for installing and using SPAGAT on Linux/macOS systems are however quite similar and can be, hopefully
easily, derived from the instructions below.

**Python installation**

SPAGAT runs on Python 3 platforms (i.e. Anaconda). Currently, it is advised not to use a Python version exceeding
Python 3.6. Note: When installing the Python platform Anaconda the options

    $ Add Anaconda to the system PATH environment variable

is available in the advanced installation option. When selecting this options, the environment variables for Python,
pip, jupyter etc. are remotely set and do not have to be manually set.

A development platform which can be used to work with/on the code and which comes with Anaconda is Spyder.
Other development platforms are PyCharm or Visual Studio Code.

**SPAGAT installation**

Download or clone a local copy of the repository to your computer

    $ git clone tbd

and install the required dependencies via conda:

    $ tbd

Finally, install SPAGAT in the folder where the setup.py is located using

    $ pip install -e .

or install directly via python as

    $ python setup.py install

**Installation of additional packages**

Install FINE, to enable energy system optimization...

tbd