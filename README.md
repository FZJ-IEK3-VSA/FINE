<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://www.fz-juelich.de/SharedDocs/Bilder/IBG/IBG-3/DE/Plant-soil-atmosphere%20exchange%20processes/INPLAMINT%20(BONARES)/Bild3.jpg?__blob=poster" alt="Forschungszentrum Juelich Logo" width="230px"></a> 

# FINE - Framework for Integrated Energy System Assessment

The FINE python package provides a framework for modeling, optimizing and assessing energy systems. With the provided framework, systems with multiple regions, commodities and time steps can be modeled. Target of the optimization is the minimization of the total annual cost while considering technical and enviromental constraints. Besides using the full temporal resolution, an interconnected typical period storage formulation can be applied, that reduces the complexity and computational time of the model.

If you want to use FINE in a published work, please [**kindly cite following publication**](https://www.sciencedirect.com/science/article/pii/S036054421830879X) which gives a description of the first stages of the framework. The python package which provides the time series aggregation module and its corresponding literatur can be found [**here**](https://github.com/FZJ-IEK3-VSA/tsam).

## Features
* representation of an energy system by multiple locations, commodities and time steps
* complexity reducing storage formulation based on typical periods


## Documentation

A "Read the Docs" documentation of FINE can be found [**here**](https://vsa-fine.readthedocs.io/en/latest/).

## Installation
You can directly install FINE via pip as follows

	pip install FINE

However in this case, the connection to this GitHub repository is not preserved. If you want to preserve it,
you can clone a local copy of the repository to your computer

	git clone https://github.com/FZJ-IEK3-VSA/FINE.git

Then install FINE via pip as follow
	
	cd FINE
	pip install . 
	
Or install directly via python as 

	python setup.py install
		
	
## Examples

A number of [**examples**](examples/) shows the capabilities of FINE.

## License

MIT License

Copyright (C) 2016-2018 Lara Welder (FZJ IEK-3), Jochen Linßen (FZJ IEK-3), Martin Robinius (FZJ IEK-3), Detlef Stolten (FZJ IEK-3)

You should have received a copy of the MIT License along with this program.
If not, see https://opensource.org/licenses/MIT


## About Us 
<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://fz-juelich.de/SharedDocs/Bilder/IEK/IEK-3/Abteilungen2015/VSA_DepartmentPicture_2017.jpg?__blob=normal" alt="Abteilung VSA" width="400px" style="float:center"></a> 

We are the [Process and Systems Analysis](http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html) department at the [Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)](http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the Forschungszentrum Jülich. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.


## Contributions and Users


Within the BMWi funded project [**METIS**](http://www.metis-platform.net/) we develop together with the RWTH-Aachen ([**Prof. Aaron Praktiknjo**](http://www.wiwi.rwth-aachen.de/cms/Wirtschaftswissenschaften/Die-Fakultaet/Institute-und-Lehrstuehle/Professoren/~jgfr/Praktiknjo-Aaron/?allou=1&lidx=1)), the EDOM Team at FAU ([**PD Lars Schewe**](http://www.mso.math.fau.de/de/edom/team/schewe-lars/dr-lars-schewe/)) and the [**Jülich Supercomputing Centre**](http://www.fz-juelich.de/ias/jsc/DE/Home/home_node.html) new methods and models within FINE.

<a href="http://www.metis-platform.net/"><img src="http://www.metis-platform.net/metis-platform/SharedDocs/Bilder/DE/kick-off.jpg?__blob=normal" alt="METIS Team" width="400px" style="float:center"></a> 

Dr. Martin Robinius is teaching a [**course**](https://www.campus-elgouna.tu-berlin.de/energy/v_menu/msc_business_engineering_energy/modules_and_curricula/project_market_coupling/) at TU Berlin in which he is introducing FINE to students.


## Acknowledgement

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>
