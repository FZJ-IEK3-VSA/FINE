---
title: "ETHOS.FINE: A Framework for Integrated Energy System Assessment"
tags:
  - Python
  - energy systems modeling
  - optimization
  - energy
authors:
  - name: Theresa Klütz
    orcid: 0000-0003-1792-5598
    affiliation: "1"
  - name: Kevin Knosala
    orcid: 0000-0002-8825-5934
    affiliation: "1, 2"
  - name: Johannes Behrens
    orcid: 0009-0003-2396-0282
    affiliation: "1, 2"
  - name: Rachel Maier
    orcid: 0000-0003-0664-3886
    affiliation: "1, 2"
  - name: Maximilian Hoffmann
    orcid: 0000-0003-1232-8110
    affiliation: "1"
  - name: Noah Pflugradt
    orcid: 0000-0002-1982-8794
    affiliation: "1"
  - name: Detlef Stolten
    orcid: 0000-0002-1671-3262
    affiliation: "1, 2"
affiliations:
  - name: Forschungszentrum Jülich GmbH, Institute of Climate and Energy Systems – Jülich Systems Analysis, 52425 Jülich, Germany
    index: 1
  - name: RWTH Aachen University, Chair for Fuel Cells, Faculty of Mechanical Engineering, 52062 Aachen, Germany
    index: 2
date: 13 January 2025
bibliography: FINE-literature.bib
---

# Summary

The decarbonization of energy systems worldwide requires a transformation in their design and operation across all sectors, including the residential and commercial, industrial, and transportation sectors. Energy system models are often used to assess these changes. These models provide scenarios for potential future system designs and show how new technologies and infrastructure will meet future energy demand. Thus, they support investment decisions and policy making. The Python-based Framework for Integrated Energy System Assessment (`ETHOS.FINE`) is a software package that provides a toolbox for modeling, analyzing and evaluating such energy systems using mathematical optimization. 

`ETHOS.FINE` is not limited to a single instance of energy systems. Instead, it can be freely adapted to consider multiple commodities, regions, time steps and investment periods. The optimization objective is to minimize the net present value of the system and is subject to technical and environmental constraints. If only one investment period is considered, the net present value equals the total annual costs of the system. The generic object-oriented implementation allows for arbitrary spatial scales and number of regions – from the local level, e.g., individual buildings, to the regional one, e.g., districts or industrial sites, to the national and international levels. 

To reduce model size and complexity, the spatial technological resolution can be aggregated using built-in aggregation methods that are described in @Patil_2022. These methods include the aggregation of modeled regions and their given input parameters, and the grouping of technologies within the regions, e.g. to aggregate multiple time series for variable renewable energy sources. This also applies to the temporal resolution of the model. Apart from using the full temporal resolution defined by the input data, integrated time series aggregation methods using the built-in Python package tsam[^1] allow to reduce the complexity of the model and its computation time [@Hoffmann_2022], while still allowing the flexibility of seasonal storage technologies, despite the reduced model complexity [@Kotzur_2018]. `ETHOS.FINE` supports the aggregation of time steps to typical periods, the segmentation of the time series and the combination of both. The aggregation methods, spatial and temporal aggregation, can be used directly in `ETHOS.FINE` by calling the corresponding functions. In addition, `ETHOS.FINE` allows the investigation of transformation paths by considering multiple investment periods in a perfect foresight approach, as well as the stochastic optimization for a single year optimization with multiple sets of input parameters, e.g., changing energy demand forecasts or weather conditions, to find more robust energy system designs. 

[^1]: tsam - Time Series Aggregation Module, https://github.com/FZJ-IEK3-VSA/tsam,

# Methodology

`ETHOS.FINE` comprises seven main classes: The EnergySystemModel class can be seen as the container of the model, collecting all relevant input data for its setup. All technologies to be considered are added to this container.
The Component class contains the parameters, variables, and constraints common to all system components, such as capacity limits and restrictions on the operation of technologies. The five classes - Source, Sink, Conversion, Transmission, and Storage - provide the functionality to model energy generation and consumption, conversion processes, energy storage for later use, and energy transfer between regions. Each class introduces a specific set of constraints that is added to the optimization program. The specific set of constraints is described in the additional model classes. Supplemental subclasses provide additional component features, e.g., the ability to model partial load behavior and ramping constraints for power plants. The described structure is shown in \autoref{fig:finestructure}. Objects of the Source, Sink, Conversion and Storage classes are assigned to the modeled locations, which are represented as nodes in the model. Transmission class objects are assigned to the connections between the nodes. 

![a) Structure of the main classes in `ETHOS.FINE`. Additional model classes contain the definition of the specific variables, sets and constraints for each class to build the optimization model. b) Simplified representation of the model structure in `ETHOS.FINE` with the corresponding component classes. Each node represents a region that can exchange goods and energy carriers via transmission components (based on @gross2023_thesis). \label{fig:finestructure}](ETHOS-FINE-Schema.png)

The energy system model can be set up as a linear program (LP), a quadratic program (QP), or a mixed integer linear program (MILP), depending on the chosen representation of the added components. The optimization program is written as a Pyomo[^2] instance to allow a flexible choice of solvers, i.e. `ETHOS.FINE` optimizes energy systems using both, commercial and open source solvers. In a future version, other Python libraries, e.g. linopy [@Hofmann2023], may be integrated to improve the setup of the optimization program. 
Depending on the spatial and temporal resolution of the modeled system, the input parameters are primarily given as Pandas.DataFrames[^3] with regions and time steps serving as indices and columns. The model output provides detailed information on the investment required in each region for the installation and operation of the selected components, as well as the temporally-resolved operation of each component. This also includes charging and discharging of storage components and commodity flows between regions via transmission components. In addition, the framework provides plotting options for spatially and temporally resolved results. Model input and output can be saved to netCDF files to support reproducibility. 

[^2]: Pyomo, Pyoton Optimization Modeling Language, https://pyomo.org/,
[^3]: Pandas, Python Data Analysis Library, https://pandas.pydata.org/,

# Statement of need

`ETHOS.FINE` offers a unique generic model setup with a high degree of freedom for model developers. 
Beyond energy system models, its generic implementation allows the modeling of all kinds of optimization problems, such as material flows and resource consumption or conversion as part of life cycle analysis. `ETHOS.FINE` was developed to provide a flexible techno-economic analysis tool to analyze the energy transition on all levels of interest especially with regards to sector-coupled systems. The software exhibits many of the features described by @Groissbock2019 and is under constant development. Its code is openly accessible on GitHub which allows for contributions and feedback from a wider modeling community. The use cases described in the next section demonstrate the broad range of analyses that can be conducted with the tool.

There are several other open-source available energy system modeling frameworks that are also implemented in Python, e.g. Calliope [@Pfenninger2018], PyPSA [@PyPSA], oemof [@oemof] and CLOVER [@Sandwell2023]. The tools are used for similar use cases, providing different sets of functionalities and possible analysis tools. They differ mainly in the setup process of the models. `ETHOS.FINE` offers a highly flexible alternative to these tools. 

`ETHOS.FINE` is designed to be used by researchers, students, and for teaching purposes in the field of energy system modeling. In particular, its exceptional capabilities with respect to complexity reduction [@kotzur_modelers_2021] using spatial [@Patil_2022] and temporal aggregation [@Hoffmann_2020; @Hoffmann_2021; @Hoffmann_2022; @hoffmann_temporal_2023], as well as heuristics for dealing with MILPs [@Kannengiesser2019; @singh_budget-cut_2022] open a wide field of applications from small to global scale energy system models.
For newcomers who are not familiar with programming, it also has the flexibility to set up models by using Excel files, the usability of which is described in one of the example Jupyter notebooks published in the GitHub repository. 

# Examples for previous usage

`ETHOS.FINE` has been used in various studies for energy system analyses at different scales, taking advantage of its ability to dynamically adapt to computational complexity. First applications can be found in @Welder2018 and @Welder2019: The authors analyzed hydrogen-to-electricity reconversion pathways in a multi-regional energy system model implemented in `ETHOS.FINE` for the northern part of Germany. Later, @Welder2022 and @gross2023_thesis used the framework to model the future energy system of Germany with a high spatial resolution and thereby to investigate the need for new infrastructure. @Caglayan2019_1 built an `ETHOS.FINE` model of the European energy system, and analyzed the influence of varying weather years on the cost-optimal system design based on 100% use of renewable energy source. Their findings are also used to determine a robust system design based on variable renewable energy sources, ensuring security of supply for a wide range of weather years [@Caglayan2019_2]. @knosala_2021 evaluated hydrogen technologies in residential buildings in a multi-commodity, single-building model. The building model from this work was also used for a sensitivity analysis of energy carrier costs for the application of hydrogen in residential buildings [@knosala_2022]. @Spiller_2022 analyzed the carbon emission reduction potentials of hotels on energy self-sufficient islands. More recently, @weinand_low-carbon_2023 used the framework to assess the Rhine Rift Valley for its potential for lithium extraction from deep geothermal wells. Meanwhile, @jacob_future_2023 investigated the potential of Carnot batteries in the German electricity system. @busch_role_2023 analyzed the role of liquid hydrogen, also on a national scale, while @franzmann_green_2023 examined the cost potential of green hydrogen for global trade. These examples illustrate the variety of applications that can be addressed by `ETHOS.FINE`.

`ETHOS.FINE` is part of the Energy Transformation paTHway Optimization Suite (`ETHOS`)[^4], a collection of modeling tools developed by the Institute of Energy and Climate Research - Jülich Systems Analysis at Forschungszentrum Jülich. `ETHOS` offers a holistic view of energy systems at arbitrary scales, providing tools for geospatial analysis of renewable energy potential, time series simulation tools for residential and industrial sector, discrete choice models for the transportation sector, modeling of global energy supply routes, and local infrastructure assessments, among others. An example of the use of this model suite can be found in @Stolten2022. The model framework `ETHOS.FINE` serves as a basis for several model implementations within `ETHOS`, e.g. for the optimization programs to analyse of the transformation of single buildings, the transport sector, and the local, German, European and global energy system, or to determine the cost potential of global hydrogen production. 

[^4]: ETHOS - Energy Transformation paTHway Optimization Suite, https://www.fz-juelich.de/en/iek/iek-3/expertise/model-services

# Acknowledgements

We acknowledge contributions from Lara Welder, Robin Beer, Julian Belina, Toni Busch, Arne Burdack, Henrik Büsing, Dilara Caglayan, Philipp Dunkel, David Franzmann, Patrick Freitag, Maike Gnirß, Thomas Grube, Lars Hadidi, Heidi Heinrichs, Jason Hu, Shitab Ishmam, Timo Kannengießer, Sebastian Kebrich, Leander Kotzur, Stefan Kraus, Felix Kullmann, Dane Lacey, Jochen Linssen, Nils Ludwig, Lilly Madeisky, Drin Marmullaku, Gian Müller, Lars Nolting, Kenneth Okosun, Olalekan Omoyele, Shruthi Patil, Jan Priesmann, Oliver Rehberg, Stanley Risch, Martin Robinius, Thomas Schöb, Julian Schönau, Kai Schulze, Bismark Singh, Andreas Smolenko, Lana Söltzer, Maximilian Stargardt, Peter Stenzel, Chloi Syranidou, Johannes Thürauf, Henrik Wenzel, Lovindu Wijesinghe, Christoph Winkler, Bernhard Wortmann and Michael Zier during the development of this software package.

This work was initially supported by the Helmholtz Association under the Joint Initiative "Energy System 2050 - A Contribution of the Research Field Energy." The authors also gratefully acknowledge financial support by the Federal Ministry for Economic Affairs and Energy of Germany as part of the project METIS (project number 03ET4064, 2018-2022).

This work was supported by the Helmholtz Association under the program `Energy System Design`. 

# References
