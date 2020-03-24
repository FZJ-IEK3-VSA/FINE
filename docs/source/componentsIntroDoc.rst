**********************
Energy System Modeling
**********************

For creating your own energy system model, you start adding different components such as sources (e.g.,
photovoltaic panels), storage units (e.g., batteries), conversion units (e.g., heat pumps), transmission units
(e.g., electricity grids) and sinks (e.g., households with electricity or heat demands) to your model using
the framework. Then FINE will find the optimal sizing and an optimal unit commitment of all components for you.
Here, FINE can consider a user-defined number of regions and number of time steps, i.e. it models a spatially
and temporally resolved energy system.

The five named component types, i.e., sources, sinks, storage, conversion and transmission units are the main
components of the energy system model, which can also be seen in the python package description (Source and Sink
classes, Storage class, Conversion class, Transmission class). Shared features of these components are provided
in the Component class. The actual optimization problem is created in the EnergySystemModel class.

It is important to have a clear idea of what you know about your energy system beforehand and what you want to
optimize. Two types of "values" of must be distinguished:

* Parameters: Known values must be specified in the component specifications and are called parameters.
* Variables: Values to be optimized are called variables. Their optimal values are determined during the model run.

