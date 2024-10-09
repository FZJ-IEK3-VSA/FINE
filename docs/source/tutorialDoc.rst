********
Tutorial
********

The repository of ETHOS.FINE provides a certain amount of different `examples <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples>`_ to get to know to the package. 
The examples are sorted by their complexity and should provide a good overview on the scope of the package. 

* `00_Tutorial <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/00_Tutorial>`_
    * In this application, an energy supply system, consisting of two regions, is modeled and optimized. Recommended as starting point to get to know to ETHOS.FINE.
* `01_1node_Energy_System_Workflow <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/01_1node_Energy_System_Workflow>`_
    * In this application, a single region energy system is modeled and optimized. The system includes only a few technologies. 
* `02_EnergyLand <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/02_EnergyLand>`_
    * In this application, a single region energy system is modeled and optimized. Compared to the previous examples, this example includes a lot more technologies considered in the system. 
* `03_Multi-regional_Energy_System_Workflow <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/03_Multi-regional_Energy_System_Workflow>`_
    * In this application, an energy supply system, consisting of eight regions, is modeled and optimized. The example shows how to model multi-regional energy systems. The example also includes a notebook to get to know the optional performance summary. The summary shows how the optimization performed.
* `04_Model_Run_from_Excel <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/04_Model_Run_from_Excel>`_
    * ETHOS.FINE can also be run by excel. This example shows how to read and run a model using excel files.
* `05_District_Optimization <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/05_District_Optimization>`_
    * In this application, a small district is modeled and optimized. This example also includes binary decision variables.
* `06_Water_Supply_System <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/06_Water_Supply_System>`_
    * The application cases of ETHOS.FINE are not limited. This application shows how to model the water supply system. 
* `07_NetCDF_to_save_and_set_up_model_instance <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/07_NetCDF_to_save_and_set_up_model_instance>`_
    * This example shows how to save the input and optimized results of an energy system Model instance to netCDF files to allow reproducibility.
* `08_Spatial_and_technology_aggregation <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/08_Spatial_and_technology_aggregation>`_
    * These two examples show how to reduce the model complexity. Model regions can be aggregated to reduce the number of regions (spatial aggregation). Input parameters are automatically adapted. Furthermore, technologies can be aggregated to reduce complexity, e.g. reducing the number of different PV components (technology aggregation). Input parameters are automatically adapted. 
* `9_Stochastic_Optimization <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/9_Stochastic_Optimization>`_
    * In this application, a stochastic optimization is performed. It is possible to perform the optimization of an energy system model with different input parameter sets to receive a more robust solution.
* `10_PerfectForesight <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/10_PerfectForesight>`_
    *  In this application, a transformation pathway of an energy system is modeled and optimized showing how to handle several investment periods with time-dependent assumptions for costs and operation.
* `11_Partload <https://github.com/FZJ-IEK3-VSA/FINE/tree/master/examples/11_Partload>`_
    * In this application, a hydrogen system is modeled and optimized considering partload behavior of the electrolyzer. 