#%%
from modelBuilder import outputDataHandler

model_base_folder = "/storage_cluster/projects/2022-a-burdack-phd/workspace/modelresults/testmodelbuilder"  # set your result folder

postpro = outputDataHandler.OutputHandler(
    model_base_folder=model_base_folder,
    xr_dss=None,
    regions_shp=None,
    transmission_shp=None,
)


# data = postpro.get_var_per_region(
#     variable="operationVariablesOptimum",
#     component="electricity_grid",
#     agg="sum",
# )

# #%%
# postpro.plotOperationColorMap(
#     self = postpro,
#     data=data,
#     zlabel="Operation Elec Grid fixed [GWh]",
#     fileName="zz_elec_grid.png",
#     shading="flat",
#     dpi=200,
#     pad=0.12,
#     aspect=50,
#     fraction=0.05,
#     orientation="vertical",
# )

# #%%
# data = postpro.get_var_per_region(
#     variable="operationVariablesOptimum",
#     component="ofpv_fixed",
#     agg="sum",
# )

# #%%
# postpro.plotOperationColorMap(
#     self = postpro,
#     data=data,
#     zlabel="Operation OFPV fixed [GWh]",
#     fileName="zz_ofpv_fixed.png"
# )
# #%%

# data = postpro.get_var_per_region(
#     variable="stateOfChargeOperationVariablesOptimum",
#     component="battery_LIIon",
#     agg="sum",
# )
# postpro.plotOperationColorMap(
#     data=data,
#     zlabel="SOC Battery LI-Ion [GWh]",
#     fileName="zz_battery_LIIon.png"
# )

postpro.store_standard_evaluation()
postpro.store_default_plots()
print("postprocessed", flush=True)