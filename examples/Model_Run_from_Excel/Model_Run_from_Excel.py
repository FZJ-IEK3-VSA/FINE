# %% [markdown]
# # Using Excel for parameter input and model runs
#
# Besides using a python script or a jupyter notebook it is also possible to read and run a model using excel files.
#
# The energySystemModelRunFromExcel() function reads a model from excel, optimizes it and stores it to an excel file.
#
# The readEnergySystemModelFromExcel() function reads a model from excel.
#
# The model run can also be started on double-klicking on the run.bat Windows batch script in the folder where this notebook is located (still requires that a Python version, the FINE package and an optmization solver are installed).
#
# Moreover, it is possible to run the model inside the Excel file with a VBA Macro (see scenarioInputWithMacro.xlsx)

# %% [markdown]
# # Import FINE package
#

# %%
import FINE as fn

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Call energySystemModelRunFromExcel function

# %% [markdown]
# ### Read model from excel file, optimize and store to excel file
# Checkout the output excel file in the folder where this notebook is located

# %% tags=["nbval-skip"]
esM = fn.energySystemModelRunFromExcel()

# %% [markdown]
# ### Read only

# %%
esM, esMData = fn.readEnergySystemModelFromExcel()

# %%
