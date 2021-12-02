# %% [markdown]
# # DSM (BETA Phase!)

# %%
import functions
import FINE as fn

# %% [markdown]
# ## Get basic energy system data

# %%
dsm_test_esM_ = functions.dsm_test_esM()

# %% [markdown]
# ## Run without DSM for reference purposes

# %%
"""
Given a one-node system with two generators, check whether the load and generation is shifted correctly in both
directions with and without demand side management.
"""

functions.run_esM_without_DSM()

# %% [markdown]
# ## Run with DSM (tFwd=tBwd=1)
# w/o & w/- time series aggregation

# %%
functions.run_esM_with_DSM(timeSeriesAggregation=False, tBwd=1, tFwd=1)

# %%
functions.run_esM_with_DSM(True, numberOfTypicalPeriods=23, tBwd=1, tFwd=1)

# %% [markdown]
# ## Run with DSM (tFwd=2, tBwd=0)

# %%
functions.run_esM_with_DSM(timeSeriesAggregation=False, tBwd=0, tFwd=2)

# %% [markdown]
# ## Running into trouble
#
# If tBwd + tFwd +1 is not divisor of total number of timesteps (24), infeasibilities can occur. These can be fixed with relaxing the dynamic state of charge constaints. Less notable for a larger amount of time steps!

# %% tags=["nbval-skip"]
functions.run_esM_with_DSM(timeSeriesAggregation=False, tBwd=2, tFwd=2)

# %%
