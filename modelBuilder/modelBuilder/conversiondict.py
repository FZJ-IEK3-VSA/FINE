#%%
import pandas as pd
import pprint

ted_path = r"/fast/home/a-burdack/modelBuilder/modelBuilder/data/technoeconomic_params.csv"
# ted_data_check = pd.read_csv(ted_path,delimiter=";")
# ted_data_check

#%%

ted_data = pd.read_csv(
    ted_path,
    delimiter=";",
    header=0, 
    usecols=["component", "attribute", "region", "investment_period", "conversion_commodity", "values"],
    )

for i in range(len(ted_data)): 
    if not (ted_data.iloc[i,3]=="constant" or ted_data.iloc[i,3]=="example"):
        ted_data.iloc[i,3] = int(ted_data.iloc[i,3])

ted_data_commodityConversionFactors = ted_data[ted_data["attribute"]=="commodityConversionFactors"]
ted_data = ted_data[ted_data["attribute"]!="commodityConversionFactors"]
ted_data = ted_data.drop("conversion_commodity",axis=1)

# add opexPerCapacity
ted_data_investPerCap = ted_data[ted_data["attribute"]=="investPerCapacity"]
ted_data_opexFix = ted_data[ted_data["attribute"]=="opexFix"]
ted_data_opexFix.reset_index(inplace=True, drop=True) 
ted_data_investPerCap.reset_index(inplace=True, drop=True) 

# test: opexFix should not be more regionalized then investment costs
for i in range(len(ted_data_opexFix)):
    component_opexFix = ted_data_opexFix.loc[i,"component"]
    region_opexFix =    ted_data_opexFix.loc[i,"region"]
    if not (region_opexFix == "world" or region_opexFix == "example"):
        for j in range(len(ted_data_investPerCap)):
                component = True if ted_data_investPerCap.loc[j, "component"] ==component_opexFix else False
                attribute = True if ted_data_investPerCap.loc[j, "attribute"] =="investPerCapacity" else False
                region =    True if ted_data_investPerCap.loc[j, "region"]    ==region_opexFix else False
                if component is True and attribute is True and region is True:
                    print("check")
                    break
        if component is True and attribute is True and region is True:
            continue
        else:
            raise ValueError(f"opexFix entries in techno-economic data csv for {component_opexFix} can not be more regionalized that investmentPerCapacity!")

# iterate over all investPerCap lines in ted_data
for i in range(len(ted_data_investPerCap)):
    # create intermediate df --> will contain new line with opexPerCapacity for every investPerCapacity entry 
    df = pd.DataFrame(columns=list(list(ted_data_investPerCap.columns)))
    # iterate over columns of ted_data: component, attribute, region, investment period, values
    for column in list(ted_data_investPerCap.columns):
        if column == "values":
            # filter until regions and investment periods
            ted_data_filter = ted_data[ted_data["component"]==ted_data_investPerCap.loc[i,"component"]]
            ted_data_filter_attr = ted_data_filter[ted_data_filter["attribute"]=="opexFix"]
            ted_data_filter_attr_reg = ted_data_filter_attr[ted_data_filter_attr["region"]==ted_data_investPerCap.loc[i,"region"]]
            ted_data_filter_fin = ted_data_filter_attr_reg[ted_data_filter_attr_reg["investment_period"]==ted_data_investPerCap.loc[i,"investment_period"]]
            # investment period not available
            if ted_data_filter_fin.empty:
                ted_data_filter_fin = ted_data_filter_attr_reg[ted_data_filter_attr_reg["investment_period"]=="constant"]
            if ted_data_filter_fin.empty:
                ted_data_filter_fin = ted_data_filter_attr_reg[ted_data_filter_attr_reg["investment_period"]=="example"]
            # region not available, try world and investment period
            if ted_data_filter_fin.empty:
                ted_data_filter_attr_world = ted_data_filter_attr[ted_data_filter_attr["region"]=="world"]
                ted_data_filter_fin = ted_data_filter_attr_world[ted_data_filter_attr_world["investment_period"]==ted_data_investPerCap.loc[i,"investment_period"]]
                # try if const available
                if ted_data_filter_fin.empty:
                    ted_data_filter_fin = ted_data_filter_attr_world[ted_data_filter_attr_world["investment_period"]=="constant"]
                # try if example available
                if ted_data_filter_fin.empty:
                    ted_data_filter_fin = ted_data_filter_attr_world[ted_data_filter_attr_world["investment_period"]=="example"]
            # world not available, try example and investment period
            if ted_data_filter_fin.empty:
                ted_data_filter_attr_example = ted_data_filter_attr[ted_data_filter_attr["region"]=="example"]
                ted_data_filter_fin = ted_data_filter_attr_example[ted_data_filter_attr_example["investment_period"]==ted_data_investPerCap.loc[i,"investment_period"]]
                # try if const available
                if ted_data_filter_fin.empty:
                    ted_data_filter_fin = ted_data_filter_attr_example[ted_data_filter_attr_example["investment_period"]=="constant"]
                # try if example available
                if ted_data_filter_fin.empty:
                    ted_data_filter_fin = ted_data_filter_attr_example[ted_data_filter_attr_example["investment_period"]=="example"]  
            # set opexFix value
            opex_fix = ted_data_filter_fin[column].iloc[0]
            df.loc[0,column] = float(ted_data_investPerCap.loc[i,column]) * float(opex_fix)
        elif column == "attribute":
            df.loc[0,column] = "opexPerCapacity"
        else:
            df.loc[0,column] = ted_data_investPerCap.loc[i,column]
    # for loop finished, df line with operationPerCapacity loaded and can be added to ted_data 
    ted_data = pd.concat([ted_data,df])
# opexfix is now replaced by opexPerCapacity and can be removed from ted data
ted_data = ted_data[ted_data["attribute"]!="opexFix"]

#ted_data.to_excel("/fast/home/a-burdack/modelBuilder/modelBuilder/data/ted_data.xlsx",index=False) # to test 

# set index
ted_data.set_index(["component", "attribute", "region", "investment_period"], inplace=True)
# turn multiindex df into multiindex series --> only possible with one column
ted_data = ted_data.squeeze()
# add commodityConversionFactors dict component, region and investment specific by using for loop
list_conv_components = list(ted_data_commodityConversionFactors["component"].unique())
for component in list_conv_components:
    list_conv_components_regions = list(ted_data_commodityConversionFactors.loc[ted_data_commodityConversionFactors["component"]==component,"region"].unique())
    for region in list_conv_components_regions:
        ted_data_commodityConversionFactors_filtered = ted_data_commodityConversionFactors.loc[ted_data_commodityConversionFactors["component"]==component]
        ted_data_commodityConversionFactors_filtered = ted_data_commodityConversionFactors_filtered.loc[ted_data_commodityConversionFactors_filtered["region"]==region]
        list_conv_components_investment_periods = list(ted_data_commodityConversionFactors_filtered["investment_period"])
        for investment_period in list_conv_components_investment_periods:
            ted_data_commodityConversionFactors_filtered = ted_data_commodityConversionFactors_filtered.loc[ted_data_commodityConversionFactors_filtered["investment_period"]==investment_period]
            df_component_conv_factors = ted_data_commodityConversionFactors[ted_data_commodityConversionFactors["component"]==component]
            # build converionfactor dict
            conversion_dict = df_component_conv_factors.set_index('conversion_commodity')['values'].to_dict()
            ted_data[component,"commodityConversionFactors",region,investment_period] = conversion_dict
            
pprint.pprint(ted_data)

#%%
x=1








#%%

component   = "electrolyzer_pem_compressor"
attribute   = "commodityConversionFactors"
regions     = ["BEL","DEU"]
investment_periods = [2020,2030] # in modelmanager: self.esM.investmentPeriodNames
return_format = "series_years"

def get_data(component, attribute, regions, investment_periods)->dict:
    if return_format == "dict":
        dictionary= {
            ip: {
                region: _iterate_available_data(component, attribute, region, ip) 
                for region in regions
            }
        for ip in investment_periods
        }
        return dictionary
    elif return_format == "series_years":
        series={
            ip: _iterate_available_data(component, attribute, "world", ip)
            for ip in investment_periods    
        }
        return series
    elif return_format == "series_regions":
        series={
            region: _iterate_available_data(component, attribute, region, "constant")
            for region in regions    
        }
        return series
    elif return_format == "value":
        value= _iterate_available_data(component, attribute, "world", "constant")
        return value
    else:
        raise ValueError("Error: When calling the get_data function, the return_format parameter can only be 'dict', 'series_years', 'series_regions', or 'value'.")

def _iterate_available_data(component, attribute, region, ip)-> dict|float|int|str:
    try: # try region
        try: # try investment period
            return ted_data[component][attribute][region][ip]
        except: # try investment period
            try: # try investment period
                return ted_data[component][attribute][region]["constant"]
            except: # try investment period
                return ted_data[component][attribute][region]["example"]
    except: # try region
        try: # try region
            try: # try investment period
                return ted_data[component][attribute]["world"][ip]
            except: # try investment period
                try: # try investment period
                    return ted_data[component][attribute]["world"]["constant"]
                except: # try investment period
                    return ted_data[component][attribute]["world"]["example"]
        except: # try region
            try: # try investment period
                return ted_data[component][attribute]["example"][ip]
            except: # try investment period
                try: # try investment period
                    return ted_data[component][attribute]["example"]["constant"]
                except: # try investment period
                    return ted_data[component][attribute]["example"]["example"]
  

##############################################################################################
# Test get_data()                                                                           ##
##############################################################################################



FINE_args= {
    'commodityConversionFactors': get_data(component, 'commodityConversionFactors', regions, investment_periods)
}
pprint.pprint(FINE_args)

# %%
