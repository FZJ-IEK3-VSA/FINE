#%%
import fine.IOManagement.xarrayIO_2 as xrIO
import fine as fn
import os
import geokit as gk
import geopandas as gpd
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numbers
import glob

import yaml

from modelBuilder.data import data_folder as model_builder_data_folder
from modelBuilder.singletons import InputDataInfo, ModelTechnoEconomicData

class OutputHandler():

    def __init__(self,
        model_base_folder=None,
        xr_dss=None,
        regions_shp=None,
        transmission_shp=None,
        minValue=0.0001) -> None:
        
        #load data
        if isinstance(model_base_folder, str):
            assert os.path.isdir(model_base_folder)
            #a valid path is given, so set default paths
            self.esM_path = os.path.join(model_base_folder, "esM_optimized.nc4")
            self.model_base_folder = model_base_folder
            self.regions_shp_path = os.path.join(self.model_base_folder, "spatial_data", "regions.shp")
            self.transmissions_shp_path = os.path.join(self.model_base_folder, "spatial_data", "transmissions.shp")
            if not os.path.isfile(self.transmissions_shp_path): self.transmissions_shp_path = False
        else:
            #no path was given, all inputs need to be there to start
            assert isinstance(xr_dss, dict)
            assert isinstance(regions_shp, gpd.GeoDataFrame)
            assert isinstance(transmission_shp, gpd.GeoDataFrame) or transmission_shp == False #there must not be a transmission!

        #load esM object
        #if esM was handed use them, else load it from model_base_folder
        assert xr_dss is None or isinstance(xr_dss, dict)
        if xr_dss is None:
            self.loadESM(self.esM_path)
        else:
            self.xr_dss = xr_dss
        
        # dict containing all technologies per class (sink, source, ...)
        modeling_class_technologies_dict = {}
        for modeling_class in self.xr_dss["Input"].keys():
            modeling_class_technologies_dict[modeling_class] = list(self.xr_dss["Input"][modeling_class].keys())

        self.modeling_class_technologies_dict = modeling_class_technologies_dict

        #load regions:
        #if regions were handed use them, else load them from model_base_folder
        assert regions_shp is None or isinstance(regions_shp, gpd.GeoDataFrame)
        if regions_shp is None:
            self.regions_shp = gpd.read_file(self.regions_shp_path)
        else:
            self.regions_shp = regions_shp

        #load transmissions:
        assert transmission_shp is None or isinstance(transmission_shp, gpd.GeoDataFrame)
        #if transmissions were handed use them, else load them from model_base_folder
        if transmission_shp is None:
            if self.transmissions_shp_path:
                #load if file exists
                self.transmission_shp = gpd.read_file(self.regions_shp_path)
            else:
                #or set to false
                self.transmission_shp = False
        else:
            self.regions_shp = regions_shp

        print('ESM and all shapes loaded!')

        #load ted
        self.ted = yaml.load(open(os.path.abspath(os.path.join(model_builder_data_folder, "technoeconomic_params.yaml"))), Loader=yaml.FullLoader)

        self._get_component_dict()
        self._get_clustered_techs()
        self.minValue = minValue

        self.output_folder= os.path.join(self.model_base_folder, 'ESM_summary')
        os.makedirs(self.output_folder, exist_ok=True)

    def loadESM(self, nc4_path):
        '''loads the 
        '''
        print('Previous ESM will be loaded.', flush=True)
        self.xr_dss = xrIO.readNetCDFToDatasets(nc4_path)
        if not self.ResultsFeasible: print("\nWARNING, LOADED ESM HAS NO RESULTS!! \n")

    def ResultsFeasible(self):
        return not self.esM.getOptimizationSummary("SourceSinkModel") is None

    def _get_clustered_techs(self):
        '''get all technologies which where loaded with different clusters
            e.g.: "ofpv_fixed" as "ofpv_fixed__0", "ofpv_fixed__1", "ofpv_fixed__2")
        '''

        #get tech basenames from ted
        all_known_techs = []
        all_known_techs = list(set(ModelTechnoEconomicData().data.index.get_level_values("component").tolist()))
        all_known_techs.append("Lull")


        #find all clustered_techs
        clustered_techs = []
        components = list(self.comp_dict_results.keys())
        for known_tech in all_known_techs:
            count = 0
            for comp in components:
                #found tech in components
                if known_tech in comp:
                    count = count + 1
                #if found more than 0 its a clustered var
                if count > 0:
                    clustered_techs.append(known_tech)
                    break
                
        #find componentNames which are non clustered
        components_reduced = list(components)
        for component in components:
            for clustered_tech in clustered_techs:
                if clustered_tech in component:
                    components_reduced.remove(component)
                    break
        #join with clustered_techs 
        components_reduced.extend(clustered_techs)

        #make modell dict out of it
        componentNamesReduced = {}
        for comp in components_reduced:
            if not comp in clustered_techs:
                componentNamesReduced[comp] = self.comp_dict_results[comp]
            else:
                #find an example cluster:
                for comp2 in components:
                    if comp in comp2:
                        componentNamesReduced[comp] = self.comp_dict_results[comp2]

        self.clustered_techs = clustered_techs
        self.componentNamesReduced = componentNamesReduced

    def _get_component_dict(self):
        
        comp_dict_results = {}

        for ip in InputDataInfo().investment_period_names:
            for model in self.xr_dss["Results"][str(ip)].keys():
                for comp in self.xr_dss["Results"][str(ip)][model]:
                    comp_dict_results[comp] = model
        
        self.comp_dict_results = comp_dict_results

        comp_dict_input = {}

        for model in self.xr_dss["Input"].keys():
            for comp in self.xr_dss["Input"][model]:
                comp_dict_input[comp] = model
        
        self.comp_dict_input = comp_dict_input

    def get_var_per_region(self, variable, component:str, ip:str, agg="sum", apply_min_value = False, plot=False, kwargs_plot={}):
        '''extracting component variables per region

        Parameters
        ----------
        variable : str
            variable name: direct match
        component : str
            component name: will be searched for with regex:
            eg: ofpv_fixed --> will find [ofpv_fixed__0, ofpv_fixed__1, ofpv_fixed__2]
        agg : str or function or bool, optional
            The function how the variables shall be aggregated, should be passed as a 
            string-formatted default function name (e.g. 'sum' etc.). In special cases
            agg can be passed as a method as well (must be able to deal with NaN values).
            Can also be False so no aggregation is applied. By default 'max'.
        plot : bool, optional
            decide, if var should be plotted or returned, by default False
        kwargs_plot : dict, optional
            kwargs for plotLocationalColorMap, by default {}

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        '''
        func_dict={
            "sum":np.nansum, "nansum":np.nansum, "np.nansum":np.nansum,
            "mean":np.nanmean, "nanmean":np.nanmean, "np.nanmean":np.nanmean,
            "median":np.nanmedian, "nanmedian":np.nanmedian, "np.nanmedian":np.nanmedian,
            "min":np.nanmin, "nanmin":np.nanmin, "np.nanmin":np.nanmin,
            "max":np.nanmax, "nanmax":np.nanmax, "np.nanmax":np.nanmax,
        }
        if not agg is False:
            if callable(agg):
                # agg is a function already
                print(f"WARNING: It is recommended to pass a method name as string. Here, agg was passed as a method and will be applied as: {agg}")
            else:
                assert isinstance(agg, str), f"If agg is not passed as a callable method, it must be a str formatted method name."
                assert agg.lower() in func_dict.keys(), f"agg ('{agg.lower()}') was not found in available default methods. Pass as a callable method or better select from the following method names: {','.join(func_dict.keys())}"
                # replace method name str by actual function
                agg=func_dict[agg.lower()]
                
        #find components names:
        list_selected_comp_names = []
        for comp_name in self.comp_dict_results:
            if re.search(component, comp_name):
                list_selected_comp_names.append(comp_name)
        assert len(list_selected_comp_names) > 0, f"No components found for {component}. Components are: {[c for c in self.comp_dict_results.keys()]}"
        
        #find model type
        model_name_and_types = {c: self.comp_dict_results[c] for c in list_selected_comp_names}
        model_types = list(set(model_name_and_types.values()))
        # assert len(model_type) == 1, f"Multiple model types found for {list_selected_comp_names}: {model_type}"
        # model_type = model_type[0]

        if "TransmissionModel" in model_types: assert len(model_types) == 1

        selection = []
        for model_type in model_types:
            component_per_model_type = [comp for comp in model_name_and_types if model_name_and_types[comp]==model_type]
            for component_i in component_per_model_type:
                model_type_input = self.comp_dict_input[component_i]
                
                #try to get values from Results
                if variable in self.xr_dss["Results"][ip][model_type][component_i]:
                    selection_i = self.xr_dss["Results"][ip][model_type][component_i][variable].to_dataframe().T
                    selection_i.index = pd.MultiIndex.from_tuples([(component_i, variable)], names=["first", "second"])
                    assert len(selection_i) == 1, "Something failed internally!"

                #try to get values from Input
                elif variable in self.xr_dss["Input"][model_type_input][component_i]:
                    selection_i = self.xr_dss["Input"][model_type_input][component_i][variable].to_dataframe().T
                    selection_i.index = pd.MultiIndex.from_tuples([(component_i, variable)], names=["first", "second"])
                    assert len(selection_i) == 1, "Something failed internally!"
                else:
                    possible_vars = list(self.xr_dss["Input"][model_type_input][component_i]) + \
                        list(self.xr_dss["Results"][ip][model_type][component_i])
                    raise KeyError(f"Variable {variable} not found. Try with {str(possible_vars)}")
            
                selection.append(selection_i)
        selection = pd.concat(selection, axis=0)

        if apply_min_value:
            selection[selection.abs()<self.minValue] = 0


        if not agg:
            return selection.T
        else:
            try:
                values = agg(selection.values, axis=0) #aggregating with numpy is much faster, but agg does not necessarily work
                selection_agg=pd.Series(
                    values,
                    index=selection.columns,
                    dtype=float,
                )
            except:
                selection_agg = selection.apply(agg, axis=0) # if does not work, use pandas agg
            selection_agg.name = f"{variable}_{component}"

        if not plot:
            return selection_agg
        else:
            
            kwargs_plot_default = {
                "title": component,
                "zlabel": variable,
                "fileName": f"{variable}_{component}.png",
            }
            kwargs_plot_default.update(kwargs_plot)

            if "TransmissionModel" in model_types:
                output = self.plotTransmission(
                    data = selection_agg.unstack(),
                    ip=ip,
                    **kwargs_plot_default
                )
            else:
                output = self.plotLocationalColorMap(
                    data = selection_agg,
                    ip = ip,
                    **kwargs_plot_default
                )

            return output
        
    def plotLocationalColorMap(
        self,
        data,
        ip,
        crs="epsg:4326",
        indexColumn="locationID",
        figsize=6,
        fontsize=12,
        dpi=300,
        fileName=None,
        vmax=-1,
        vmin=0,
        cmap="viridis",
        title=False,
        zlabel=False,
    ):
        '''plotting the "LocationColorMap" from FINE a little bit better and with custom data!

        Parameters
        ----------
        data : pd.DataFrame
            output from "self.get_var_per_region"
        crs : str, optional
            SRS of the loaded shape file, by default "epsg:4326"
        indexColumn : str, optional
            name of the region colum of the shape file, by default "locationID"
        figsize : int or tuple
            figsize. if tuple, uses tuple. if number, its, by default 6
        fontsize : int, optional
            fontsize, by default 12
        dpi : int, optional
            dpi of th eplot, by default 300
        fileName : _type_, optional
            filename (without path!), by default None
        vmax : int, optional
            vmax, by default -1
        vmin : int, optional
            vmin, by default 0
        cmap : str, optional
            cmap, by default "viridis"
        title : bool, optional
            title, by default False
        zlabel : bool, optional
            zlabel, by default False

        Returns
        -------
        fig, ax
            matplotlib fig and ax
        '''

        # Get geodata
        locfilepath=self.regions_shp_path
        gdf = gpd.read_file(locfilepath).to_crs({"init": crs})


        # Make sure the data and gdf indices match
        ## 1. Sort the indices to obtain same order
        data.sort_index(inplace=True)
        gdf.sort_values(indexColumn, inplace=True)

        ## 2. Take first 20 characters of the string for matching. (In gdfs usually long strings are cut in the end)
        gdf[indexColumn] = gdf[indexColumn].apply(lambda x: x[:20])
        data.index = data.index.str[:20]

        gdf.loc[gdf[indexColumn] == data.index, "data"] = data.fillna(0).values

        vmax = gdf["data"].max() if vmax == -1 else vmax

        bounds = gdf.geometry.to_crs({"init": "epsg:3395"}).total_bounds
        dx = bounds[2] - bounds[0]
        dy = bounds[3] - bounds[1]

        if isinstance(figsize, numbers.Number):
            figsize = (figsize, dy/dx*figsize)

        if not zlabel:
            zlabel = ""

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")

        gdf.plot(
            column="data",
            ax=ax,
            cmap=cmap,
            edgecolor="black",
            alpha=1,
            linewidth=0.2,
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={
                'label': zlabel,
                'shrink': 0.5,
                }
        )

        cb_ax = fig.axes[1]
        cb_ax.tick_params(labelsize=fontsize)
        cb_ax.set_ylabel(zlabel, fontsize=fontsize)

        if title:
            ax.set_title(title)

        fig.tight_layout()

        if fileName:
            fileName = os.path.join(self.output_folder, f"standard_plots_{ip}", fileName)
            os.makedirs(os.path.dirname(fileName), exist_ok=True)
            plt.savefig(fileName, dpi=dpi, bbox_inches="tight")
            print("Figure saved to", fileName)
            fig.clear()
            plt.close(fig)

        return fig, ax

    def plotTransmission(
        self,
        data,
        ip,
        title,
        zlabel,
        unit="",
        loc0="bus_0",
        loc1="bus_1",
        crs="epsg:4326",
        color="k",
        loc=7,
        alpha=0.5,
        ax=None,
        fig=None,
        linewidth=10,
        figsize=6,
        fontsize=12,
        fileName=None,
        dpi=300,
    ):
        """
        Plot build transmission lines from a shape file.

        **Required arguments:**

        :param esM: considered energy system model
        :type esM: EnergySystemModel class instance

        :param compName: component name
        :type compName: string

        :param transmissionShapeFileName: file name or path to a shape file
        :type transmissionShapeFileName: string

        :param loc0: name of the column in which the location indices are stored (e.g. start/end of line)
        :type loc0: string

        :param loc1: name of the column in which the location indices are stored (e.g. end/start of line)
        :type loc1: string

        **Default arguments:**
        :param ip: investment periods
            |br| * the default value is 0
        :type ip: int

        :param crs: coordinate reference system
            |br| * the default value is 'epsg:3035'
        :type crs: string

        :param variableName: parameter for plotting installed capacity ('_capacityVariablesOptimum') or operation
            ('_operationVariablesOptimum').
            |br| * the default value is '_capacityVariablesOptimum'
        :type variableName: string

        :param color: color of the transmission line
            |br| * the default value is 'k'
        :type color: string

        :param loc: location of the legend in the plot
            |br| * the default value is 7
        :type loc: 0 <= integer <= 10

        :param alpha: transparency of the legend
            |br| * the default value is 0.5
        :type alpha: 0 <= scalar <= 1

        :param fig: None or figure to which the plot should be added
            |br| * the default value is None
        :type fig: matplotlib Figure

        :param ax: None or ax to which the plot should be added
            |br| * the default value is None
        :type ax: matplotlib Axis

        :param linewidth: line width of the plot
            |br| * the default value is 0.5
        :type linewidth: positive float

        :param figsize: figure size in inches
            |br| * the default value is (6,6)
        :type figsize: tuple of positive floats

        :param fontsize: font size of the axis
            |br| * the default value is 12
        :type fontsize: positive float

        :param save: indicates if figure should be saved
            |br| * the default value is False
        :type save: boolean

        :param fileName: output file name
            |br| * the default value is 'operation.png'
        :type fileName: string

        :param dpi: resolution in dots per inch
            |br| * the default value is 200
        :type dpi: scalar > 0
        """

        # Get geodata
        locfilepath=self.regions_shp_path
        gdf_location = gpd.read_file(locfilepath).to_crs({"init": crs})
        transfilepath=self.transmissions_shp_path
        gdf_transmission = gpd.read_file(transfilepath).to_crs({"init": crs})

        cap = data.fillna(0).copy()
        capMax = cap.max().max()
        if capMax == 0:
            return fig, ax
        cap = cap / capMax

        if ax is None:
            #make a baseplot
            #nice scaling of the plot
            bounds = gdf_location.geometry.to_crs({"init": "epsg:3395"}).total_bounds
            dx = bounds[2] - bounds[0]
            dy = bounds[3] - bounds[1]

            if isinstance(figsize, numbers.Number):
                figsize = (figsize, dy/dx*figsize)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.set_aspect("equal")
            ax.axis("off")

            gdf_location.boundary.plot(
                ax=ax,
                edgecolor="black",
                alpha=1,
                linewidth=1,
            )

        ax.axis("off")

        #actual plot
        for key, row in gdf_transmission.iterrows():
            capacity = cap.loc[row[loc0], row[loc1]]
            gdf_transmission[gdf_transmission.index == key].plot(ax=ax, color=color, linewidth=linewidth * capacity)

        #legend
        lineMax = plt.Line2D(
            range(1),
            range(1),
            linewidth=linewidth,
            color=color,
            marker="_",
            label="{:>4.4g}".format(capMax) + " " + unit,
        )
        lineMax23 = plt.Line2D(
            range(1),
            range(1),
            linewidth=linewidth * 2 / 3,
            color=color,
            marker="_",
            label="{:>4.4g}".format(capMax * 2 / 3) + " " + unit,
        )
        lineMax13 = plt.Line2D(
            range(1),
            range(1),
            linewidth=linewidth * 1 / 3,
            color=color,
            marker="_",
            label="{:>4.4g}".format(capMax * 1 / 3) + " " + unit,
        )

        leg = ax.legend(
            handles=[lineMax, lineMax23, lineMax13],
            prop={"size": fontsize},
            loc=loc,
            title=zlabel
        )
        leg.get_frame().set_edgecolor("white")
        leg.get_frame().set_alpha(alpha)

        if title:
            ax.set_title(title)
        
        fig.tight_layout()

        if fileName:
            fileName = os.path.join(self.output_folder, f"standard_plots_{ip}", fileName)
            os.makedirs(os.path.dirname(fileName), exist_ok=True)
            plt.savefig(fileName, dpi=dpi, bbox_inches="tight")
            print("Figure saved to", fileName)
            fig.clear()
            plt.close(fig)

        

        return fig, ax

    def plotOperationColorMap(
        self,
        data,
        ip,
        loc=None,
        locTrans=None,
        nbPeriods=365,
        nbTimeStepsPerPeriod=24,
        cmap="viridis",
        vmin=0,
        vmax=-1,
        xlabel="Days [d]",
        ylabel="Hours [h]",
        zlabel="",
        shading="flat",
        unit="GW",
        title=None,
        titleFontSize=None,
        figsize=(12, 4),
        fontsize=12,
        fileName="",
        xticks=None,
        yticks=None,
        xticklabels=None,
        yticklabels=None,
        monthlabels=False,
        dpi=200,
        pad=0.12,
        aspect=50,
        fraction=0.2,
        orientation="horizontal",
        **kwargs,
    ):
        """
        Plot operation time series of a component at a location.

        **Required arguments:**

        :param esM: considered energy system model
        :type esM: EnergySystemModel class instance

        :param compName: component name
        :type compName: string

        :param loc: location
        :type loc: string

        **Default arguments:**

        :param ip: investment period of transformation path analysis.
        :type ip: int

        :param locTrans: second location, required when Transmission components are plotted
            |br| * the default value is None
        :type locTrans: string

        :param nbPeriods: number of periods to be plotted
            |br| * the default value is 365
        :type nbPeriods: integer

        :param nbTimeStepsPerPeriod: time steps per period to be plotted (nbPeriods*nbTimeStepsPerPeriod=length of time
            series)
            |br| * the default value is 24
        :type nbTimeStepsPerPeriod: integer

        :param variableName: name of the operation time series. Checkout the component model class to see which options
            are available.
            |br| * the default value is '_operationVariablesOptimum'
        :type variableName: string

        :param cmap: heat map (color map) (see matplotlib options)
            |br| * the default value is 'viridis'
        :type cmap: string

        :param vmin: minimum value in heat map
            |br| * the default value is 0
        :type vmin: integer

        :param vmax: maximum value in heat map. If -1, vmax is set to the maximum value of the operation time series.
            |br| * the default value is -1
        :type vmax: integer

        :param xlabel: x-label of the plot
            |br| * the default value is 'day'
        :type xlabel: string

        :param ylabel: y-label of the plot
            |br| * the default value is 'hour'
        :type ylabel: string

        :param zlabel: z-label of the plot
            |br| * the default value is 'operation'
        :type zlabel: string

        :param figsize: figure size in inches
            |br| * the default value is (12,4)
        :type figsize: tuple of positive floats

        :param fontsize: font size of the axis
            |br| * the default value is 12
        :type fontsize: positive float

        :param save: indicates if figure should be saved
            |br| * the default value is False
        :type save: boolean

        :param fileName: output file name
            |br| * the default value is 'operation.png'
        :type fileName: string

        :param xticks: user specified ticks of the x axis
            |br| * the default value is None
        :type xticks: list

        :param yticks: user specified ticks of the ý axis
            |br| * the default value is None
        :type yticks: list

        :param xticklabels: user specified tick labels of the x axis
            |br| * the default value is None
        :type xticklabels: list

        :param yticklabels: user specified tick labels of the ý axis
            |br| * the default value is None
        :type yticklabels: list

        :param monthlabels: specifies if month labels are to be plotted (only works correctly if
            365 days are specified as the number of periods)
            |br| * the default value is False
        :type monthlabels: boolean

        :param dpi: resolution in dots per inch
            |br| * the default value is 200
        :type dpi: scalar > 0

        :param pad: pad parameter of colorbar
            |br| * the default value is 0.12
        :type pad: float

        :param aspect: aspect parameter of colorbar
            |br| * the default value is 15
        :type aspect: float

        :param fraction: fraction parameter of colorbar
            |br| * the default value is 0.2
        :type fraction: float

        :param orientation: orientation parameter of colorbar
            |br| * the default value is 'horizontal'
        :type orientation: float

        """
        isStorage = False


        if locTrans is None:
            if loc is None:
                timeSeries = data.unstack("space").sum(axis=1)
            else:
                assert loc in data.index.get_level_values(level=1)
                timeSeries = data.unstack("space")[loc]
        else:
            raise NotImplementedError("Do it yourself.")
            timeSeries = data["values"].loc[(self.compName, self.loc, locTrans)].values
        timeSeries = timeSeries / self.xr_dss["Parameters"].hoursPerTimeStep if not isStorage else timeSeries
        timeSeries = timeSeries.values

        try:
            timeSeries = timeSeries.reshape(nbPeriods, nbTimeStepsPerPeriod).T
        except ValueError as e:
            raise ValueError(
                "Could not reshape array. Your timeSeries has {} values and it is therefore not possible".format(
                    len(timeSeries)
                )
                + " to reshape it to ({}, {}). Please correctly specify nbPeriods".format(
                    nbPeriods, nbTimeStepsPerPeriod
                )
                + " and nbTimeStepsPerPeriod The error was: {}.".format(e)
            )
        vmax = timeSeries.max() if vmax == -1 else vmax

        fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

        if shading == "flat":
            ax.pcolormesh(
                range(nbPeriods + 1),
                range(nbTimeStepsPerPeriod + 1),
                timeSeries,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading=shading,
                **kwargs,
            )
        elif shading == "gouraud":
            ax.pcolormesh(
                range(nbPeriods + 0),
                range(nbTimeStepsPerPeriod + 0),
                timeSeries,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading=shading,
                **kwargs,
            )
        else:
            raise ValueError
        ax.axis([0, nbPeriods, 0, nbTimeStepsPerPeriod])
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        #ax.xaxis.set_label_position("bottom"), ax.xaxis.set_ticks_position("bottom")

        sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm1._A = []
        cb1 = fig.colorbar(
            sm1, ax=ax, pad=pad, aspect=aspect, fraction=fraction, orientation=orientation
        )
        cb1.ax.tick_params(labelsize=fontsize)

        #cb1.ax.set_position([0.2, 0.1, 0.6, 0.03])  # Set the position and size [left, bottom, width, height]

        if zlabel != "":
            #cb1.ax.set_xlabel(zlabel, size=fontsize)
            cb1.set_label(zlabel, size=fontsize)
        elif isStorage:
            cb1.ax.set_xlabel("Storage inventory" + " [" + unit + "]", size=fontsize) #TODO: adapt
        else:
            cb1.ax.set_xlabel("Operation" + " [" + unit + "]", size=fontsize) #TODO: adapt
        cb1.ax.xaxis.set_label_position("bottom")

        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
        if xticklabels:
            ax.set_xticklabels(xticklabels, fontsize=fontsize)
        if yticklabels:
            ax.set_yticklabels(yticklabels, fontsize=fontsize)

        if monthlabels:
            import datetime

            xticks, xlabels = [], []
            for i in range(1, 13, 2):
                xlabels.append(datetime.date(2050, i + 1, 1).strftime("%b"))
                xticks.append(datetime.datetime(2050, i + 1, 1).timetuple().tm_yday)
                ax.set_xticks(xticks), ax.set_xticklabels(xlabels, fontsize=fontsize)

        if title:
            if titleFontSize:
                plt.title(title, fontsize=titleFontSize)
            else:
                plt.title(title)

        fig.tight_layout()

        if fileName:
            fileName = os.path.join(self.output_folder, f"standard_plots_{ip}", fileName)
            os.makedirs(os.path.dirname(fileName), exist_ok=True)
            plt.savefig(fileName, dpi=dpi, bbox_inches="tight")
            print("Figure saved to", fileName)
            fig.clear()
            plt.close(fig)

        return fig, ax


    def plot_dilara(
        self,
        ip,
        area_varname = None,
        area_data = None,
        tranmission_varname = None,
        tranmission_data = None,
        pie_varname_type = None,
        pie_data = None,
    ):
        

        #1) do the area plot

        if area_varname and not area_data:
            raise NotImplementedError
            area_data = ""
        
        if area_data:
            fig, ax = self.plotLocationalColorMap(
                self,
                data=area_data,
                ip=ip,
                crs="epsg:4326",
                indexColumn="locationID",
                figsize=6,
                fontsize=12,
                dpi=300,
                fileName=None,
                vmax=-1,
                vmin=0,
                cmap="viridis",
                title=False,
                zlabel=False,
            )
        
        # 2) do the tranmission plot

        if tranmission_varname and not tranmission_data:
            raise NotImplementedError
            tranmission_data = ""

        if tranmission_data:

            if area_data:
                self.plotTransmission(
                    data = tranmission_data,
                    ip=ip,
                    title=False,
                    zlabel=False,
                    unit="",
                    loc0="bus_0",
                    loc1="bus_1",
                    crs="epsg:4326",
                    color="k",
                    loc=7,
                    alpha=0.5,
                    ax=ax,
                    fig=fig,
                    linewidth=10,
                    figsize=6,
                    fontsize=12,
                    fileName=None,
                    dpi=300,
                )
            else:
                fig, ax = self.plotTransmission(
                    data = tranmission_data,
                    ip=ip,
                    title=False,
                    zlabel=False,
                    unit="",
                    loc0="bus_0",
                    loc1="bus_1",
                    crs="epsg:4326",
                    color="k",
                    loc=7,
                    alpha=0.5,
                    linewidth=10,
                    figsize=6,
                    fontsize=12,
                    fileName=None,
                    dpi=300,
                )
        else:
            assert fig
            assert ax
        
        #3) plot pies


    def store_default_plots(self):
        '''Store default plots for capacity, operation and FLH.
        '''
        print("Starting default evaluation.")

        componentNamesReduced = self.componentNamesReduced
        
        for comp in componentNamesReduced:
            for ip in list(map(str,InputDataInfo().investment_period_names)):
                #check if tech is clusterd:
                print(comp, ip)
                model = componentNamesReduced[comp]
                
                #plot capacity:
                if model == "TransmissionModel":
                    self.get_var_per_region(
                        variable="capacity",
                        component=comp,
                        ip=ip,
                        agg="sum",
                        plot=True,
                        kwargs_plot={
                            "title": f"Capcity for {comp}",
                            "zlabel": "Capacity [$GW$]",
                        }
                    )
                else:
                    self.get_var_per_region(
                        variable="capacity",
                        component=comp,
                        ip=ip,
                        agg="sum",
                        plot=True,
                        kwargs_plot={
                            "title": f"Capcity for {comp}",
                            "zlabel": "Capacity [$GW$]",
                        }
                    )

                #plot operation:
                if model == "TransmissionModel":
                    print(f"Cannot plot Transmissions: {comp}")
                elif model == "StorageModel":
                    self.get_var_per_region(
                        variable="operationDischarge",
                        component=comp,
                        ip=ip,
                        agg="sum",
                        plot=True,
                        kwargs_plot={
                            "title": f"Operation for {comp}",
                            "zlabel": "Operation [$GWh$]",
                        }
                    )
                
                else:
                    self.get_var_per_region(
                        variable="operation",
                        component=comp,
                        ip=ip,
                        agg="sum",
                        plot=True,
                        kwargs_plot={
                            "title": f"Operation for {comp}",
                            "zlabel": "Operation [$GWh$]",
                        }
                    )


                #plot flh
                if model == "TransmissionModel":
                    print(f"Cannot plot Transmissions: {comp}")
                elif model == "StorageModel":
                    #get vars
                    oper = self.get_var_per_region(
                        variable="operationDischarge",
                        component=comp,
                        ip=ip,
                        agg="sum",
                    )
                    capacity = self.get_var_per_region(
                        variable="capacity",
                        component=comp,
                        ip=ip,
                        agg="sum",
                    )

                    flh = oper / capacity
                    self.plotLocationalColorMap(
                        data=flh,
                        ip=ip,
                        title=f"FLH for {comp}",
                        zlabel="FLH in [$h$]",
                        fileName=f"FLH_{comp}.png",
                    )

                else:
                    #get vars
                    oper = self.get_var_per_region(
                        variable="operation",
                        component=comp,
                        ip=ip,
                        agg="sum",
                    )
                    capacity = self.get_var_per_region(
                        variable="capacity",
                        component=comp,
                        ip=ip,
                        agg="sum",
                    )

                    flh = oper / capacity
                    self.plotLocationalColorMap(
                        data=flh,
                        ip=ip,
                        title=f"FLH for {comp}",
                        zlabel="FLH in [$h$]",
                        fileName=f"FLH_{comp}.png",
                    )
                
                #plot operational colormap
                if model == "TransmissionModel":
                    print(f"Cannot plot Transmissions: {comp}")

                elif model == "StorageModel":
                    try: 
                        data = self.get_var_per_region(
                            variable="stateOfChargeOperationVariablesOptimum",
                            component=comp,
                            ip=ip,
                            agg="sum",
                        )
                        self.plotOperationColorMap(
                            data=data,
                            ip=ip,
                            nbPeriods=ModelTechnoEconomicData().esm_params["esM"]["numberOfTimeSteps"],
                            nbTimeStepsPerPeriod=1,
                            zlabel=f"SOC {comp} [GWh]",
                            fileName=f"Operation_Map_{comp}.png",
                            shading="flat",
                            title=None,
                            dpi=200,
                            pad=0.02,
                            aspect=50,
                            fraction=0.05,
                            orientation="vertical",
                        )
                    except KeyError:
                        print(f"No operational plot for {comp}")
                else:
                    try: 
                        data = self.get_var_per_region(
                            variable="operationVariablesOptimum",
                            component=comp,
                            ip=ip,
                            agg="sum",
                        )
                        self.plotOperationColorMap(
                            data=data,
                            ip=ip,
                            nbPeriods=ModelTechnoEconomicData().esm_params["esM"]["numberOfTimeSteps"],
                            nbTimeStepsPerPeriod=1,
                            zlabel=f"Operation {comp} [GWh]",
                            fileName=f"Operation_Map_{comp}.png",
                            shading="flat",
                            title=None,
                            dpi=200,
                            pad=0.02,
                            aspect=50,
                            fraction=0.05,
                            orientation="vertical",
                        )
                    except:
                        print(f"No operational plot for {comp}")






    def store_standard_evaluation(self):
        '''makes a standardized evaluation of the esm resutls
        '''

        print("Doing default plots.")

        def filter_min_value(ser):
            ser[ser.abs() < self.minValue] = 0
            return ser

        #init containers:
        comp_results = pd.DataFrame()
        comp_results_region = []
        #get component results
        components=self.comp_dict_results
        for compName in components:  
            model=components[compName]
            for ip in list(map(str,InputDataInfo().investment_period_names)): # investmentPeriods müssen in str gewandelt werden
                #result vars
                for var in ["operation", "capacity", "TAC"]:
                    #Transmission model
                    try:
                        if model == "TransmissionModel":
                            #print('Added Transmissionmodel component ' + compName +' to ESM Summary!')
                            ans = self.xr_dss["Results"][ip][model][compName][var].sum("space_2").to_dataframe().T
                    
                        elif model == "StorageModel":

                            if var == "operation":
                                ans = np.fabs(
                                    self.xr_dss["Results"][ip][model][compName]["chargeOperationVariablesOptimum"] \
                                        - self.xr_dss["Results"][ip][model][compName]["dischargeOperationVariablesOptimum"]
                                    ).sum(dim="time").to_dataframe(name="operation").T
                            else:
                                ans = self.xr_dss["Results"][ip][model][compName][var].to_dataframe().T
                    
                        else:
                            ans = self.xr_dss["Results"][ip][model][compName][var].to_dataframe().T
                    except KeyError:
                        #var not found, thats unlucky
                        ans = pd.DataFrame()
                        
                    ans.index = pd.MultiIndex.from_tuples([(compName, ip, var)], names=["component", "investment period" ,"variable"])
                    ans = filter_min_value(ans)
                    comp_results_region.append(ans)

            #input vars
            var = "1d_capacityMax"
            var_short_name = "capacity_max"
            for ip in list(map(str,InputDataInfo().investment_period_names)):
                try:
                    ans = self.get_var_per_region(variable = var,ip=ip, component=compName).to_frame().T
                    ans.index = pd.MultiIndex.from_tuples([(compName,ip, var_short_name)], names=["component","investment period", "variable"])
                except KeyError:
                    ans = pd.DataFrame()
                ans = filter_min_value(ans)
                comp_results_region.append(ans)
            var = "ts_operationRateMax"
            var_short_name = "max_flh"
            for ip in list(map(str,InputDataInfo().investment_period_names)):
                try:
                    ans = self.get_var_per_region(variable = var, ip=ip, component=compName, agg=False)*int(ModelTechnoEconomicData().esm_params["esM"]["hoursPerTimeStep"])
                    ans = ans.groupby("space").sum().T #add up time
                    ans.index = pd.MultiIndex.from_tuples([(compName, ip, var_short_name)], names=["component","investment period", "variable"])
                except KeyError:
                    ans = pd.DataFrame()
                ans = filter_min_value(ans)
                comp_results_region.append(ans)
        comp_results_region = pd.concat(comp_results_region, axis=0)
        comp_results_region.columns.name = "space"
        #unstack the vars
        comp_results_region = comp_results_region.unstack("variable").reorder_levels(["variable", "space"], axis=1)
        #sort them
        comp_results_region = comp_results_region.reindex(sorted(comp_results_region.columns), axis=1)
        comp_results_region.index.name = None
        comp_results_region = comp_results_region.fillna(0)

        #calc abs vars before agg, as rel vars cannot be aggregated!
        #set operation_max_of_capacity_max
        ans = comp_results_region["capacity_max"] * comp_results_region["max_flh"]
        ans.columns = pd.MultiIndex.from_product([["operation_max_of_capacity_max"], list(comp_results_region["capacity_max"].columns)], names=["variable", "space"])
        comp_results_region = pd.concat([comp_results_region, ans], axis=1)
        del ans
        #set operation_max_of_build_capacity
        ans = comp_results_region["capacity"] * comp_results_region["max_flh"]
        ans.columns = pd.MultiIndex.from_product([["operation_max_of_build_capacity"], list(comp_results_region["capacity"].columns)], names=["variable", "space"])
        comp_results_region = pd.concat([comp_results_region, ans], axis=1)
        del ans
        #del max_flh
        comp_results_region = comp_results_region.drop(columns = ["max_flh"])

        #cluster results
        res=self.clustered_techs        
        comp_results_region_sum = pd.DataFrame()
        filtered_wo_res = comp_results_region
        for tech in res:
            # filter of comp_results_region lines containing clustered techs, or regionalized conversions 
            filtered = comp_results_region[comp_results_region.index.get_level_values('component').str.contains(tech)]
            filtered_wo_res = filtered_wo_res[filtered_wo_res.index.get_level_values('component').str.contains(tech)==False]
            
            if not filtered.empty:
                # sum clusters of clustered technologies
                summed = filtered.groupby(level='investment period').sum()
                # Set new index with techname and without cluster (wind_onshore__1 --> wind_onshore)
                summed['component'] = tech
                summed.set_index('component', append=True, inplace=True)
                summed = summed.reorder_levels(['component', 'investment period'])
                # resulting dataframe 
                comp_results_region_sum = pd.concat([comp_results_region_sum, summed])
        # adding not clustered technologies again
        comp_results_region_sum = pd.concat([comp_results_region_sum, filtered_wo_res])
        
        comp_results_region = comp_results_region_sum
        comp_results = comp_results_region.stack("variable").sum(axis=1).unstack("variable")
        df_comp_summary=pd.DataFrame.from_dict(comp_results)

        #additional vars
        df_comp_summary["FLH"] = df_comp_summary["operation"] / df_comp_summary["capacity"]
        df_comp_summary["LCOX_EURct_kWh"] = df_comp_summary["TAC"] / df_comp_summary["operation"] *1E5
        df_comp_summary["rel_capacity_build"] = df_comp_summary["capacity"] / df_comp_summary["capacity_max"]
        df_comp_summary["curtailment"] = 1 - (df_comp_summary["operation"] / df_comp_summary["operation_max_of_build_capacity"])

        #filter nans:
        comp_results[comp_results==np.inf] = np.nan

        #save results
        df_comp_summary.to_csv(os.path.join(self.output_folder, "summary.csv"))
        comp_results_region.to_csv(os.path.join(self.output_folder, "summary_regions.csv"))

        print(f'Saved quick summary to: {self.output_folder}')

        self.comp_results = df_comp_summary
        self.comp_results_regions = comp_results_region

    def store_renewables_evaluation(self):
        # Inputs
        all_sources = [k for k in self.xr_dss["Input"]["Source"]]
        if "Lull" in all_sources: all_sources.remove("Lull")
        results_per_tech = []
        for s in all_sources:
            max_cap = self.xr_dss["Input"]["Source"][s]["1d_capacityMax"].to_series()
            max_cap.name = f"max_cap"
            flh = self.xr_dss["Input"]["Source"][s]["ts_operationRateMax"].sum(dim="time").to_series()
            flh.name = f"max_flh"

            cap = self.xr_dss["Results"]["0"]["SourceSinkModel"][s]["capacityVariablesOptimum"].to_series()
            cap.name = f"cap"

            rel_build = cap/max_cap
            rel_build.name = f"rel_build"

            #lcoe
            tac = self.xr_dss["Results"]["0"]["SourceSinkModel"][s]["TAC"].to_series()
            operation = self.xr_dss["Results"]["0"]["SourceSinkModel"][s]["operationVariablesOptimum"].sum(dim="time").to_series()
            operation.name = "operation"
            lcox = tac/operation * 1E5
            lcox.name = f"lcox"

            vars_per_tech = pd.concat(
                [
                    max_cap,
                    flh,
                    cap,
                    rel_build,
                    operation,
                    lcox,
                ],
                axis=1
            )
            regions =  list(vars_per_tech.index) 
            vars_per_tech.index = pd.MultiIndex.from_product([[s], regions], names= ["tech", "locationID"])
            results_per_tech.append(vars_per_tech)

        renewables_results = pd.concat(results_per_tech)

        #save results
        renewables_results.to_csv(os.path.join(self.output_folder, "renewables_results.csv"))

        print(f'Saved renwable results to: {self.output_folder}')

        self.renewables_results = renewables_results
    
    def get_lull_results(self, lull_component_name='Lull'):
        '''starts the lull analyzer
        '''
        l = LullAnalyzer(
            xr_dss=self.xr_dss,
            lull_component_name=lull_component_name,
            minValue=0.00001,
            clustered_techs=self.clustered_techs,
            comp_dict=self.comp_dict_results,
            outputpath=self.output_folder,
        )
        l.get_lull_evaluation()


class OutputHandler_CSP(OutputHandler):

    def get_csp_vars_agg(self):
        
        print("Starting CSP agg postprocessing.")

        if not hasattr(self, "comp_results_regions"):
            self.store_standard_evaluation()
        

        container = [self.comp_results]

        for htf in ["ss", "he"]:
            
            plant_name = f"csp_{htf}_powerplant"
            storage_name = f"csp_{htf}_storage"
            solarfield_name = f"csp_{htf}_sf"

            solar_field_found = False

            for c in self.comp_results.index:
                if solarfield_name in c:
                    solarfield_name = c
                    solar_field_found = True
                    break
            
            if not solar_field_found:
                #no csp components, add them with zero:
                continue
            
            def filter_min_value(num):
                if np.abs(num) < self.minValue:
                    num = 0
                return num

            tac = filter_min_value(np.float(self.comp_results.loc[[plant_name, storage_name, solarfield_name], "TAC"].sum()))
            capacity = filter_min_value(np.float(self.comp_results.loc[[plant_name], "capacity"]))
            operation = filter_min_value(np.float(self.comp_results.loc[[plant_name], "operation"]))
            capacity_max = filter_min_value(np.float(self.comp_results.loc[[plant_name], "capacity_max"]))
            if capacity_max > 0: capacity_max=np.nan #there is no real limitation in electrical power capacity apart from 0, so drop values. 
            operation_max_of_build_capacity = filter_min_value(np.float(self.comp_results.loc[[solarfield_name], "operation_max_of_build_capacity"]))
            operation_max_of_capacity_max = filter_min_value(np.float(self.comp_results.loc[[solarfield_name], "operation_max_of_capacity_max"]))
            
            rel_capacity_build = filter_min_value(np.float(self.comp_results.loc[[solarfield_name], "rel_capacity_build"]))
            curtailment = filter_min_value(np.float(self.comp_results.loc[[solarfield_name], "curtailment"]))


            
            eta_powerplant = np.float(-1/self.xr_dss["Input"]["Conversion"][plant_name][f"0d_commodityConversionFactors.heat_csp_{htf}"])
            cap_plant_GW_heat = capacity / eta_powerplant
            cap_solarfield_GW_heat = filter_min_value(np.float(self.comp_results.loc[[solarfield_name], "capacity"]))
            size_storage_GWh_heat = filter_min_value(np.float(self.comp_results.loc[[storage_name], "capacity"]))
            

            def zero_devide(a, b):
                if b == 0:
                    return np.nan
                else:
                    return a/b
            #SM and tes

            sm = zero_devide(cap_solarfield_GW_heat,cap_plant_GW_heat)
            tes = zero_devide(size_storage_GWh_heat, cap_plant_GW_heat)

            flh = zero_devide(operation, capacity)
            LCOH_EURct_kWh = zero_devide(tac, operation) * 1E5 #BEUR/GWh = EUR/Wh to EURct/kWh

            output = pd.DataFrame(
                [[tac, capacity, capacity_max, operation, operation_max_of_build_capacity, operation_max_of_capacity_max,flh, LCOH_EURct_kWh, rel_capacity_build, curtailment,sm, tes]],
                columns = ["TAC", "capacity", "capacity_max", "operation", "operation_max_of_build_capacity", "operation_max_of_capacity_max","FLH", "LCOX_EURct_kWh", "rel_capacity_build", "curtailment","SM", "TES"],
                index = [f"csp_{htf}"],
            )

            container.append(output)
        
        self.comp_results = pd.concat(container, axis=0)
        self.comp_results.to_csv
        
        #save results
        self.comp_results.to_csv(os.path.join(self.output_folder, "summary_wcsp.csv"))
        #comp_results_region.to_csv(os.path.join(self.output_folder, "summary_regions.csv"))

        print(f'Saved quick summary to: {self.output_folder}')


    def get_csp_vars_regions(self):
        
        if not hasattr(self, "comp_results_regions"):
            self.store_standard_evaluation()
        

        container = [self.comp_results_regions]

        for htf in ["ss", "he"]:
            
            plant_name = f"csp_{htf}_powerplant"
            storage_name = f"csp_{htf}_storage"
            solarfield_name = f"csp_{htf}_sf"
            
            solar_field_found = False
            
            for c in self.comp_results.index:
                if solarfield_name in c:
                    solarfield_name = c
                    solar_field_found = True
                    break
            
            if not solar_field_found:
                #no csp components, add them with zero:
                continue

            def filter_min_value(ser):
                ser[ser.abs() < self.minValue] = 0
                return ser


            tac = filter_min_value(self.comp_results_regions.loc[[plant_name, storage_name, solarfield_name], "TAC"].sum(axis=0).to_frame("tac").T)
            capacity = filter_min_value(self.comp_results_regions.loc[[plant_name], "capacity"])
            operation = filter_min_value(self.comp_results_regions.loc[[plant_name], "operation"])

            
            eta_powerplant = np.float(-1/self.xr_dss["Input"]["Conversion"][plant_name][f"0d_commodityConversionFactors.heat_csp_{htf}"])
            cap_plant_GW_heat = capacity / eta_powerplant
            cap_solarfield_GW_heat = filter_min_value(self.comp_results_regions.loc[[solarfield_name], "capacity"])
            size_storage_GWh_heat = filter_min_value(self.comp_results_regions.loc[[storage_name], "capacity"])
            
            sm = cap_solarfield_GW_heat.reset_index(drop=True) / cap_plant_GW_heat.reset_index(drop=True)
            sm.columns = pd.MultiIndex.from_product([["SM"], list(tac.columns)])
            sm.index = [f"csp_{htf}"]

            tes = size_storage_GWh_heat.reset_index(drop=True) / cap_plant_GW_heat.reset_index(drop=True)
            tes.columns = pd.MultiIndex.from_product([["TES"], list(tac.columns)])
            tes.index = [f"csp_{htf}"]

            flh = operation.reset_index(drop=True) / capacity.reset_index(drop=True)
            flh.columns = pd.MultiIndex.from_product([["FLH"], list(tac.columns)])
            flh.index = [f"csp_{htf}"]

            LCOH_EURct_kWh = tac.reset_index(drop=True) / operation.reset_index(drop=True) * 1E5 #BEUR/GWh = EUR/Wh to EURct/kWh
            LCOH_EURct_kWh.columns = pd.MultiIndex.from_product([["LCOX_EURct_kWh"], list(tac.columns)])
            LCOH_EURct_kWh.index = [f"csp_{htf}"]

            capacity.columns = pd.MultiIndex.from_product([["capacity"], list(tac.columns)])
            capacity.index = [f"csp_{htf}"]
            operation.columns = pd.MultiIndex.from_product([["operation"], list(tac.columns)])
            operation.index = [f"csp_{htf}"]
            tac.columns = pd.MultiIndex.from_product([["TAC"], list(tac.columns)])
            tac.index = [f"csp_{htf}"]
            

            output = pd.concat(
                [tac, capacity, operation, flh, LCOH_EURct_kWh, sm, tes],
                axis=1
            )

            container.append(output)
        
        self.comp_results = pd.concat(container, axis=0)
        self.comp_results.to_csv
        
        #save results
        self.comp_results.to_csv(os.path.join(self.output_folder, "summary_regions_wcsp.csv"))
        #comp_results_region.to_csv(os.path.join(self.output_folder, "summary_regions.csv"))

        print(f'Saved quick summary to: {self.output_folder}')

class LullAnalyzer():

    def __init__(self, xr_dss, lull_component_name, minValue, clustered_techs, comp_dict, outputpath):
        self.xr_dss = xr_dss
        self.lull_component_name = lull_component_name
        self.minValue = minValue
        self.clustered_techs = clustered_techs
        self.comp_dict = comp_dict

        self.outputpath_lull = os.path.join(outputpath, "lull_results")
        os.makedirs(self.outputpath_lull, exist_ok=True)

        #get the raw data:
        self.rawLullAnalysis(lull_component_name=self.lull_component_name) #TODO: mabe save intermeds?

    def get_lull_evaluation(self):

        #get the data
        self.getLullCharacteristics(
            tMin=0,
            tMax=8760,
            regionalLevel=2,
            region=None,
        )

        if self.df_lullCharacteristics.shape[0] == 0:
            print("No lulls found.")
            return
        
        #plotting distribution
        self.scatterPlotLullAttributes(
            char1="Dauer",
            char2="Defizit",
            kind="reg",
            hue="Land",
            scatter=False,
        )

        #plotting PowerFlows
        #plot max duration:
        max_duration = self.df_lullCharacteristics.loc[self.df_lullCharacteristics["Dauer [Stunden]"].argmax()]
        self.plotPowerFlow(
            tMin=max(max_duration.Start - 2, 0),
            tMax=min(max_duration.Ende + 2, 8760),
            commodity='electricity',
            regionalLevel=1,
            region=max_duration.Region,
            fname="powerflowDiagramMaxDuration.png",
        )

        #plot max Defizit:
        max_deficit = self.df_lullCharacteristics.loc[self.df_lullCharacteristics["Defizit [GWh]"].argmax()]
        self.plotPowerFlow(
            tMin=max(max_deficit.Start - 2, 0),
            tMax=min(max_deficit.Ende + 2, 8760),
            commodity='electricity',
            regionalLevel=1,
            region=max_deficit.Region,
            fname="powerflowDiagramMaxDefizit.png",
        )

        #plot max Load:
        max_load = self.df_lullCharacteristics.loc[self.df_lullCharacteristics["Max. Fehlleistung [GW]"].argmax()]
        self.plotPowerFlow(
            tMin=max(max_load.Start - 2, 0),
            tMax=min(max_load.Ende +2, 8760),
            commodity='electricity',
            regionalLevel=1,
            region=max_load.Region,
            fname="powerflowDiagramMaxLoad.png",
        )

        #all_region_plot:
        self.plotPowerFlow(
            tMin=max(max_deficit.Start - 2, 0),
            tMax=min(max_deficit.Ende + 2, 8760),
            commodity='electricity',
            regionalLevel=2,
            region=max_deficit.Region,
            fname="powerflowDiagramMaxDefizitAllReg.png",
        )

    ## Helper for plotPowerFlow
    def getAutomatedRegionalComponentTimeSeries(
        self,
        tMin=0,
        tMax=8760,
        commodity='electricity',
        sum_clusters=True,
        regionalLevel=1,
        region=None,
    ):
        '''_summary_

        Parameters
        ----------
        tMin : int, optional
            _description_, by default 0
        tMax : int, optional
            _description_, by default 8760
        commodity : str, optional
            _description_, by default 'electricity'
        sum_clusters : bool, optional
            _description_, by default True
        regionalLevel : int, optional
            _description_, by default 1
        region : _type_, optional
            _description_, by default None

        Returns
        -------
        pd.DataFrame
            time series of technologies between tmin and tmax: index: tech, columns, time step
        '''

        XX = np.s_[tMin:tMax]
                
        components = self.comp_dict
        flowComponents = self.comp_dict.copy()
        
        D = dict()

        if regionalLevel == 1:
            for c in self.comp_dict: 
                if self.comp_dict[c] == "TransmissionModel":
                    if self.xr_dss["Input"]["Transmission"][c]["0d_commodity"].values == commodity: 
                        data = self.xr_dss["Results"]["0"][self.comp_dict[c]][c]["operationVariablesOptimum"]

                        ts_export = data.sel(space=region).sum(dim="space_2").to_dataframe()
                        ts_export = ts_export["operationVariablesOptimum"]

                        #ts_import = data.sel(space_2 = region).sum(dim="space").to_dataframe()
                        ts_import_wo_losses = data.sel(space_2=region)
                        rel_losses = (self.xr_dss["Input"]["Transmission"][c]["2d_losses"] * self.xr_dss["Input"]["Transmission"][c]["2d_distances"]).sel(space_2=region)
                        ts_import = ((1-rel_losses) * ts_import_wo_losses).sum(dim="space").to_dataframe(name="operationVariablesOptimum")
                        ts_import = ts_import["operationVariablesOptimum"]

                        ts_net = ts_import - ts_export

                        ts_export = ts_net.copy()
                        ts_export[ts_export>0] = 0

                        ts_import = ts_net.copy()
                        ts_import[ts_import<0] = 0

                        D["Net Import"] = ts_import
                        D["Net Export"] = ts_export
                     
                if self.comp_dict[c] == "SourceSinkModel":
                    if c in self.xr_dss["Input"]["Source"]:
                        if self.xr_dss["Input"]["Source"][c]["0d_commodity"].values == commodity:
                            try:
                                data = self.xr_dss["Results"]["0"][self.comp_dict[c]][c]["operationVariablesOptimum"].sel(space=region).to_dataframe()
                                data = data["operationVariablesOptimum"]

                                D[c] = data#.where(data >= self.minValue, 0.0)
                            except:
                                print('No '+ c +' in '+ region)
                
                elif self.comp_dict[c] == "StorageModel":
                    if self.xr_dss["Input"]["Storage"][c]["0d_commodity"] == commodity:
                        try:
                            data_chrg = self.xr_dss["Results"]["0"][self.comp_dict[c]][c]["chargeOperationVariablesOptimum"].sel(space=region).to_dataframe()
                            data_chrg = data_chrg["chargeOperationVariablesOptimum"]

                            data_dchrg = self.xr_dss["Results"]["0"][self.comp_dict[c]][c]["dischargeOperationVariablesOptimum"].sel(space=region).to_dataframe()
                            data_dchrg = data_dchrg["dischargeOperationVariablesOptimum"]

                            ts_net = data_dchrg - data_chrg

                            ts_dcharge = ts_net.copy()
                            ts_dcharge[ts_dcharge>0] = 0

                            ts_charge = ts_net.copy()
                            ts_charge[ts_charge<0] = 0

                            D[f"{c}_charge"] = ts_dcharge
                            D[f"{c}_discharge"] = ts_charge
                        except:
                            print('No '+ c +' charge in '+ region)
                
                elif self.comp_dict[c] == "ConversionModel":
                    if f"0d_commodityConversionFactors.{commodity}" in self.xr_dss["Input"]["Conversion"][c]:
                        try:
                            data = self.xr_dss["Results"]["0"][self.comp_dict[c]][c]["operationVariablesOptimum"].sel(space=region).to_dataframe()
                            data = data["operationVariablesOptimum"]

                            commodity_conversion_factor = self.xr_dss["Input"]["Conversion"][c][f"0d_commodityConversionFactors.{commodity}"]

                            if commodity_conversion_factor > 0.0:
                                D[c] = data.where(data >= 0, 0)
                            else:
                                data = data * -1
                                D[c] = data.where(data <= 0, 0)
                        except:
                            print('No '+ c +' in '+ region)
            

        if regionalLevel == 2:
            for c in self.comp_dict:  
                if self.comp_dict[c] == "SourceSinkModel":
                    if c in self.xr_dss["Input"]["Source"]:
                        if self.xr_dss["Input"]["Source"][c]["0d_commodity"].values == commodity:
                            try:
                                data = self.xr_dss["Results"]["0"][self.comp_dict[c]][c]["operationVariablesOptimum"].sum("space").to_dataframe()
                                data = data["operationVariablesOptimum"]
                                D[c] = data # filter out negative values
                            except:
                                print('No '+ c)
                
                elif self.comp_dict[c] == "StorageModel":
                    if self.xr_dss["Input"]["Storage"][c]["0d_commodity"] == commodity:
                        
                        try:                   
                            data_chrg = self.xr_dss["Results"]["0"][self.comp_dict[c]][c]["chargeOperationVariablesOptimum"].sum(dim="space").to_dataframe()
                            data_chrg = data_chrg["chargeOperationVariablesOptimum"]

                            data_dchrg = self.xr_dss["Results"]["0"][self.comp_dict[c]][c]["dischargeOperationVariablesOptimum"].sum(dim="space").to_dataframe()
                            data_dchrg = data_dchrg["dischargeOperationVariablesOptimum"]

                            ts_net = data_dchrg - data_chrg

                            ts_dcharge = ts_net.copy()
                            ts_dcharge[ts_dcharge>0] = 0

                            ts_charge = ts_net.copy()
                            ts_charge[ts_charge<0] = 0

                            D[f"{c}_charge"] = ts_dcharge
                            D[f"{c}_discharge"] = ts_charge
                        except:
                            print('No '+ c )

                elif self.comp_dict[c] == "ConversionModel":
                    if f"0d_commodityConversionFactors.{commodity}" in self.xr_dss["Input"]["Conversion"][c]:

                        try:
                            data = self.xr_dss["Results"]["0"][self.comp_dict[c]][c]["operationVariablesOptimum"].sum(dim="space").to_dataframe()
                            data = data["operationVariablesOptimum"]

                            commodity_conversion_factor = self.xr_dss["Input"]["Conversion"][c][f"0d_commodityConversionFactors.{commodity}"]

                            if commodity_conversion_factor > 0.0:
                                D[c] = data.where(data >= 0, 0)
                            else:
                                data = data * -1
                                D[c] = data.where(data <= 0, 0)
                        except:
                            print('No '+ c )
        
        D = pd.DataFrame.from_dict(D)
        D = D.iloc[XX].copy()
        D =  D.transpose()

    
        if sum_clusters:
            res=self.clustered_techs
            for r in res:
                # Sum potential Data RES clusters 
                tech=D[D.index.str.contains(r)]
                D=D[D.index.str.contains(r)==False]
                tech=tech.sum(axis=0)
                D.loc[r]=tech

        return D


    def getRegionalDemandTimeSeries(self, tMin=805, tMax=820, commodity='electricity', regionalLevel=2, region="DEU.1_1"):
        '''_summary_

        Parameters
        ----------
        tMin : int, optional
            _description_, by default 805
        tMax : int, optional
            _description_, by default 820
        commodity : str, optional
            _description_, by default 'electricity'
        regionalLevel : int, optional
            _description_, by default 2
        region : str, optional
            _description_, by default "DEU.1_1"

        Returns
        -------
        pd.DataFrame:
            demand time series as Dataframe between tmin and tmax
        '''

        XX = np.s_[tMin:tMax]
        demand = dict()
        components = self.comp_dict
        
        if regionalLevel == 1:
            for c in components:  
                if c in self.xr_dss["Input"]["Sink"]:
                    if self.xr_dss["Input"]["Sink"][c]["0d_commodity"] == commodity:
                        data = self.xr_dss["Input"]["Sink"][c]["ts_operationRateFix"].sel(space=region).to_dataframe()
                        data = data["ts_operationRateFix"]

                        demand[c] = data

        if regionalLevel == 2:
            for c in components:  
                if c in self.xr_dss["Input"]["Sink"]:
                    if self.xr_dss["Input"]["Sink"][c]["0d_commodity"] == commodity:
                        data = self.xr_dss["Input"]["Sink"][c]["ts_operationRateFix"].sum(dim="space").to_dataframe()
                        data = data["ts_operationRateFix"]

                        demand[c] = data

        demand = pd.DataFrame.from_dict(demand)
        demand = demand.iloc[XX].copy()
        demand =  demand.transpose()

        return demand

    ## Data pipeline
    def getTimeseries(self, lull_component_name, variableName="operationVariablesOptimum", locTrans=None):
        '''returns raw time series from ESM for all regions and time steps for the given compName

        Parameters
        ----------
        compName : _type_
            _description_
        variableName : str, optional
            _description_, by default "operationVariablesOptimum"
        locTrans : _type_, optional
            _description_, by default None

        Returns
        -------
        pd.DataFrame
            raw time series, X: region, y: time step
        '''
        
        #check for lull components:
        lull_components = [k for k in self.comp_dict if lull_component_name in k]
        timeSeries = None
        for lull_component in lull_components:
            if self.xr_dss["Results"]["0"][self.comp_dict[lull_component]][lull_component]["operation"].sum() != 0:
                timeSeries_i = self.xr_dss["Results"]["0"][self.comp_dict[lull_component]][lull_component][variableName].to_dataframe().unstack("time")[variableName]
            else:
                #no values, so get a df of zeroes
                numberOfTimeSteps = self.xr_dss["Parameters"].numberOfTimeSteps
                regions = self.xr_dss["Parameters"].locations
                values = np.zeros((len(regions), numberOfTimeSteps))
                timeSeries_i = pd.DataFrame(
                    data=values,
                    index=regions,
                    columns=list(range(numberOfTimeSteps))
                )
                assert timeSeries_i.sum().sum() == 0
            
            if timeSeries is None:
                timeSeries = timeSeries_i
            else:
                timeSeries = timeSeries + timeSeries_i

        if not locTrans is None:
            timeSeries = timeSeries.loc[locTrans]
        timeSeries=timeSeries.where(timeSeries >= self.minValue, 0.0)
        return timeSeries

    def rawLullAnalysis(self, lull_component_name):
        '''returns time sereis with information about the current lull

        Parameters
        ----------
        timeSeries : pd.DataFrame
            lull time series

        Returns
        -------
        dict
            dict of timeseries with info about Deficit, Duration and maxResLoad
        '''
        
        rawLullAnalysis = self._load_rawLullAnalysis()

        if rawLullAnalysis == {}:

            print('Start raw lull analysis.')
            time_series_lull = self.getTimeseries(lull_component_name=lull_component_name)
            # raw lull analysis to iterate trough time_series_lull to attribute every time step with a deficit, duration and max load lost value
            zeroedTimeSeries = time_series_lull.copy()

            for col in zeroedTimeSeries.columns:
                zeroedTimeSeries[col].values[:] = 0.0

            # Create zeroed Dataframes to track lull characteristics
            lullDeficit = zeroedTimeSeries.copy()
            lullDuration = zeroedTimeSeries.copy()
            maxResLoad = zeroedTimeSeries.copy()

            # Iterate through time series to count lulls and their characteristics
            for ix in lullDeficit.index:
                print('For: '+str(ix))
                deficit = 0
                duration = 0
                thisMaxResLoad = 0
                for t in lullDeficit.columns:
                    
                    if np.isclose(time_series_lull.loc[ix, t], 0.0):
                        if t > 0.0 and t < 8757 and time_series_lull.loc[ix, t-1] != 0.0 and time_series_lull.loc[ix, t+1] != 0.0 and time_series_lull.loc[ix, t+2] != 0.0:   
                            duration += 1
                            deficit += time_series_lull.loc[ix, t]
                            lullDeficit.loc[ix, t] = deficit
                            lullDuration.loc[ix, t] = duration
                        if t > 0.0 and t < 8758 and time_series_lull.loc[ix, t-1] != 0.0 and time_series_lull.loc[ix, t+1] != 0.0:   
                            duration += 1
                            deficit += time_series_lull.loc[ix, t]
                            lullDeficit.loc[ix, t] = deficit
                            lullDuration.loc[ix, t] = duration
                        else:
                            duration = 0.0
                            deficit = 0.0
                            thisMaxResLoad = 0.0
                    else:
                        duration += 1
                        deficit += time_series_lull.loc[ix, t]
                        lullDeficit.loc[ix, t] = deficit
                        lullDuration.loc[ix, t] = duration
                        if time_series_lull.loc[ix, t] > thisMaxResLoad:
                            thisMaxResLoad = time_series_lull.loc[ix, t]
                        maxResLoad.loc[ix, t] = thisMaxResLoad


            rawLullAnalysis = dict(Deficit=lullDeficit, Duration=lullDuration, MaxResLoad=maxResLoad)
            print('Got raw lull analysis.')
            self._save_rawLullAnalysis(rawLullAnalysis)

        self.characteristics = {"Dauer": 0,"Defizit": 1,"Max. Fehlleistung": 2}
        self.rawLullAnalysisDict = rawLullAnalysis

    def _save_rawLullAnalysis(self, rawLullAnalysis):    
        for key in rawLullAnalysis:
            save_path = os.path.join(
                self.outputpath_lull,
                f"raw_lull_analysis_{key}.csv"
            )
            rawLullAnalysis[key].to_csv(save_path)
        print("rawLullAnalysis saved to:", self.outputpath_lull)
    
    def _load_rawLullAnalysis(self):
        
        load_path_glob = os.path.join(
            self.outputpath_lull,
            "raw_lull_analysis_*.csv"
        )
        load_paths = glob.glob(load_path_glob)

        #load files
        rawLullAnalysis = {}
        for load_path in load_paths:
            temp = pd.read_csv(
                load_path,
                index_col=[0],
                header=[0]
            )
            temp.columns.name="time"
            temp.columns = [int(i) for i in temp.columns]
            #get rawLullAnalysis-name
            file_name = os.path.basename(load_path)
            rawLullAnalysis_name = file_name.replace("raw_lull_analysis_", "").replace(".csv", "")
            rawLullAnalysis[rawLullAnalysis_name] = temp
        
        return rawLullAnalysis
        

    def getLullCharacteristicTimeSeries(self, tMin=0, tMax=8760, regionalLevel=1, region='DEU.1_1', char='Deficit'):
        '''returns selection from self.rawLullAnalysisDict

        Parameters
        ----------
        tMin : int, optional
            _description_, by default 0
        tMax : int, optional
            _description_, by default 8760
        regionalLevel : int, optional
            1: all regions
            2: specific region
        region : str, optional
            _description_, by default 'DEU.1_1'
        char : str, optional
            _description_, by default 'Deficit'

        Returns
        -------
        pd.DataFrame
            df with char over. X: region Y: time
        '''
        rawLullAnalysisDict = self.rawLullAnalysisDict

        XX = np.s_[tMin:tMax]
        print("Getting lull " + char + " time series.")

        lullChar = pd.DataFrame(rawLullAnalysisDict[char])


        if regionalLevel == 1:
            lullChar = lullChar.loc[(region)]
            lullChar = lullChar.transpose()
            lullChar = lullChar.iloc[XX].copy()
        
        if regionalLevel == 2:
            lullChar = lullChar.transpose()
            lullChar = lullChar.iloc[XX].copy()

        lullChar=lullChar.where(lullChar >= self.minValue, 0.0)
        
        return lullChar

    def getLullTimes(self, tMin=0, tMax=8760, regionalLevel=1, region = "DEU.1_1"):
        '''_summary_

        Parameters
        ----------
        tMin : int, optional
            _description_, by default 0
        tMax : int, optional
            _description_, by default 8760
        regionalLevel : int, optional
            _description_, by default 1
        region : str, optional
            _description_, by default "DEU.1_1"

        Returns
        -------
        list of dicts
            [(region, lull start, lull end), (region, lull start, lull end), ...]
        '''
        
        XX = np.s_[tMin:tMax]
        lullDuration = self.getLullCharacteristicTimeSeries(tMin=tMin, tMax=tMax, char="Duration", regionalLevel=regionalLevel, region=region)
         # Prepare lull duration data
        lullDuration = lullDuration.transpose()
        lullDuration = lullDuration.iloc[XX].copy()
        #print(lullDuration)
        print("Use lull Duration time series to get lull start and end times.")
        
        # Get regions for lull analysis
        regions = []
        if regionalLevel == 1:
            last = 0
            pairs = []
            
            for i in range(tMin,tMax):
                if lullDuration.loc[i] > 0 and last == 0:
                    start = i
                elif lullDuration.loc[i]==0 and last > 0:
                    pairs.append((region, start, i))
                last = lullDuration.loc[i]
        if regionalLevel == 2:
            regions = list(self.rawLullAnalysisDict["Duration"].index)
            
            pairs = []
            lullDuration = pd.DataFrame(lullDuration)
            for reg in regions:
                last = 0
                start=None
                for i in range(tMin,tMax):
                    #print(lullDuration.loc[reg, i])
                    if i == 8759:
                        if lullDuration.loc[reg,i] > 0 and last == 0:
                            start = i
                            pairs.append((reg, start, i+1))
                            #print('Fall 1 eingetreten!')
                        elif lullDuration.loc[reg,i] > 0 and last > 0:
                            pairs.append((reg, start, i+1))
                            #print('Fall 2 eingetreten!')
                        else:
                            pass
                            #print('Fall 3 eingetreten!')
                    elif lullDuration.loc[reg,i] > 0 and last == 0:
                        start = i
                    elif lullDuration.loc[reg,i]==0 and last > 0:
                        pairs.append((reg, start, i))
                    last = lullDuration.loc[reg,i]
             
        return pairs

    def getLullCharacteristics(self, tMin = 0, tMax = 8760, regionalLevel=1, region= "DEU.1_1"):
        '''get the summed lull characteristics

        Parameters
        ----------
        tMin : int, optional
            _description_, by default 0
        tMax : int, optional
            _description_, by default 8760
        regionalLevel : int, optional
            _description_, by default 1
        region : str, optional
            _description_, by default "DEU.1_1"

        Returns
        -------
        dict:
            (duration, deficits, maxload, gid1, gid0, start, end)
        '''
        pairs = self.getLullTimes(
            tMin=tMin,
            tMax=tMax,
            regionalLevel=regionalLevel,
            region=region)
        if pairs: 
            print('Lull times found!')
        else:
            print('No lull times found!')
        lullDeficit = self.getLullCharacteristicTimeSeries(tMin=tMin, tMax=tMax, char="Deficit", regionalLevel=regionalLevel, region=region)
        maxloadlull = self.getLullCharacteristicTimeSeries(tMin=tMin, tMax=tMax, char="MaxResLoad", regionalLevel=regionalLevel, region=region)
        print("Get all lulls and their charcteristics.")
        # Retrieving characteristic tuples where st is the first day of loss of load, ed the first day without. ed-st gived the number of days where loss of load is > 0. 
        # Thus, final lull characteristics have to found at loc ed-1 in the respective time series.
        
        lullCharacteristics = []
        if regionalLevel == 1:
            for reg,st,ed in pairs:
                lullCharacteristics.append((ed-st , lullDeficit.loc[ed-1], maxloadlull.loc[ed-1], reg, reg[0:3], st, ed))

        if regionalLevel == 2:
            for reg,st,ed in pairs:
                try:
                    lullCharacteristics.append((ed-st , lullDeficit[reg][ed-1], maxloadlull[reg][ed-1], reg, reg[0:3], st, ed))
                except:
                    print('Error in ' + reg + ' with lull starting in '+ str(st)+' and ending in '+ str(ed))
    
        print("Got all lulls and their charcteristics.")

        df_lullCharacteristics = pd.DataFrame(
            data = lullCharacteristics,
            columns = ["duration", "deficit", "maxload", "gid1", "gid0", "start", "end"]
        )
        df_lullCharacteristics.to_csv(os.path.join(self.outputpath_lull, "lull_characteristics.csv"))

        self.df_lullCharacteristics = df_lullCharacteristics
    
    ## Plots
    def plotPowerFlow(self, tMin=805, tMax=820, commodity='electricity', regionalLevel=2, region="DEU.1_1", fname=None):
        
        plt.figure(figsize=(8,14))
        gs = GridSpec(2,1, hspace=0.05, height_ratios=[2,4])
        plt.rc('font', size=16)

        ax = plt.subplot(gs[1])
        XX = np.s_[tMin:tMax]

        ### Stackplots and their for plot of power flow as first subplot

        # Import Time Series
        D = self.getAutomatedRegionalComponentTimeSeries(
            tMin=tMin,
            tMax=tMax,
            regionalLevel=regionalLevel,
            region=region,
            commodity=commodity,
            sum_clusters=True
        )
        #print(D)
        #Delete all zero components, so that they do not appear in negative and positive stackplot legend
        D=D.loc[~(D==0).all(axis=1)]

        # Splitting components by their sign in contribution to the energy balance
        posFlowComponents = D[(D[D.columns] >= 0).all(axis=1)].index.values.tolist()
        negFlowComponents = D[(D[D.columns] <= 0).all(axis=1)].index.values.tolist()
        #print(negFlowComponents)
        flowComponents = D.index.tolist()

        # Get separate demand time Series
        demand = self.getRegionalDemandTimeSeries(tMin=tMin,tMax=tMax,commodity='electricity', regionalLevel=regionalLevel, region=region)

        # Prepare colormap fitting to the data
        cmap = plt.get_cmap('tab20', len(flowComponents))    # tab20, PiYG
        colormap=[]
        for i in range(cmap.N):
            rgba = cmap(i)
            colormap.append(col.rgb2hex(rgba))
        
        # Avoid similar colors in the two stackplots for positive and negative inputs and outputs of the energy balance
        posColorMapEntries = colormap [0:len(posFlowComponents)]
        negColorMapEntries = colormap[len(posFlowComponents):]

        # Subplot for demand and power flow
        plt.stackplot(D.columns, D[(D[D.columns] >= 0.00000000).all(axis=1)],  zorder=2, colors= posColorMapEntries, labels=posFlowComponents,linewidth=0)#colors = colormap
        if not len(negFlowComponents) == 0:
            plt.stackplot(D.columns, D[(D[D.columns] <= 0.00000000).all(axis=1)], zorder=2, colors= negColorMapEntries, labels=negFlowComponents,linewidth=0)
        plt.plot(demand.columns, demand.iloc[0], 'k', linewidth=2, label="Demand") # iloc ist hardgecoded, bei mehreren Sinks Iteration oder Auswahl notwendig
        
        dlull = self.getLullCharacteristicTimeSeries(
            tMin=tMin,
            tMax=tMax,
            char= "Deficit",
            regionalLevel=2,
        )

        plt.ylabel("GW", )
        plt.xlabel("Zeit\n[h]", )
    
        plt.xlim(D.columns[0], D.columns[-1])
        
        plt.grid(alpha=0.5, zorder=0)

        plt.legend(fontsize=16, loc="upper left", ncol=2, bbox_to_anchor=(-0.02,2.05), columnspacing=1.0, handlelength=1.5)

        
        # Subplot for deficiency
        ax = plt.subplot(gs[0])
        
        plt.plot(D.columns, dlull[region], "--k", linewidth=2)

        plt.ylabel("Defizit\n[GWh]", )

        plt.xlim(D.columns[0], D.columns[-1])
        plt.xticks(D.columns[30::24*2], [], rotation=0, ha="center")
        plt.grid(alpha=0.5)

        if not fname:
            fname = os.path.join(self.outputpath_lull, "powerflowDiagram.png")
        else:
            fname = os.path.join(self.outputpath_lull, fname)

        plt.savefig(fname, dpi=200, bbox_inches='tight')
    
        plt.close()

    def scatterPlotLullAttributes(self, char1="Dauer", char2="Defizit", kind="reg", hue="Land", scatter=False, fname=None):

        # Get lull data
        #lullCharacteristics = self.getLullCharacteristics(tMin=tMin, tMax=tMax)
        sns.set_theme(style="darkgrid")
        data = self.df_lullCharacteristics#pd.DataFrame(lullCharacteristics)

        # Represent lull characteristics by int value
        ichar1 = self.characteristics[char1]
        ichar2 = self.characteristics[char2]

        data.rename(columns={"duration": 'Dauer [Stunden]', "deficit": 'Defizit [GWh]', "maxload": 'Max. Fehlleistung [GW]', "gid1": 'Region', "gid0": 'Land', "start": 'Start', "end": 'Ende'}, inplace=True)
        labels = list(data)

        # Find maximum values for lull characteristics to set xlim and ylim
        max=[]
        try:
            max.append(data['Dauer [Stunden]'].values.max())
        except:
            print('No max duration value found to define axis limit')
        try:
            max.append(data['Defizit [GWh]'].values.max())
        except:
            print('No max deficit value found to define axis limit')
        try:
            max.append(data['Max. Fehlleistung [GW]'].values.max())
        except:
            print('No max Lost Load value found to define axis limit')
        #print(max)
        if scatter:
            g = sns.jointplot(x=labels[ichar1], y=labels[ichar2], data=data,
                            kind=kind, 
                            xlim=(0, max[ichar1]+5), ylim=(0, max[ichar2]+10),
                            color='#023D6B', height=7, hue=hue)
        else:
            g = sns.jointplot(x=labels[ichar1], y=labels[ichar2], data=data,
                            kind=kind, truncate=False,
                            xlim=(0, max[ichar1]+5), ylim=(0, max[ichar2]+10),
                            color='#023D6B', height=7)
        plt.suptitle('Kombinierte Flauteneigenschaften \n ' +  char1 + ' und ' + char2 + ' \n mit jeweiliger, \n univariater Dichtefunktion.')
        plt.tight_layout()
        if not fname:
            fname = os.path.join(self.outputpath_lull, "lull_occurance_deficit_over_time.png")
        g.figure.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    
    
#%%
