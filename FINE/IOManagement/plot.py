import numpy as np
import geokit as gk
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt

colors = ['yellowGreen','#4d6619', 'gold','darkOrange','purple', 'skyBlue','steelBlue','#2A4E6C','sienna']


def piechart_plot_function(shapes, pieDataframe, transmission_dataframe=None, pieColors=None, piechart_locations=None, ax=None, plot_settings=None, srs=gk.srs.EPSG3035):
    """    Plot the data of a component for each location.

    **Required arguments:**

    :param esM: considered energy system model
    :type esM: EnergySystemModel class instance

    :param shapes: Dataframe of the regions. It can be obtained by gk.vector.extractFeatures...
        The column name of geometry objects in the dataframe has to be 'geom'
        *Region names have to be the index
    :type shapes: pd.DataFrame

    :param pieDataframe:
        Dataframe of the attributes to be plotted in each region as piechart.
        *Region names have to be the index, columns have to be individual attributes
    :type pieDataframe: pd.DataFrame

    :param transmission_dataframe: 
        Dataframe of the transmission connections with attribute to be represented with lineWidth
        *The name of attribute column has to be weight
        *It has the geometries of connections between region centroids, the column name has to be 'geom'

    :type transmission_dataframe: pd.DataFrame

    :param pieColor: A list of colors to be used
    :type pieColor: list; optional

    :param piechart_locations: 
        A dictionary of geometry objects for centroids at which the piechart will be plotted
    :type piechart_locations: dictionary; optional

    :param ax: A matplotlib axis to plot the shapes on
    :type ax: matplotlib axis; optional

    :param plotSettings: A dictionary involving arguments for plottings.
    :type plotSettings: dictionary; optional

    Returns:
    --------
    The map figure and axis
"""
    
    if plot_settings is None: plot_settings={}
    if 'transmissionLineColor' not in plot_settings.keys(): plot_settings['transmissionLineColor']='#767676'
    if 'markerScaling' not in plot_settings.keys(): plot_settings['markerScaling']=1/1.5
    if 'lineWidthScaling' not in plot_settings.keys(): plot_settings['lineWidthScaling']=1/1.5
    if 'bbox_to_anchor' not in plot_settings.keys(): plot_settings['bbox_to_anchor']=(0.4,.93)
    if 'fontSize' not in plot_settings.keys(): plot_settings['fontSize']=16

    if ax is None: fig, ax = plt.subplots()
    if pieColors is None: pieColors = colors

    shapes.sort_index(inplace=True)
    pieDataframe.sort_index(inplace=True)

    if not list(shapes.index) == list(pieDataframe.index): print('WHATTHEFUCK!!!')

    if piechart_locations is None:
        piechart_locations={}
        for ix in shapes.index: 
            piechart_locations[ix]=shapes.loc[ix].geom.Centroid()


    def drawWedge(start, stop):
        x = [0] + np.cos(np.arange(2 * np.pi * start, 2 * np.pi * stop, 0.1)).tolist()
        y = [0] + np.sin(np.arange(2 * np.pi * start, 2 * np.pi * stop, 0.1)).tolist()
        return np.column_stack([x, y])
    

    def plotPieGraph(ax, x, y, shares, colors, markerScaling, **k ):
        total = sum(shares)
        start = 0
        for share, cl  in zip(shares, colors):
            if total==0: continue
            stop = start+ share/total
            ax.plot(x, y, marker=(drawWedge(start, stop), 0), ms=np.power(total,markerScaling), markerfacecolor=cl , **k)
            start = stop

    totalGenerationCap = {}
    for i,row_ in pieDataframe.iterrows():
        totalGenerationCap[i]= np.array(pieDataframe.loc[i])
    
    gk.drawGeoms(shapes, srs=srs, hideAxis=True, fc='#F5F5F5', ax=ax, ec='#D3D3D3')
    ax.set_aspect('equal')
    
    for i,v in zip(piechart_locations.values(), totalGenerationCap.values()):
        plotPieGraph(ax, i.GetX(), i.GetY(), v,colors=pieColors,zorder=4, markeredgecolor=(.1,.1,.1), lw=0.5, markerScaling=plot_settings['markerScaling'])
    
    ax.axis("off")
    
    handles = []
    labelHandles = []
    for i, pieValueDesciption in enumerate(pieDataframe.columns): 
        pieColor = pieColors[i%len(colors)]   # TODO:add proper color cycling!!
        handles.append(Patch(color=pieColor, label=pieValueDesciption))
        labelHandles.append(pieValueDesciption)

    pieValues = pieDataframe.sum(axis=1).values
    pieMarkerHandles = [Line2D((),(), marker='o', ms=np.power(np.percentile(pieValues, 5), plot_settings['markerScaling']), linestyle='None', color='gray'), 
           Line2D((),(), marker='o', ms=np.power(np.percentile(pieValues, 50), plot_settings['markerScaling']), linestyle='None', color='gray'),
           Line2D((),(), marker='o', ms=np.power(np.percentile(pieValues, 95), plot_settings['markerScaling']), linestyle='None', color='gray')]
    handles.extend(pieMarkerHandles)
    labelHandles.extend([np.round(np.percentile(pieValues, percentage), 2) for percentage in [5, 50, 95]])

    if transmission_dataframe is not None:
        for j,rw in transmission_dataframe.iterrows(): 
            gk.drawGeoms(rw.geom, ax=ax, srs=srs, lw=np.power(rw.weight, plot_settings['lineWidthScaling']), fontsize= plot_settings['fontSize'], color=plot_settings['transmissionLineColor'])
                

        transmissionValues = transmission_dataframe.values
        transmissionHandles =[Line2D((),(), lw=np.power(np.percentile(transmissionValues, 5), plot_settings['lineWidthScaling']), color='gray'), 
                            Line2D((),(), lw=np.power(np.percentile(transmissionValues, 50), plot_settings['lineWidthScaling']), color='gray'),
                            Line2D((),(), lw=np.power(np.percentile(transmissionValues, 95), plot_settings['lineWidthScaling']), color='gray'),]
        handles.extend(transmissionHandles)
        labelHandles.extend([np.round(np.percentile(transmissionValues, percentage), 2) for percentage in [5, 50, 95]])
    
    leg = ax.legend(handles, labelHandles, ncol=2, loc="upper right", fontsize=plot_settings['fontSize'], bbox_to_anchor=plot_settings['bbox_to_anchor'], 
                labelspacing=1.2, handlelength=1.3, handletextpad=0.5, framealpha=0)

    return fig, ax