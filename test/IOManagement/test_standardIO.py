import FINE.IOManagement.standardIO as stdio
import os

def test_plotLocationalPieMap(multi_node_test_esM_optimized):
    """Test this function"""

    compNames = ['Wind (onshore)', 'Wind (offshore)', 'PV', ]

    locationsShapeFileName = os.path.join(os.path.join(os.path.dirname(__file__), 
        "../../examples/Multi-regional_Energy_System_Workflow/InputData/SpatialData/ShapeFiles/clusteredRegions.shp"))

    plot_settings = {}
    plot_settings['markerScaling'] = 1/4

    stdio.plotLocationalPieMap(multi_node_test_esM_optimized, compNames, locationsShapeFileName, variableName='operationVariablesOptimum', doSum=True, plot_settings=plot_settings)

    stdio.plotLocationalPieMap(multi_node_test_esM_optimized, compNames, locationsShapeFileName, variableName='capacityVariablesOptimum', doSum=False)
