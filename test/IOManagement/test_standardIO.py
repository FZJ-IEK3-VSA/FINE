import FINE.IOManagement.standardIO as stdio
import os

def test_plotLocationalPieMap(multi_node_test_esM):
    """Test this function"""

    compNames = ['Wind (onshore)', 'Wind (offshore)', 'PV', ]

    locationsShapeFileName = os.path.join(os.path.join(os.path.dirname(__file__), 
        "../../examples/Multi-regional Energy System Workflow/InputData/SpatialData/ShapeFiles/clusteredRegions.shp"))

    plot_settings = {}
    plot_settings['markerScaling'] = 1/4

    stdio.plotLocationalPieMap(multi_node_test_esM, compNames, locationsShapeFileName, variableName='operationVariablesOptimum', doSum=True, plot_settings=plot_settings)

    stdio.plotLocationalPieMap(multi_node_test_esM, compNames, locationsShapeFileName, variableName='capacityVariablesOptimum', doSum=False)
