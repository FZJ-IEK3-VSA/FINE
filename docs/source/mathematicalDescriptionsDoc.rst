Mathematical Descriptions
#########################

The underlying mathematical structure of FINE leads to big linear optimization problems,
mixed-integer linear optimization problems, or mixed-integer quadratic optimization problems.
The objective function describes for the case of FINE the net present value of the system which is to be minimized.
The constraints enforce that the operation and design of the system is within eligible technical and ecological boundaries.
Variables are for example the capacity of a component or its operation in each region and at each time step.
The structure allows to consider several investment periods. 
The following applies: The net present value equals the total annual costs of the system if the modeled time horizon is set up with only one investment period. 

The mathematical description is based on the description in `Welder (2022) <https://publications.rwth-aachen.de/record/861215/files/861215.pdf>`_ 
and has been updated with the changes of the latest ETHOS.FINE version.

.. toctree::
   :maxdepth: 2

   mathematicalDocumentation/parametersAndSetsDoc
   mathematicalDocumentation/basicComponentDoc
   mathematicalDocumentation/sourceSinkDoc
   mathematicalDocumentation/conversionDoc
   mathematicalDocumentation/storageDoc
   mathematicalDocumentation/transmissionDoc
   mathematicalDocumentation/interComponentDoc
   mathematicalDocumentation/objectiveFunctionDoc

A more detailed description of the underlying mathematical optimization problem will be provided in a future release.
