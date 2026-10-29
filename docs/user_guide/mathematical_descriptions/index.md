# Mathematical Descriptions

The underlying mathematical structure of FINE leads to big linear optimization problems,
mixed-integer linear optimization problems, or mixed-integer quadratic optimization problems.
The objective function describes for the case of FINE the net present value of the system which is to be minimized.
The constraints enforce that the operation and design of the system is within eligible technical and ecological boundaries.
Variables are for example the capacity of a component or its operation in each region and at each time step.
The structure allows considering several investment periods.
The following applies: The net present value equals the total annual costs of the system if the modeled time horizon
is set up with only one investment period.

The mathematical description is based on the description in
[Welder (2022)](https://publications.rwth-aachen.de/record/861215/files/861215.pdf)
and has been updated with the changes of the latest ETHOS.FINE version.

A more detailed description of the underlying mathematical optimization problem will be provided in a future release.

## Contents

- [Parameters and Sets](parameters_and_sets.md)
- [Basic Component Model](basic_component.md)
- [Source and Sink](source_sink.md)
- [Conversion](conversion.md)
- [Storage](storage.md)
- [Transmission](transmission.md)
- [Inter-Component Constraints](inter_component.md)
- [Objective Function](objective_function.md)
