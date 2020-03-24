*************************
Mathematical Descriptions
*************************

The underlying mathematical structure of FINE leads to big linear optimization problems

.. math::
   & \min{c^Tx} \\
   s.t. \> & Ax=b \\
   & Cx \leq d

mixed-integer linear optimization problems

.. math::
   & \min{c^Tx} \\
   s.t. \> & Ax = b \\
   & Cx \leq d \\
   & x_i \in \mathbb Z \qquad \forall i \in \mathfrak{I}

or mixed-integer quadratic optimization problems

.. math::
   & \min{\frac{1}{2}x^TQx+c^Tx} \\
   s.t. \> & Ax=b \\
   & Cx \leq d \\
   & x_i \in \mathbb Z \qquad \forall i \in \mathfrak{I}

The objective function describes for the case of FINE the total annual cost of the system (which is to be minimized).
The constraints enforce that the operation and design of the system is within eligible technical and ecological boundaries.
Variables represent are for example the capacity of a component or its operation in each region and at each time step.

A more detailed description of the underlying mathematical optimization problem will be provided in a future release.

