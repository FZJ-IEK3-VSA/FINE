DC power flow extension
#######################

A basic *Transmission* component is modeled with a simple commodity
exchange based on balance equations and a linear loss factor. However,
the transmission of a commodity is generally subject to far more complex
physics. The incorporation of a higher modeling detail of these physics
into the optimization program has to be seen in the context of
increasing computation times. With respect to this topic, `Syranidis et
al. (2018) <https://doi.org/10.1016/j.rser.2017.10.110>`_ reviewed the modeling of
electrical power flow across transmission networks. They discuss the
general formulation of an AC power flow with a set of non-linear
equations for which direct, analytical solutions are rarely feasible and
which are therefore often solved with iterative methods. Based on the
premise that the optimization program provided by FINE should stay a
mixed integer linear program, these equations cannot be incorporated in
the framework. A linearization of these equations, as provided by the DC
power flow method, is however suitable for incorporation. The
linearized equations result in an acceptable increase in computation
time while increasing the electrical power flow modeling detail to a
more sophisticated level.

In the following, the constraints constituting the DC power flow are
presented, based on the detailed description by `Van den Bergh et
al. (2014) <https://www.mech.kuleuven.be/en/tme/research/energy_environment/Pdf/wpen2014-12.pdf>`_. The constraints thereby extend
the *Transmission* component model. In the following, let
:math:`\mathcal{C}^\text{trans,LPF}\subseteq\mathcal{C}^\text{trans}\subset\mathcal{C}`
be the set of *Transmission* components that are modeled with a DC power
flow.

The constraints that enforce the linear power flow are implemented for
each component :math:`\text{c}\in\mathcal{C}^\text{trans,LPF}`, for
all :math:`\text{l}\in\mathcal{L}^\text{c}`, and for all
:math:`\theta \in \Theta` as

.. math::

   \begin{aligned}
       o_\text{$\omega$,a,$\theta$}-o_{\omega,\hat{\text{a}},\theta}=\left(\phi^\text{c,l$_1$,$\theta$}-\phi_\text{c,l$_2$,$\theta$}\right) / \text{x}_\text{c,a}~.
   \end{aligned}

Here, :math:`\phi_\text{c,l,p,t}\in\mathbb{R}` is the
variable which models the phase
angle. :math:`\text{x}_\text{c,a}`
represents the electric reactance of the line between locations
l\ :math:`_1` and l\ :math:`_2` (:math:`\text{a} \in \mathcal{A}_\text{c}`). These equations
leave one degree of freedom for the phase angle variables at each time
step. To obtain a unique solution, an additional set of constraints is
given by

.. math::

   \begin{aligned}
       \phi_\text{c,l$_\text{ref}$,$\theta$}=0
   \end{aligned}

for each component :math:`\text{c}\in\mathcal{C}^\text{trans,LPF}`
and for all :math:`\theta \in \Theta` which
sets the phase angle for one location :math:`\text{l}_\text{ref}` to
zero.

At this point, it should be remarked that the reactance parameter is in
practice a function of the capacity of the line. The capacity expansion of transmission lines modeled with a *DC power flow* is not implemented to reduce model complexity,
i.e., AC line capacities, which are modeled with a *DC power flow*, are kept at a fixed value and thus their reactance parameters remain constant.
