Inter-Component Constraints
###########################

Inter-component constraints are constraints that involve variables and
parameters from multiple components in :math:`\mathcal{C}`. The
inter-component constraints which are modeled within the framework are
commodity balances, annual commodity inflow/ outflow limits and shared potential constraints.

Commodity balances
******************

The constraints that provide the basic structure of the energy system
are the commodity balances. They are defined for all commodities
:math:`\text{g}\in\mathcal{G}`, at all locations in
:math:`\text{l}\in\mathcal{L}`, if the commodity appears at that
location in the model, and, there, for all periods and time steps
:math:`\theta \in \Theta`. The commodity
appears at a location when the set

.. math::

   \begin{aligned}
       &\mathcal{C}_\text{g,l} ~=~ \big\{ &&\text{\small c} ~\vert~ \forall~ \text{\small c}\in\mathcal{C}:
        \text{\small g}\in\mathcal{G}^\text{c}~\wedge~ \big(\text{\small l}\in\mathcal{L}_\text{c}~\lor \nonumber \\
       & &&(\exists~\text{\small l\text{*}}\in\mathcal{L}: (\text{\small l},\text{\small l*})\in\mathcal{L}_\text{c}~\lor~(\text{\small l*},\text{\small l})\in\mathcal{L}_\text{c})\big)\big\}
   \end{aligned}

is not empty. In this case the commodity balance equation is given for
all as

.. math::

   \begin{aligned}
       &\sum\limits_{\text{c}~\in~\mathcal{C}_\text{g,l}} &&C_\text{c,g,l,$\theta$} ~=~ 0~.
   \end{aligned}

The definition of :math:`C_\text{c,g,l,$\theta$}` is given in
the component model extensions.

Shared potential constraints
****************************

As already explained in the *Basic* component model, two or more
components can share a potential in an energy system. The framework
ensures that for each location/ connection where a shared potential is
specified the share on the maximum capacity of all components with the
same identifier does not exceed 100%. Each component for which a maximum capacity is defined can
be associated with the shared potential by setting the parameter
:math:`\text{sharedPotentialID}_\text{c}=\text{sharedPotentialID}`
(default: :math:`\emptyset`). Let

:math:`\mathcal{I}^\text{ID}` be the
set containing all shared potential IDs and let
:math:`\mathcal{L}^\text{ID}` be the set of locations or
connections at which components compete for a maximum potential,
respectively. The shared potential constraints are then given for all
:math:`\text{i}\in\mathcal{I}^\text{ID}`
and all :math:`\text{l}\in\mathcal{L}^\text{ID}` by

.. math::

   \begin{aligned}
       &\sum\limits_{\text{c}~\in~\mathcal{C}^\text{i}}
       k_\text{c,l}/\text{\small K}^\text{max}_\text{c,l} ~\leq~ 1,\nonumber \\
       &\text{with}~~\mathcal{C}^\text{i} = \left\{ \text{\small c} ~\vert~ \forall~ \text{\small c}\in\mathcal{C}:
        \text{\small sharedPotentialID}_\text{c}=\text{\small i} \right\}~.
   \end{aligned}
