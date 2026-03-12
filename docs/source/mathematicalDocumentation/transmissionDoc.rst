Transmission Component Model Extension
######################################

Components which transmit commodities between locations in
:math:`\mathcal{L}` and along arcs :math:`\text{a} \in \mathcal{A}`
are modeled as bidirectional *Transmission* components by default. Examples of
*Transmission* components are electric lines or bidirectional gas
pipelines. The *Transmission* component model extends the *Basic*
component model. In the following, the set of all *Transmission*
components is labeled
:math:`\mathcal{C}^\text{trans}\subset\mathcal{C}^\text{edge}`. 

Specification of operation variables and associated commodities
***************************************************************

A *Transmission* component
:math:`\text{c}\in\mathcal{C}^\text{trans}` only has one type of
basic operation variables. It is associated with one
commodity :math:`\mathcal{G}^\text{c}=\{\text{g}\}`, with
:math:`\text{g}\in\mathcal{G}` , which is the commodity that the
component transmits. If a capacity is defined for this component, it is
related to this commodity. For example, the capacity of an electric line
is related to the nominal electric power it can transmit, e.g. in
MW\ :math:`_\text{el}`.

Specification of additional constraints
***************************************

A *Transmission* component can be operated bidirectionally. This means
that the flow from :math:`\text{l}_\text{1}\in\mathcal{L}` to
:math:`\text{l}_\text{2}\in\mathcal{L}`, which is described as arc :math:`\text{a}`, has to use the same route and
infrastructure as a flow from :math:`\text{l}_\text{2}` to
:math:`\text{l}_\text{1}`, which is describes as arc :math:`\hat{\text{a}}`. To enforce this behavior,
the constraint

.. math::

   \begin{aligned}
       k_\text{c,a} = k_\text{c,$\hat{\text{a}}$}
   \end{aligned}

is stated for all :math:`\text{c}\in\mathcal{C}^\text{trans}` and all
:math:`\text{a,$\hat{\text{a}}$}\in\mathcal{A}_\text{c}`.  
Furthermore, the equation of the maximum operation of a component is supplemented with the
equation

.. math::

   \begin{aligned}
       o_{\omega\text{,a,}\theta}+o_{\omega\text{,$\hat{\text{a}}$,}\theta} \leq \text{\small T}^\text{hours} \cdot k_\text{c,a} \label{eqTransBasic}
   \end{aligned}

for all :math:`\text{c}\in\mathcal{C}^\text{trans}` and all
:math:`\text{a,} \hat{\text{a}} \in \mathcal{A}_\text{c}`. This
set of equations increases the tendency that, for basic optimization
solutions, one of the commodity flows
:math:`o_{\omega\text{,a,}\theta}` or
:math:`o_{\omega\text{,$\hat{\text{a}}$,}\theta}` is
set to zero.

Specification of commodity balance contributions
************************************************

Contributions to the commodity balance equations are modeled for a
:math:`\text{c}\in\mathcal{C}^\text{trans}`, for
:math:`\text{g}\in\mathcal{G}^\text{c}`, for all
:math:`\text{l}\in\mathcal{L}`, and
for all :math:`\theta \in \Theta`.

To describe the commodity balance equations, we define two sets of arcs:
The set :math:`\text{a}^\text{in} \in \mathcal{A}^\text{c}` equals :math:`(\text{l,l}_\text{out})` 
and includes all eligible connections for commodity flows from connected locations to location l.
The set :math:`\text{a}^\text{out} \in \mathcal{A}^\text{c}` equals :math:`(\text{l}_\text{in}\text{,l})` 
and includes all eligible connections for commodity flows from location l to connected locations.

.. math::

   \begin{aligned}
       C_\text{c,g,l,$\theta$} ~=~  
       \sum\limits_{\substack{\text{a}^\text{in}~\in~\mathcal{A}^\text{c}}}
       (1-\eta_{\text{a}^\text{in}} \cdot \text{\small d}_{\text{a}^\text{in}}) \cdot o_{\omega \text{,a}^\text{in} \text{,} \theta} - \sum\limits_{\substack{\text{a}^\text{out} ~\in~\mathcal{A}^\text{c}}} o_{\omega \text{,a}^\text{out} \text{,} \theta}~.
   \end{aligned}

Here, :math:`\eta_\text{a}^\text{in}` is a linear
loss factor per length and
capacity. :math:`\text{d}_\text{a}^\text{in}` is
the length between the two connected locations. The term thus represents incoming and outgoing
flows of a commodity g at the location l at period p and time step t.

Specification of objective function contributions
*************************************************

The parameters
:math:`\text{F}^\text{K}_\text{c,a}`
and
:math:`\text{F}^\text{B}_\text{c,a}`
in equations for the objective function contribution
are set equal to
:math:`1/2 \cdot \text{\small d}_\text{a}`
for *Transmission* components. The factor :math:`1/2` compensates that
each connection is taken into account twice in the objective
function. The length d of the connection is included so that the capital
and operational cost factors can be given as not only capacity but also
length related.

The cost factor :math:`\text{F}^\text{O}_\text{$\omega$,a}`, is given as

.. math::

   \begin{aligned}
       &~~\text{\small F}^\text{O}_\text{$\omega$,a} ~=~ \hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}}~.\ \label{eqCostOpTrans}
   \end{aligned}

with :math:`\hat{X}^{\text{opex}_\text{O}}_{\omega\text{,a}}\in\mathbb{R}^{\geq0}` which describes the costs per operation of component c.

.. toctree::
   :maxdepth: 1

   DCpowerflowDoc