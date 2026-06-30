Basic Component Model
#####################

The *Basic* component model comprises sets of variables, constraints,
inter-component constraint-contributions and objective function
contributions that apply to all components specified in
:math:`\mathcal{C}`. In this context, the variables and constraints can
be divided into either being time-independent or time-dependent. 

Component Parameters 
********************

Dimensioning of components
==========================

For each component :math:`c \in C`, capacity variables can be introduced with the parameter 
:math:`K_\text{c} \in \{0,1\}`: 

.. math::

   \begin{aligned}
       K_\text{c}=
       \begin{cases}
           1 &\text{\small , if the component is modeled with a physical capacity, or}\\
           0 &\text{\small , if the component is modeled without a physical capacity.}\\
       \end{cases} 
   \end{aligned}

A component which is modeled with physical capacity is for example a gas power plant while an electricity demand does not require
one.

The following parameters refer to all components
:math:`\text{c}\in\mathcal{C}` with :math:`K_\text{c}=1`:

.. list-table:: Basic capacity parameters

 * - **Parameter**
   - **Domain**
   - **Description**
 * - :math:`K^\text{min}_\text{c,l}`
   - | :math:`\mathbb{R}_0^+` with :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}`  
   - | minimum capacity of component c at location l 
 * - :math:`K^\text{max}_\text{c,l}`
   - | :math:`\mathbb{R}_0^+` with :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}`  
   - | maximum capacity of component c at location l 
 * - :math:`K^\text{fix}_\text{c,l}`
   - | :math:`\mathbb{R}_0^+` with :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}`  
   - | fixed capacity of component c at location l 
 * - :math:`K^\text{unit}_\text{c}`
   - | :math:`\mathbb{R}_0^+` with :math:`c \in \mathcal{C}`  
   - | capacity per plant unit of component c  
 * - :math:`B_\text{c}`
   - | :math:`\left\{0,1\right\}` with :math:`c \in \mathcal{C}`  
   - | introduces decision variable to state 
     | if a capacity is built or not 
 * - :math:`B^\text{fix}_\text{c,l}`
   - | :math:`\left\{0,1\right\}` with :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}`  
   - | fixes decision variable to state if a component c
     | is built or not at location l
 * - :math:`M_\text{c}`
   - | :math:`\mathbb{R}^+` with :math:`c \in \mathcal{C}`  
   - | required auxiliary parameter if :math:`K^\text{bin}_\text{c} = 1`
 * - :math:`\text{E}_\text{c,l}`
   - | :math:`\left\{0,1\right\}` with
     | :math:`c \in \mathcal{C}^\text{node}, l \in \mathcal{L}_\text{c}`  
   - | eligibility of node component c  
     | at location l 
 * - :math:`\text{E}_\text{c,a}`
   - | :math:`\left\{0,1\right\}` with
     | :math:`c \in \mathcal{C}^\text{edge}, a \in \mathcal{A}_\text{c}`  
   - | eligibility of edge component c 
     | at arc a


Operation of components
=======================

.. list-table:: Basic operation parameters

 * - **Parameter**
   - **Domain**
   - **Description**
 * - :math:`R^\text{min}_{\text{c,l,}\theta}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}, \theta \in \Theta`  
   - | minimum operation rate of component c 
     | at location l and time step t 
 * - :math:`R^\text{max}_{\text{c,l,}\theta}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}, \theta \in \Theta`  
   - | maximum operation rate of component c 
     | at location l and time step t 
 * - :math:`R^\text{fix}_{\text{c,l,}\theta}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}, \theta \in \Theta`  
   - | fixed operation rate of component c
     | at location l and time step t 

Cost contribution of components
===============================

.. list-table:: Basic cost parameters

 * - **Parameter**
   - **Domain**
   - **Description**
 * - :math:`\hat{X}^{\text{capex}_\text{K}}_\text{c,l}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}`  
   - | expenditures per capacity
 * - :math:`\hat{X}^{\text{opex}_\text{K}}_\text{c,l}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}`  
   - | operational expenditures per capacity
 * - :math:`\hat{X}^{\text{capex}_\text{B}}_\text{c,l}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}`  
   - | expenditures if capacity is built
 * - :math:`\hat{X}^{\text{opex}_\text{B}}_\text{c,l}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`c \in \mathcal{C}, l \in \mathcal{L}_\text{c}`  
   - | operational expenditures per capacity if capacity is built
 * - :math:`T^\text{EL}_\text{c}`
   - | :math:`\mathbb{Z}_0^+` with
     | :math:`c \in \mathcal{C}`  
   - | economic lifetime of component c 

Time-independent variables and constraints
******************************************

A capacity variable
:math:`k_\text{c,l}\in\mathbb{R}^{\geq0}`
is declared for all locations
:math:`\text{l}\in\mathcal{L}_\text{c}` in the
energy system at which the component can appear. Implicitly, this
capacity is modeled either as a continuous or discrete value by

.. math::

   \begin{aligned}
   k_\text{c,l} =
       \text{\small K}^\text{unit}_\text{c,l} \cdot n_\text{c,l} 
   \end{aligned}

with :math:`n_\text{c,l} \in \mathbb{R}_0^+` if the capacity is modeled as a continuous value, and with 
:math:`n_\text{c,l} \in \mathbb{N}_0^+` if the capacity is modeled as discrete value.
The variable :math:`n_\text{c,l}` describes the number of installed plant units for each component and 
location, and the parameter :math:`K^\text{unit}_\text{c}` describes the capacity per plant unit.

Furthermore, the component can be modeled together with a binary design
decision variable
:math:`b_\text{c,l}\in\left\{0,1\right\}`,
for all locations
:math:`\text{l}\in\mathcal{L}_\text{c}`, if its
boolean parameter :math:`B_\text{c}` is set to
true (=1). This modeling
approach is based on the work of `Bemporad and
Morari (1999) <https://doi.org/10.1016/S0005-1098(98)00178-2>`_ who give a general description
and discussion of this approach in the context of linear integer
programming. The optimal value of :math:`b_\text{c,l}` states whether a component is
built (=1) or not built (=0). The consideration of the
binary decision variables is enforced in the model for all
:math:`b_\text{c,l}` by the
constraint

.. math::

   \begin{aligned}
    \text{\small M}_\text{c} \cdot b_\text{c,l} ~\geq~ k_\text{c,l}~,
   \end{aligned}

where
:math:`\text{M}_\text{c}\in\mathbb{R}_0^{+}`. The
constraint enforces that
:math:`b_\text{c,l} = 1` if :math:`k_\text{c,l} > 1`. The parameter
:math:`\text{M}_\text{c}` has to be chosen large
enough such that it does not function as an upper limit on the
capacity. 

Lower and upper boundaries can be specified for the capacity variables
of the component. Lower bounds are enforced, if
:math:`\text{K}^\text{min}_\text{c,l} \in\mathbb{R}^{\geq0}`
is defined for all
:math:`\text{l} \in\mathcal{L}_\text{c}` of
this component, by

.. math::

   \begin{aligned}
   &&&k_\text{c,l} \geq
   \begin{cases}
       K^\text{min}_\text{c,l}\cdot b_\text{c,l} &,~\text{\small if}~ B_\text{c}=1,\\    
       K^\text{min}_\text{c,l} &,~\text{\small if}~ B_\text{c}=0.
   \end{cases}
   \end{aligned}

Upper bounds are enforced, if
:math:`\text{K}^\text{max}_\text{c,l}\in\mathbb{R}^{\geq0}`
is defined for all
:math:`\text{l} \in\mathcal{L}_\text{c}`, by

.. math::

   \begin{aligned}
   k_\text{c,l}  ~\leq~
   \text{K}^\text{max}_\text{c,l}~~.
   \end{aligned}

Moreover, for both the capacity and the binary decision variables, fixed
values can be individually specified for a component by

.. math::

   \begin{aligned}
   k_\text{c,l}  &~=~~ && K^\text{fix}_\text{c,l}~~\text{\small and} \\
   k^\text{bin}_\text{c,l}  &~=~~ && K^\text{bin,fix}_\text{c,l}~~,
   \end{aligned}

if
:math:`K^\text{fix}_\text{c,l} \in\mathbb{R}^{\geq0} ,~K^\text{bin,fix}_\text{c,l} \in \left\{0,1\right\}`
are defined for all
:math:`\text{l}\in\mathcal{L}_\text{c}`,
respectively.

Basic time-dependent variables and constraints
**********************************************

Operational variables
:math:`o_{\omega \text{,l,} \theta}\in\mathbb{R}^{\geq0}`
are declared for all operation types of a component :math:`\omega \in \Omega`, for all locations
:math:`\text{l}\in\mathcal{L}^\text{c}` and for
all periods and time steps :math:`\theta \in \Theta`. The compound index set
:math:`\Omega` is individually
defined in the respective component extension and describes which modes :math:`m \in \mathcal{M}` need to be considered for component :math:`c \in \mathcal{C}`. 
The compound index sets are described in :ref:`Compound Index Sets`. 

Each operation variable of a component that is modeled with a physical
capacity (:math:`K_\text{c} = 1`) is limited in one of four ways. 
First, the operation variable is limited by

.. math::

   \begin{aligned}
       o_{\omega \text{, l,} \theta}  ~\leq~ \text{\small T}^\text{hours} \cdot \text{\small a}_{\omega} \cdot k_\text{c,l} 
   \end{aligned}

if the operation of the component is merely limited by its capacity and
a time-independent factor :math:`\text{a}_{\omega}\in\mathbb{R}^{\geq0}` (default: 1) with :math:`\omega \in \Omega`. 

Second, the operation variable is fixed to

.. math::

   \begin{aligned}
       o_{\omega \text{,l,} \theta}  ~=~ \text{\small T}^\text{hours} \cdot \text{\small R}^\text{fix}_{\text{c,l,} \theta} \cdot k_\text{c,l}
   \end{aligned}

if a fixed, relative operation rate :math:`\text{R}^\text{fix}_{\text{c,l,} \theta}`
is specified for all locations :math:`\text{l}\in\mathcal{L}_\text{c}` and for all periods and time steps
:math:`\theta \in \Theta`. 

Third, the operation rate is limited by

.. math::

   \begin{aligned}
       o_{\omega \text{,l,} \theta}  ~\leq~ \text{\small T}^\text{hours} \cdot \text{\small R}^\text{max}_{\text{c,l,} \theta} \cdot k_\text{c,l}
   \end{aligned}

if a maximum, relative operation rate :math:`\text{R}^\text{max}_{\text{c,l,} \theta}`
is specified for all locations :math:`\text{l}\in\mathcal{L}_\text{c}` and for
all periods and time steps :math:`\theta \in \Theta`. 

Lastly, the operation rate is limited by

.. math::

   \begin{aligned}
       o_{\omega \text{,l,} \theta}  ~\geq~ \text{\small T}^\text{hours} \cdot \text{\small R}^\text{min}_{\text{c,l,} \theta} \cdot k_\text{c,l}
   \end{aligned}

if a minimum, relative operation rate :math:`\text{R}^\text{min}_{\text{c,l,} \theta}`
is specified for all locations :math:`\text{l}\in\mathcal{L}_\text{c}` and for
all periods and time steps :math:`\theta \in \Theta`. 


Each operation variable of a component which is modeled without a
physical capacity (:math:`K_\text{c} = 0`) is limited in one of three ways: 
The operation variable is fixed to

.. math::

   \begin{aligned}
       o_{\omega \text{,l,} \theta}  ~=~ \text{\small T}^\text{hours} \cdot \text{\small R}^\text{fix}_{\text{c,l,} \theta}
   \end{aligned}

if a fixed, relative operation rate :math:`\text{R}^\text{fix}_{\text{c,l,} \theta}`
is specified for all locations :math:`\text{l}\in\mathcal{L}_\text{c}` and for all periods and time steps
:math:`\theta \in \Theta`.  This
constraint can apply, for example, to the model of an electricity
demand. 

The operation variable is limited by

.. math::

   \begin{aligned}
       o_{\omega \text{,l,} \theta}  ~\leq~ \text{\small T}^\text{hours} \cdot \text{\small R}^\text{max}_{\text{c,l,} \theta}
   \end{aligned}

if a maximum, relative operation rate :math:`\text{R}^\text{max}_{\text{c,l,} \theta}`
is specified for all locations :math:`\text{l}\in\mathcal{L}_\text{c}` and for
all periods and time steps :math:`\theta \in \Theta`. This constraint
can apply, for example, to the model of an optional commodity import.

The operation variable is limited by

.. math::

   \begin{aligned}
       o_{\omega \text{,l,} \theta}  ~\geq~ \text{\small T}^\text{hours} \cdot \text{\small R}^\text{min}_{\text{c,l,} \theta}
   \end{aligned}

if a minimum, relative operation rate :math:`\text{R}^\text{min}_{\text{c,l,} \theta}`
is specified for all locations :math:`\text{l}\in\mathcal{L}_\text{c}` and for
all periods and time steps :math:`\theta \in \Theta`. 

Basic inter-component constraint contributions
**********************************************

Inter-component constraint contributions are defined to model
constraints which do affect multiple components. The contributions are specified for each component individually
and are afterwards aggregated to comprehensive constraints.

The constraints which model the basic structure of the energy system are
thereby the commodity balance constraints. They have to be defined for
all commodities :math:`\text{g} \in\mathcal{G}`, at all locations in
:math:`\text{l}\in\mathcal{L}` at which the commodity appears and
there for all periods and time steps
:math:`\theta \in \Theta`. The contribution
of a component to a balance equation is labeled
:math:`C_{\text{c,g,l,}\theta}` and
has to be defined for each component which is added to the model. This
takes place in the individual component model extensions.

Moreover, two or more components can compete for a limited capacity
potential in an energy system. For example, existing salt caverns can be
dedicated to be used for either hydrogen or methane storage. Components
which share a potential in FINE are provided with an identifier. If an identifier is defined for a
component, the share of that component on the maximum potential is at
all locations :math:`\text{l} \in\mathcal{L}^\text{c}` defined by :math:`k_\text{c,l}/\text{\small k}^\text{max}_\text{c,l}`. 

Basic objective function contribution
*************************************

The objective function in the framework is defined as the net present value :math:`NPV` of all components :math:`\text{c} \in \mathcal{C}` and is
minimized during optimization. As for the inter-component constraint
contributions, the objective function contributions
:math:`NPV_\text{c}` [costUnit/a] are specified for each component
individually by

.. math::

   \begin{aligned}
       NPV_\text{c} =& \sum\limits_{\text{l}~\in~\mathcal{L}_\text{c}}\hspace{-3pt}
           \left( NPV^\text{K}_\text{c,l} ~+~ NPV^\text{B}_\text{c,l} ~+~ NPV^\text{O}_\text{c,l} \right) 
   \end{aligned}

and are aggregated to one comprehensive objective function
afterwards. The capacity related total annual cost contributions are
determined by

.. math::

   \begin{aligned}
       &NPV^\text{K}_\text{c,l} = \text{\small F}^\text{K}_\text{c,l} \cdot \left(\frac{\hat{X}^{\text{capex}_\text{K}}_\text{c,l}}{\text{\small CCF}_\text{c,l}} + \hat{X}^{\text{opex}_\text{K}}_\text{c,l}\right) \cdot k_\text{c,l}
   \end{aligned}

if the component is modeled with a physical capacity. Otherwise,
:math:`NPV^\text{K}_\text{c,l}` is set
to 0. The parameters
:math:`\hat{X}^{\text{capex}_\text{K}}_\text{c,l}`
[costUnit/nominalCapacity] and
:math:`\hat{X}^{\text{opex}_\text{K}}_\text{c,l}\in\mathbb{R}^{\geq0}`
[costUnit/(nominalCapacity\ :math:`\cdot`\ a)] describe the capital and
annual operational expenditures in relation to the capacity. The
parameter
:math:`\text{F}^\text{K}_\text{c,l}` can
be defined individually for a component (default: 1). The total annual
cost contributions related to the binary decision variables are
determined by

.. math::

   \begin{aligned}
       &NPV^\text{B}_\text{c,l} = \text{\small F}^\text{B}_\text{c,l} \cdot \left(\frac{\hat{X}^{\text{capex}_\text{B}}_\text{c,l}}{\text{\small CCF}_\text{c,l}} + \hat{X}^{\text{opex}_\text{B}}_\text{c,l}\right) \cdot b_\text{c,l} 
   \end{aligned}

if the component is modeled with binary decision variables. Otherwise
:math:`NPV^\text{B}_\text{c,l}` is set
to 0. The parameters
:math:`\hat{X}^{\text{capex}_\text{B}}`
[costUnit] and
:math:`\hat{X}^{\text{opex}_\text{B}}\in\mathbb{R}^{\geq0}`
[costUnit/a] describe the capital and annual operational expenditures
which arise if the component is built. The parameter
:math:`\text{F}^\text{B}_\text{c,l}` can
be defined individually for a component (default: 1). The factor

.. math::

   \begin{aligned}
       &\text{\small CCF}_\text{c,l} = \frac{1}{\text{\small WACC}_\text{c,l}}-\frac{1}{\left(1+\text{\small WACC}_\text{c,l}\right)^{\text{\small T}^\text{EL}_\text{c}}\cdot\text{\small WACC}_\text{c,l}} 
   \end{aligned}

is applied to determine the annuity of the respective invest for one
calender year. Thus,
:math:`\text{WACC}_\text{c,l}\in(0,1]`
is the weighted average cost of capital and
:math:`T^\text{EL}_\text{c}\in\mathbb{Z}_0^{+}{}`
[a] is the economic lifetime of the component in years. 

With the combination of a capacity-dependent and a capacity-independent cost
factor, a simplified nonlinear *economy-of-scale* approach is
realized. The operation related total annual cost contributions are
determined by

.. math::

   \begin{aligned}
       &NPV^\text{O}_\text{c,l} = \hspace{-3pt}
       \sum\limits_{\substack{\theta \\ \in~\Theta}}\hspace{4pt}
       \sum\limits_{\substack{\text{m} \\ \in~\mathcal{M}\text{c}}}
       \text{\small F}^\text{O}_{\omega \text{,l}}\hspace{-3pt}\cdot o_{\omega,\text{,l,} \theta} \cdot \frac{f\left(p\right)}{\text{\small T}^\text{years}} 
   \end{aligned}

where :math:`F^\text{O}_{\omega \text{,l}}`
[costUnit/(nominalCapacity\ :math:`\cdot`\ h)] is defined in the
individual component model extensions.