Source/Sink Component Model Extension
#####################################

Components which generate or consume commodities across the energy
system’s boundary are modeled as so-called *Source*/*Sink*
components. Examples for *Source* components are wind turbines or
natural gas imports. Examples for *Sink* components are electricity
demands or electricity exports. The *Source*/*Sink* component model
extends the *Basic* component model. In the following, the set of all
*Source* and *Sink* components is labeled
:math:`\mathcal{C}^\text{srcSnk}\subseteq\mathcal{C}^\text{node}`. 

Specification of operation variables and associated commodities
***************************************************************

A *Source*/ *Sink* component
:math:`\text{c}\in\mathcal{C}^\text{srcSnk}` only has one type of
basic operation variables
:math:`\mathcal{O}^\text{c}=\{\text{op}\}`. It is associated with one
commodity :math:`\mathcal{G}^\text{c}=\{\text{\small g}\}`,
:math:`\text{g}\in\mathcal{G}`, which is the commodity that the
component generates or consumes. If a capacity is defined for this
component, it is related to this commodity. For example, the capacity of
a wind turbine is related to the electric power which it generates at
full load, e.g. in MW\ :math:`_\text{el}`.

Specification of commodity balance contributions
************************************************

Contributions to the commodity balance equations are modeled for a component
:math:`\text{c}\in\mathcal{C}_\text{srcSnk}`, for
:math:`\text{m}\in\mathcal{M}_\text{c}`, for 
:math:`\text{g}\in\mathcal{G}_\text{c}`, for all
:math:`\text{l}\in\mathcal{L}_\text{c}` and for all
:math:`\theta\in\Theta` as

.. math::

   \begin{aligned}
       &C_{\text{c,g,l,}\theta} ~=~ \text{\small sign}_\text{c} \cdot o_{\omega\text{,l,}\theta}, ~~\text{\small where}\nonumber \\
       &\text{\small sign}^\text{c} =
       \begin{cases}
           +1 &,~\text{\small if c is a \emph{Source} component, and} \\
           -1 &,~\text{\small if c is a \emph{Sink} component}~.\ 
       \end{cases}
   \end{aligned}

Specification of objective function contributions
*************************************************

.. list-table:: Cost and revenue parameters for *Source*/*Sink* component

 * - **Parameter**
   - **Domain**
   - **Description**
 * - :math:`\hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`\omega \in \Omega_\text{srcSnk}, l \in \mathcal{L}_\text{c}`  
   - | expenditures per operation of component c
 * - :math:`\hat{X}^{\text{g}}_{\omega\text{,l}}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`g \in \mathcal{G}_\text{c}, \omega \in \Omega_\text{srcSnk}, l \in \mathcal{L}_\text{c}`  
   - | expenditures per unit of commodity g
 * - :math:`\hat{V}^{\text{g}}_{\omega\text{,l}}`
   - | :math:`\mathbb{R}_0^+` with
     | :math:`g \in \mathcal{G}_\text{c}, \omega \in \Omega_\text{srcSnk}, l \in \mathcal{L}_\text{c}`  
   - | revenues per unit of commodity g

The cost factor :math:`\text{F}^\text{O}_{\omega\text{,l}}`, is for a *Source*/*Sink*
component :math:`\text{c}\in\mathcal{C}^\text{srcSnk}` given as

.. math::

   \begin{aligned}
       &~~\text{\small F}^\text{O}_{\omega \text{,l}} = \big(\hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}} + 
       \hat{X}^{\text{g}}_{\omega\text{,l}} + \hat{V}^{\text{g}}_{\omega\text{,l}} ~\big)~.\ 
   \end{aligned}

Thus, operational cost as well as a cost and revenue for the associated
generated or consumed commodity can be considered with the parameters
:math:`\hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}}\in\mathbb{R}^{\geq0}`,
:math:`\hat{X}^{\text{g}}_{\omega\text{,l}}\in\mathbb{R}^{\geq0}`
and
:math:`\hat{V}^{\text{g}}_{\omega\text{,l}}\in\mathbb{R}^{\leq0}`
respectively.