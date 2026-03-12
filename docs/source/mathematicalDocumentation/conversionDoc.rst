Conversion Component Model Extension
####################################

A component which converts one set of commodities into another set of
commodities, as for example a power plant is modeled to convert natural
gas into electricity and carbon dioxide, is modeled in FINE as a
so-called *Conversion* component. The *Conversion* component model
thereby extends the *Basic* component model. In the following, the set
of all *Conversion* components is labeled
:math:`\mathcal{C}^\text{conv}\subset\mathcal{C}^\text{node}`. 

Specification of operation variables and associated commodities
***************************************************************

A *Conversion* component :math:`\text{c}\in\mathcal{C}^\text{conv}`
only has one type of basic operation variables. It can however be
associated with multiple
commodities, as it converts commodities into each other. The nominal capacity of a
*Conversion* component is related to one of these commodities labeled
:math:`\text{g}^\text{nominal}`. For example, the capacity of an
electrolyzer can be related to either the consumed electricity, 
or the lower heating value (LHV) of the
generated hydrogen.

Specification of commodity balance contributions
************************************************

Inherently, a *Conversion* component contributes to the balance
equations of multiple commodities. These
contributions are modeled for a
:math:`\text{c}\in\mathcal{C}^\text{conv}`, for all
:math:`\text{g}\in\mathcal{G}^\text{c}`, for all
:math:`\text{l}\in\mathcal{L}^\text{c}` and for all
:math:`\theta \in \Theta` as

.. math::

   \begin{aligned}
       &C_{\text{c,g,l,}\theta} ~=~ \text{\small cf}_\text{c,g} \cdot o_{\omega\text{,l,}\theta}.\ \label{eqCconv}
   \end{aligned}

The conversion factor :math:`\text{cf}_\text{c,g}\in\mathbb{R}`
is by convention negative if a commodity is consumed and positive if a
commodity is generated. The nominal conversion factor
:math:`\big|\text{cf}_\text{c,g$^\text{nominal}$} \big|`
is set to 1. 

Specification of objective function contributions
**************************************************

The cost factor :math:`\text{F}^\text{O}_{\omega\text{,l}}` is for a *Conversion* component
:math:`\text{c}\in\mathcal{C}^\text{conv}` given as

.. math::

   \begin{aligned}
       &~~\text{\small F}^\text{O}_{\omega \text{,l}} ~=~ \hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}}~.\ 
   \end{aligned}