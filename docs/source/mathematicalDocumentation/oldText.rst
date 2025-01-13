Annual commodity inflow/outflow limit
*************************************

The annual commodity limitation constraints implemented in the framework
enable the modeling of, for example, annual greenhouse gas emission
limits. A commodity limitation is modeled with an identifier
(:math:`\text{commLimitID}\in\mathbb{S}`) and a limit
(:math:`\text{commLimit}^\text{commLimitID}\in\mathbb{R}`). Each
component in :math:`\mathcal{C}^\text{srcSnk}` that generates or
consumes the commodity of interest can be associated with this ID by
setting the parameter
:math:`\text{commLimitID}^\text{comp}=\text{commLimitID}` (default:
:math:`\emptyset`).

Let :math:`\mathcal{I}^\text{commLimitIDs}` be the
set containing all specified annual commodity limitation IDs. Then, the
constraints limiting the total annual commodity inflow
(:math:`\text{commLimit}^\text{commLimitID}\leq0`) or outflow
(:math:`\text{commLimit}^\text{commLimitID}\geq0`) across the energy
system’s virtual boundary are given for all
:math:`\text{ID}\in\mathcal{I}^\text{commLimitIDs}`
by

.. math::

   \begin{aligned}
       &\sum\limits_{\text{comp}~\in~\mathcal{C}^\text{ID}}
       && -1 \cdot op^\text{comp,op}_\text{annual} \cdot  \text{\small sign}^\text{ID} \leq
       \text{\small commLimit}^\text{ID} \cdot  \text{\small sign}^\text{ID}, ~~\text{with}\nonumber \\
       &\mathcal{C}^\text{ID} &&=~ \left\{ \text{\small comp} ~\vert~ \forall~\text{\small comp}\in\mathcal{C}^\text{srcSnk}:
        \text{\small commLimitID}^\text{comp}=\text{\small ID} \right\}, \nonumber \\
       &op^\text{comp,op}_\text{annual} &&=~ \sum\limits_{\text{loc}~\in~\mathcal{L}^\text{comp}}~
       \sum\limits_{(\text{p,t})~\in~\mathcal{P}\times\mathcal{T}}
       \text{\small sign}^\text{comp} \cdot op_\text{loc,p,t}^\text{comp,op} \cdot freq\left(\text{\small p}\right) / \tau^\text{years} ~~\text{and} \nonumber \\
       &\text{\small sign}^\text{ID} &&=~ \frac{\text{\small commLimit}^\text{ID}}{\left|\text{\small commLimit}^\text{ID}\right|}~.
   \end{aligned}
