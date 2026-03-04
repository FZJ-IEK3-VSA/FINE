# Source/Sink Component Model Extension

Components which generate or consume commodities across the energy system's boundary are modeled as
so-called *Source*/*Sink* components. Examples for *Source* components are wind turbines or natural gas
imports. Examples for *Sink* components are electricity demands or electricity exports. The *Source*/*Sink*
component model extends the *Basic* component model. In the following, the set of all *Source* and *Sink*
components is labeled $\mathcal{C}^\text{srcSnk}\subseteq\mathcal{C}^\text{node}$.

## Specification of Operation Variables and Associated Commodities

A *Source*/*Sink* component $\text{c}\in\mathcal{C}^\text{srcSnk}$ only has one type of basic operation
variables $\mathcal{O}^\text{c}=\{\text{op}\}$. It is associated with one commodity
$\mathcal{G}^\text{c}=\{\text{\small g}\}$, $\text{g}\in\mathcal{G}$, which is the commodity that the
component generates or consumes. If a capacity is defined for this component, it is related to this commodity.
For example, the capacity of a wind turbine is related to the electric power which it generates at full load,
e.g. in MW$_\text{el}$.

## Specification of Commodity Balance Contributions

Contributions to the commodity balance equations are modeled for a component
$\text{c}\in\mathcal{C}_\text{srcSnk}$, for $\text{g}\in\mathcal{G}_\text{c}$, for all
$\text{l}\in\mathcal{L}_\text{c}$ and for all $\theta\in\Theta$ as

$$
\begin{aligned}
    &C_{\text{c,g,l,}\theta} ~=~ \text{\small sign}_\text{c} \cdot o_{\omega\text{,l,}\theta}, ~~\text{\small where}\nonumber \\
    &\text{\small sign}^\text{c} =
    \begin{cases}
        +1 &,~\text{\small if c is a \emph{Source} component, and} \\
        -1 &,~\text{\small if c is a \emph{Sink} component}~.
    \end{cases}
\end{aligned}
$$

## Specification of Objective Function Contributions

| **Parameter** | **Domain** | **Description** |
|---|---|---|
| $\hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}}$ | $\mathbb{R}_0^+$ with $\omega \in \Omega_\text{srcSnk}, l \in \mathcal{L}_\text{c}$ | expenditures per operation of component c |
| $\hat{X}^{\text{g}}_{\omega\text{,l}}$ | $\mathbb{R}_0^+$ with $g \in \mathcal{G}_\text{c}, \omega \in \Omega_\text{srcSnk}, l \in \mathcal{L}_\text{c}$ | expenditures per unit of commodity g |
| $\hat{V}^{\text{g}}_{\omega\text{,l}}$ | $\mathbb{R}_0^+$ with $g \in \mathcal{G}_\text{c}, \omega \in \Omega_\text{srcSnk}, l \in \mathcal{L}_\text{c}$ | revenues per unit of commodity g |

The cost factor $\text{F}^\text{O}_{\omega\text{,l}}$ is for a *Source*/*Sink* component
$\text{c}\in\mathcal{C}^\text{srcSnk}$ given as

$$
\begin{aligned}
    &~~\text{\small F}^\text{O}_{\omega \text{,l}} = \big(\hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}} +
    \hat{X}^{\text{g}}_{\omega\text{,l}} + \hat{V}^{\text{g}}_{\omega\text{,l}} ~\big)~.
\end{aligned}
$$

Thus, operational cost as well as a cost and revenue for the associated generated or consumed commodity can be
considered with the parameters $\hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}}\in\mathbb{R}^{\geq0}$,
$\hat{X}^{\text{g}}_{\omega\text{,l}}\in\mathbb{R}^{\geq0}$ and
$\hat{V}^{\text{g}}_{\omega\text{,l}}\in\mathbb{R}^{\leq0}$ respectively.
