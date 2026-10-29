# Transmission Component Model Extension

Components which transmit commodities between locations in $\mathcal{L}$ and along arcs
$\text{a} \in \mathcal{A}$ are modeled as bidirectional *Transmission* components by default. Examples of
*Transmission* components are electric lines or bidirectional gas pipelines. The *Transmission* component
model extends the *Basic* component model. In the following, the set of all *Transmission* components is
labeled $\mathcal{C}^\text{trans}\subset\mathcal{C}^\text{edge}$.

## Specification of Operation Variables and Associated Commodities

A *Transmission* component $\text{c}\in\mathcal{C}^\text{trans}$ only has one type of basic operation
variables. It is associated with one commodity $\mathcal{G}^\text{c}=\{\text{g}\}$, with
$\text{g}\in\mathcal{G}$, which is the commodity that the component transmits. If a capacity is defined for
this component, it is related to this commodity. For example, the capacity of an electric line is related to
the nominal electric power it can transmit, e.g. in MW$_\text{el}$.

## Specification of Additional Constraints

A *Transmission* component can be operated bidirectionally. This means that the flow from
$\text{l}_\text{1}\in\mathcal{L}$ to $\text{l}_\text{2}\in\mathcal{L}$, which is described as arc
$\text{a}$, has to use the same route and infrastructure as a flow from $\text{l}_\text{2}$ to
$\text{l}_\text{1}$, which is described as arc $\hat{\text{a}}$. To enforce this behavior, the constraint

$$
\begin{aligned}
    k_\text{c,a} = k_{\text{c,}\hat{\text{a}}}
\end{aligned}
$$

is stated for all $\text{c}\in\mathcal{C}^\text{trans}$ and all
$\text{a,}\hat{\text{a}}\in\mathcal{A}_\text{c}$.
Furthermore, the equation of the maximum operation of a component is supplemented with the equation

$$
\begin{aligned}
    o_{\omega\text{,a,}\theta}+o_{\omega\text{,}\hat{\text{a}}\text{,}\theta} \leq \text{T}^\text{hours} \cdot k_\text{c,a}
\end{aligned}
$$

for all $\text{c}\in\mathcal{C}^\text{trans}$ and all $\text{a,} \hat{\text{a}} \in \mathcal{A}_\text{c}$.
This set of equations increases the tendency that, for basic optimization solutions, one of the commodity
flows $o_{\omega\text{,a,}\theta}$ or $o_{\omega\text{,}\hat{\text{a}}\text{,}\theta}$ is set to zero.

## Specification of Commodity Balance Contributions

Contributions to the commodity balance equations are modeled for $\text{c}\in\mathcal{C}^\text{trans}$, for
$\text{g}\in\mathcal{G}^\text{c}$, for all $\text{l}\in\mathcal{L}$, and for all $\theta \in \Theta$.

To describe the commodity balance equations, we define two sets of arcs:
The set $\text{a}^\text{in} \in \mathcal{A}^\text{c}$ equals $(\text{l,l}_\text{out})$
and includes all eligible connections for commodity flows from connected locations to location l.
The set $\text{a}^\text{out} \in \mathcal{A}^\text{c}$ equals $(\text{l}_\text{in}\text{,l})$
and includes all eligible connections for commodity flows from location l to connected locations.

$$
\begin{aligned}
    C_{\text{c,g,l,}\theta} ~=~
    \sum\limits_{\substack{\text{a}^\text{in}~\in~\mathcal{A}^\text{c}}}
    (1-\eta_{\text{a}^\text{in}} \cdot \text{d}_{\text{a}^\text{in}}) \cdot o_{\omega \text{,a}^\text{in} \text{,} \theta} - \sum\limits_{\substack{\text{a}^\text{out} ~\in~\mathcal{A}^\text{c}}} o_{\omega \text{,a}^\text{out} \text{,} \theta}~.
\end{aligned}
$$

Here, $\eta_\text{a}^\text{in}$ is a linear loss factor per length and capacity.
$\text{d}_\text{a}^\text{in}$ is the length between the two connected locations. The term thus represents
incoming and outgoing flows of a commodity g at the location l at period p and time step t.

## Specification of Objective Function Contributions

The parameters $\text{F}^\text{K}_\text{c,a}$ and $\text{F}^\text{B}_\text{c,a}$ in equations for the
objective function contribution are set equal to $1/2 \cdot \text{d}_\text{a}$ for *Transmission*
components. The factor $1/2$ compensates that each connection is taken into account twice in the objective
function. The length d of the connection is included so that the capital and operational cost factors can be
given as not only capacity but also length related.

The cost factor $\text{F}^\text{O}_{\omega\text{,a}}$ is given as

$$
\begin{aligned}
    &~~\text{F}^\text{O}_{\omega\text{,a}} ~=~ \hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}}~.
\end{aligned}
$$

with $\hat{X}^{\text{opex}_\text{O}}_{\omega\text{,a}}\in\mathbb{R}^{\geq0}$ which describes the costs per operation of component c.
