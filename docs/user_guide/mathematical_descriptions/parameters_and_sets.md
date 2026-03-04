# General Parameters and Sets

The energy system's basic framework is constituted by a number of parameters and sets that hold information
about the components, commodities and the spatial and temporal resolution with which the energy system is modeled.
The description of the parameters and sets is based on the description in
[Welder (2022)](https://publications.rwth-aachen.de/record/861215/files/861215.pdf)
and has been updated with the changes of the latest ETHOS.FINE version.

## Component Sets

The model of the energy system is based on several components with different attributes.
A component can be assigned to either the nodes or the edges of the model.
The components assigned to nodes are described by the set $\mathcal{C}^\text{node}$.
The components assigned to edges are described by the set $\mathcal{C}^\text{edge}$.
The set that contains all components $\mathcal{C}$ with which the energy system is modeled is given by

$$
\begin{aligned}
    \mathcal{C} &~=~& \mathcal{C}^\text{node} \cup \mathcal{C}^\text{edge}.
\end{aligned}
$$

## Commodity Sets

The set of all commodities (goods) that are considered in the energy system is given by the set $\mathcal{G}$.
The set $\mathcal{G}_\text{c} \subseteq \mathcal{G}$ contains all commodities associated to component $c \in \mathcal{C}$.

## Location Sets

The set of all locations that are considered in the energy system is given by $\mathcal{L}$.
A location is modeled as a *node* in the framework.

If component $c \in\mathcal{C}^\text{node}$, the set of locations at which the component is modeled is defined as

$$
\begin{aligned}
    \mathcal{L}_\text{c} = \left\{ \text{\small l} ~\vert~ \forall~\text{\small l}\in\mathcal{L}: \text{\small E}_\text{c,l}=1 \right\}.
\end{aligned}
$$

The parameter $\text{E}_\text{c,l}$ with $c \in\mathcal{C}^\text{node}$ equals 1 if a component is eligible at that location.

## Arc Sets

A connection between two locations is modeled as an *edge* called arc.
The set of all arcs $\mathcal{A}$ that are considered in the energy system is given by

$$
\begin{aligned}
    \mathcal{A} = \{ a = (l_\text{1}, l_\text{2} ) \in \mathcal{L} \times \mathcal{L} \}
\end{aligned}
$$

If component $c \in\mathcal{C}^\text{edge}$, the set of arcs at which the component is modeled is defined as

$$
\begin{aligned}
    \mathcal{A}_\text{c} = \big\{ \text{a} ~\vert~ \forall~\text{a}\in\mathcal{A}: \text{\small E}_\text{c,a}=1 \big\}.
\end{aligned}
$$

The parameter $\text{E}_\text{c,a}$ with $c \in\mathcal{C}^\text{edge}$ equals 1 if a component is eligible at that arc.

*Edge*-based components are modeled as being bidirectional by default. This implies that if a connection between
$\text{l}_\text{1}$ and $\text{l}_\text{2}$ is eligible, also the connection between $\text{l}_\text{2}$ and
$\text{l}_\text{1}$ is eligible.

## Time Sets

The parameter $\text{T}^\text{total}\in\mathbb{N}$, by default 8760 (i.e. 24 h/day $\cdot$ 365 day $=$ 8760 h),
specifies the total number of time steps with which the energy system is modeled. The corresponding index set
that encompasses all of these time steps is

$$
\begin{aligned}
    \mathcal{T}&~=~&\left\{0,\dots,\text{\small T}^\text{total}-1\right\}.
\end{aligned}
$$

The parameter $\text{T}^\text{hours}\in\mathbb{R}^{+}$ defines the number of hours per time step, by default 1 h.
The number of years $\text{T}^\text{years}$ which the energy system covers is determined by

$$
\begin{aligned}
    \text{T}^\text{years}&~=~&\frac{\text{\small T}^\text{total} \cdot \text{\small T}^\text{hours}}{8760~\text{\small h}}~\text{\small a}.
\end{aligned}
$$

Thus, the default value represents one year (1 a).

### Additional Time Sets for Time Series Aggregation

ETHOS.FINE provides support to use time series aggregation by using built-in methods integrating
the python package [tsam](https://github.com/FZJ-IEK3-VSA/tsam) into the code.
Therefore, additional sets and parameters are introduced:
The parameter $\text{T}^\text{per period}\in\mathbb{Z}^{+}$ specifies the number of time steps per period.
Thereby, $\text{T}^\text{total}$ must be a multiple of $\text{T}^\text{per period}$,
i.e. $\text{T}^\text{total} \pmod{\text{T}^\text{per\, period}} = 0$. If the energy system is investigated
with its full temporal resolution, $\text{T}^\text{per period}$ is set equal to $\text{T}^\text{total}$,
i.e. the energy system is investigated with only one period. If the energy system is modeled with typical
periods, $\text{T}^\text{per period}$ is set smaller or equal to $\text{T}^\text{total}$. The corresponding
set that contains all time steps within one period is given by

$$
\begin{aligned}
    \mathcal{T}^\text{per period}&~=~&\left\{0,\dots,\text{\small T}^\text{per period}-1\right\}.
\end{aligned}
$$

An additional time set is required to keep track of storage inventories. Storage inventories are defined right
at the beginning and at the end of the regular time steps. The set

$$
\begin{aligned}
    \mathcal{T}^\text{per period}_\text{inter}&~=~&\left\{0,\dots,\text{\small T}^\text{per period}\right\}
\end{aligned}
$$

gives these momentary points in time. The total number of periods $\text{P}^\text{total}$ results from the
total number of time steps and the time steps per period by

$$
\begin{aligned}
    \text{\small P}^\text{total} &~=~& \text{\small T}^\text{total}~/~\text{\small T}^\text{per period}.
\end{aligned}
$$

The corresponding set that encompasses all of these periods is

$$
\begin{aligned}
    \mathcal{P}^\text{total} &~=~& \left\{0,\dots,\text{\small P}^\text{total}-1\right\}.
\end{aligned}
$$

If typical periods are considered, each regular period *p* is assigned one of
$\text{P}^\text{typical}\in\mathbb{Z}^{+}$ typical periods $\text{p}^\text{typical}$. The set encompassing all
typical periods is

$$
\begin{aligned}
    \mathcal{P}^\text{typical} &~=~& \left\{0,\dots,\text{P}^\text{typical}-1\right\},~~\text{with}~\mathcal{P}^\text{typical}\subseteq\mathcal{P}^\text{total}.
\end{aligned}
$$

The function which maps the regular periods to a typical period is labeled

$$
\begin{aligned}
    map:\mathcal{P}^\text{total}\rightarrow\mathcal{P}^\text{typical}.
\end{aligned}
$$

## Investment Period Sets

ETHOS.FINE supports the modeling of multiple investment periods in a transformation pathway.
The set of all investment periods $\mathcal{IP}$ is given by the user. The default value is a single
investment period (single-year optimization). Multiple investment periods are used for perfect foresight or
stochastic optimization.
