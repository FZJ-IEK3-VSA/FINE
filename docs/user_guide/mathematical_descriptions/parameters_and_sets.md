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
    \mathcal{L}_\text{c} = \left\{ \text{l} ~\vert~ \forall~\text{l}\in\mathcal{L}: \text{E}_\text{c,l}=1 \right\}.
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
    \mathcal{A}_\text{c} = \big\{ \text{a} ~\vert~ \forall~\text{a}\in\mathcal{A}: \text{E}_\text{c,a}=1 \big\}.
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
    \mathcal{T}&~=~&\left\{0,\dots,\text{T}^\text{total}-1\right\}.
\end{aligned}
$$

The parameter $\text{T}^\text{hours}\in\mathbb{R}^{+}$ defines the number of hours per time step, by default 1 h.
The number of years $\text{T}^\text{years}$ which the energy system covers is determined by

$$
\begin{aligned}
    \text{T}^\text{years}&~=~&\frac{\text{T}^\text{total} \cdot \text{T}^\text{hours}}{8760~\text{h}}~\text{a}.
\end{aligned}
$$

Thus, the default value represents one year (1 a).

### Additional Time Sets for Time Series Aggregation

ETHOS.FINE provides support to use time series aggregation by using built-in methods integrating
the python package [tsam](https://github.com/FZJ-IEK3-VSA/tsam) into the code.
Therefore, additional sets and parameters are introduced:
The parameter $\text{T}^\text{per period}\in\mathbb{Z}^{+}$ specifies the number of time steps per period.
Thereby, $\text{T}^\text{total}$ must be a multiple of $\text{T}^\text{per period}$,
i.e. $\text{T}^\text{total} \pmod{\text{T}^\text{per period}} = 0$. If the energy system is investigated
with its full temporal resolution, $\text{T}^\text{per period}$ is set equal to $\text{T}^\text{total}$,
i.e. the energy system is investigated with only one period. If the energy system is modeled with typical
periods, $\text{T}^\text{per period}$ is set smaller or equal to $\text{T}^\text{total}$. The corresponding
set that contains all time steps within one period is given by

$$
\begin{aligned}
    \mathcal{T}^\text{per period}&~=~&\left\{0,\dots,\text{T}^\text{per period}-1\right\}.
\end{aligned}
$$

An additional time set is required to keep track of storage inventories. Storage inventories are defined right
at the beginning and at the end of the regular time steps. The set

$$
\begin{aligned}
    \mathcal{T}^\text{per period}_\text{inter}&~=~&\left\{0,\dots,\text{T}^\text{per period}\right\}
\end{aligned}
$$

gives these momentary points in time. Here, index 0 corresponds to the beginning of time step 0 and the index
$\text{T}^\text{per period}$ corresponds to the end of time step $\text{T}^\text{per period}-1$ within that
period, respectively. The total number of periods $\text{P}^\text{total}$ results from the
total number of time steps and the time steps per period by

$$
\begin{aligned}
    \text{P}^\text{total} &~=~& \text{T}^\text{total}~/~\text{T}^\text{per period}.
\end{aligned}
$$

The corresponding set that encompasses all of these periods is

$$
\begin{aligned}
    \mathcal{P}^\text{total} &~=~& \left\{0,\dots,\text{P}^\text{total}-1\right\}.
\end{aligned}
$$

Thus, $\vert\mathcal{P}^\text{total}\vert=1$ if the energy system is modeled with the full temporal resolution
and $\vert\mathcal{P}^\text{total}\vert\geq1$ if typical periods are considered.
In analogy to the set $\mathcal{T}^\text{per period}_{\text{inter}}$, the momentary points at the beginning
and at the end of a period are encompassed in the set

$$
\begin{aligned}
    \mathcal{P}^\text{total}_\text{inter} &~=~& \left\{0,\dots,\text{P}^\text{total}\right\}.
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

The frequency *f* with which each period occurs during the total investigated time is defined as

$$
\begin{aligned}
    f:
    \begin{cases}
        \left\{0\right\} \rightarrow \left\{1\right\} &\text{, with full temporal resolution, or}\\
        \mathcal{P}^\text{typical}\rightarrow\mathbb{Z}^{+} &\text{, with time series aggregation.}
    \end{cases}
\end{aligned}
$$

In the following, all basic operation variables are declared for all periods or typical periods, depending on
whether time series aggregation is considered, and all time steps within these periods. The cross-product of
these sets is given by

$$
\begin{aligned}
\mathcal{P}\times\mathcal{T} =
\begin{cases}
    \mathcal{P}^\text{total}\hspace{0.25cm}\times\mathcal{T}^\text{per period} &\text{, for a full temporal resolution, or}\\
    \mathcal{P}^\text{typical}\times\mathcal{T}^\text{per period} &\text{, with time series aggregation.}
\end{cases}
\end{aligned}
$$

Similarly, the cross-product for keeping track of storage inventories is defined by

$$
\begin{aligned}
\mathcal{P}\times\mathcal{T}_\text{inter} =
\begin{cases}
    \mathcal{P}^\text{total}\hspace{0.25cm}\times\mathcal{T}^\text{per period}_\text{inter} &\text{, for a full temporal resolution, and}\\
    \mathcal{P}^\text{typical}\times\mathcal{T}^\text{per period}_\text{inter} &\text{, with time series aggregation.}
\end{cases}
\end{aligned}
$$

## General Parameters

The general parameters are summarized in the following table:

| **Parameter** | **Domain** | **Description** |
|---|---|---|
| $\text{T}^\text{total}$ | $\mathbb{N}$ | total number of time steps |
| $\text{T}^\text{hours}$ | $\mathbb{R}^{+}$ | number of hours per time step |
| $\text{T}^\text{per period}$ | $\mathbb{Z}^{+}$ | number of time steps per period |

Other parameters, e.g., eligibilities of components, are described in [Basic Component Model](basic_component.md).

## Compound Index Sets

As several variables and parameters depend on a temporal and operational level, two more compound index sets
are added. To describe the temporal position, the compound index set $\Theta$ is introduced:

$$
\begin{aligned}
\Theta = \left\{(p,t) ~\vert~ p \in \mathcal{P}, t \in \mathcal{T} \right\}
\end{aligned}
$$

The operation of the different modeled components can be described by several modes of operation.
Those are summarized in the set $\mathcal{M}$. All eligible modes for component $c$ are described by
$M_\text{c}=1$. For each component the set of modes of operation is described by

$$
\begin{aligned}
\mathcal{M}_\text{c}&~=~&\left\{ \text{m} ~\vert~ \forall~\text{m}\in\mathcal{M}: \text{M}_\text{c}=1 \right\}.
\end{aligned}
$$

The compound index set $\Omega$ which describes the modes of operation of a certain component is given by:

$$
\begin{aligned}
\Omega = \left\{(c,m) ~\vert~ c \in \mathcal{C}, m \in \mathcal{M}_\text{c} \right\}
\end{aligned}
$$

