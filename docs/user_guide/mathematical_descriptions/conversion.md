# Conversion Component Model Extension

A component which converts one set of commodities into another set of commodities, as for example a power
plant is modeled to convert natural gas into electricity and carbon dioxide, is modeled in FINE as a
so-called *Conversion* component. The *Conversion* component model thereby extends the *Basic* component
model. In the following, the set of all *Conversion* components is labeled
$\mathcal{C}^\text{conv}\subset\mathcal{C}^\text{node}$.

## Specification of Operation Variables and Associated Commodities

A *Conversion* component $\text{c}\in\mathcal{C}^\text{conv}$ only has one type of basic operation variables.
It can however be associated with multiple commodities, as it converts commodities into each other. The nominal
capacity of a *Conversion* component is related to one of these commodities labeled $\text{g}^\text{nominal}$.
For example, the capacity of an electrolyzer can be related to either the consumed electricity, or the lower
heating value (LHV) of the generated hydrogen.

## Specification of Commodity Balance Contributions

Inherently, a *Conversion* component contributes to the balance equations of multiple commodities. These
contributions are modeled for $\text{c}\in\mathcal{C}^\text{conv}$, for all $\text{g}\in\mathcal{G}^\text{c}$,
for all $\text{l}\in\mathcal{L}^\text{c}$ and for all $\theta \in \Theta$ as

$$
\begin{aligned}
    &C_{\text{c,g,l,}\theta} ~=~ \text{cf}_\text{c,g} \cdot o_{\omega\text{,l,}\theta}.
\end{aligned}
$$

The conversion factor $\text{cf}_\text{c,g}\in\mathbb{R}$ is by convention negative if a commodity is consumed
and positive if a commodity is generated. The nominal conversion factor
$\big|\text{cf}_{\text{c,g}^{\text{nominal}}}\big|$ is set to 1.

## Specification of Objective Function Contributions

The cost factor $\text{F}^\text{O}_{\omega\text{,l}}$ is for a *Conversion* component
$\text{c}\in\mathcal{C}^\text{conv}$ given as

$$
\begin{aligned}
    &~~\text{F}^\text{O}_{\omega \text{,l}} ~=~ \hat{X}^{\text{opex}_\text{O}}_{\omega\text{,l}}~.
\end{aligned}
$$

---

## Part-Load Extension (ConversionPartLoad)

In many real-world applications, the conversion efficiency of a component is not constant but
depends on the **operation level** (part-load ratio). For example, an electrolyzer or gas turbine
may have a different efficiency at 30% load than at full load. The `ConversionPartLoad` component
extends the standard *Conversion* model to capture this nonlinear behavior using a **piecewise
linear approximation**.

### Piecewise Linear Approximation

The nonlinear efficiency curve $\eta(x)$, where $x \in [0, 1]$ is the operation level (fraction
of nominal capacity), is approximated by a piecewise linear function with $N$ segments and
$N+1$ breakpoints:

$$
(x_0, y_0),\; (x_1, y_1),\; \ldots,\; (x_N, y_N)
$$

where $x_i$ are the operation levels and $y_i = \eta(x_i)$ are the corresponding conversion
factors at the breakpoints. The breakpoints are determined automatically using the
[PWLF](https://github.com/cjekel/piecewise_linear_fit_py) library to minimize the
approximation error.

### SOS2 Formulation

The piecewise linear function is modeled in the optimization using **Special Ordered Sets of
type 2 (SOS2)** constraints. For each component $\text{c}$, location $\text{l}$, and time step
$\theta$, the following variables are introduced:

- **Point variables** $\lambda_{i,\text{l},\theta} \geq 0$ for $i = 0, \ldots, N$: weights
  for each breakpoint.
- **Segment continuous variables** $s_{j,\text{l},\theta} \geq 0$ for $j = 0, \ldots, N-1$:
  contribution of each segment.
- **Segment binary variables** $z_{j,\text{l},\theta} \in \{0, 1\}$ for $j = 0, \ldots, N-1$:
  indicates which segment is active.

### Constraints

**SOS1 on segment binaries** — Exactly one segment is active at each time step:

$$
\sum_{j=0}^{N-1} z_{j,\text{l},\theta} = 1
$$

**Big-M linking** — Continuous segment variables are zero when the segment is inactive:

$$
s_{j,\text{l},\theta} \leq z_{j,\text{l},\theta} \cdot M \quad \forall\, j
$$

**Segment capacity** — Segment variables sum to the installed capacity:

$$
\sum_{j=0}^{N-1} s_{j,\text{l},\theta} = \Delta t \cdot \text{cap}_{\text{c,l}}
$$

**Point capacity** — Point variables sum to the installed capacity:

$$
\sum_{i=0}^{N} \lambda_{i,\text{l},\theta} = \Delta t \cdot \text{cap}_{\text{c,l}}
$$

**SOS2 adjacency** — At most two consecutive point variables can be non-zero:

$$
\lambda_{0,\text{l},\theta} \leq s_{0,\text{l},\theta}, \quad
\lambda_{N,\text{l},\theta} \leq s_{N-1,\text{l},\theta}, \quad
\lambda_{i,\text{l},\theta} \leq s_{i-1,\text{l},\theta} + s_{i,\text{l},\theta} \;\; \forall\, 1 \leq i \leq N-1
$$

**Operation output** — The operation variable is linked to the piecewise linear function:

$$
o_{\text{c,l},\theta} = \sum_{i=0}^{N} \lambda_{i,\text{l},\theta} \cdot x_i
$$

### Commodity Balance Contribution

The commodity balance contribution under part-load is:

$$
C_{\text{c,g,l,}\theta} = \sum_{i=0}^{N} \lambda_{i,\text{l},\theta} \cdot x_i \cdot y_{i,\text{g}}
$$

where $y_{i,\text{g}}$ is the conversion factor for commodity $\text{g}$ at breakpoint $i$.
This replaces the constant factor $\text{cf}_\text{c,g} \cdot o_{\omega\text{,l,}\theta}$ from
the standard *Conversion* model.

!!! tip "Choosing the number of segments"
    The number of segments $N$ controls the trade-off between approximation accuracy and
    computational cost. Values between 3 and 7 are recommended. The piecewise linearization
    should be visually inspected to verify sufficient accuracy. See
    [Example 11 (Partload)](../../examples/11_Partload/11_Partload_Example.ipynb) for guidance.
