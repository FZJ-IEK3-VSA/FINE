# Basic Component Model

The *Basic* component model comprises sets of variables, constraints, inter-component constraint-contributions
and objective function contributions that apply to all components specified in $\mathcal{C}$.
In this context, the variables and constraints can be divided into either being time-independent or time-dependent.

## Component Parameters

### Dimensioning of Components

For each component $c \in C$, capacity variables can be introduced with the parameter $K_\text{c} \in \{0,1\}$:

$$
\begin{aligned}
    K_\text{c}=
    \begin{cases}
        1 &\text{\small , if the component is modeled with a physical capacity, or}\\
        0 &\text{\small , if the component is modeled without a physical capacity.}\\
    \end{cases}
\end{aligned}
$$

A component which is modeled with physical capacity is for example a gas power plant while an electricity
demand does not require one.

The following parameters refer to all components $\text{c}\in\mathcal{C}$ with $K_\text{c}=1$:

| **Parameter** | **Domain** | **Description** |
|---|---|---|
| $K^\text{min}_\text{c,l}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}$ | minimum capacity of component c at location l |
| $K^\text{max}_\text{c,l}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}$ | maximum capacity of component c at location l |
| $K^\text{fix}_\text{c,l}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}$ | fixed capacity of component c at location l |
| $K^\text{unit}_\text{c}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}$ | capacity per plant unit of component c |
| $B_\text{c}$ | $\left\{0,1\right\}$ with $c \in \mathcal{C}$ | introduces decision variable to state if a capacity is built or not |
| $B^\text{fix}_\text{c,l}$ | $\left\{0,1\right\}$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}$ | fixes decision variable to state if a component c is built or not at location l |
| $M_\text{c}$ | $\mathbb{R}^+$ with $c \in \mathcal{C}$ | required auxiliary parameter if $K^\text{bin}_\text{c} = 1$ |
| $\text{E}_\text{c,l}$ | $\left\{0,1\right\}$ with $c \in \mathcal{C}^\text{node}, l \in \mathcal{L}_\text{c}$ | eligibility of node component c at location l |
| $\text{E}_\text{c,a}$ | $\left\{0,1\right\}$ with $c \in \mathcal{C}^\text{edge}, a \in \mathcal{A}_\text{c}$ | eligibility of edge component c at arc a |

### Operation of Components

| **Parameter** | **Domain** | **Description** |
|---|---|---|
| $R^\text{min}_{\text{c,l,}\theta}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}, \theta \in \Theta$ | minimum operation rate of component c at location l and time step t |
| $R^\text{max}_{\text{c,l,}\theta}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}, \theta \in \Theta$ | maximum operation rate of component c at location l and time step t |
| $R^\text{fix}_{\text{c,l,}\theta}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}, \theta \in \Theta$ | fixed operation rate of component c at location l and time step t |

### Cost Contribution of Components

| **Parameter** | **Domain** | **Description** |
|---|---|---|
| $\hat{X}^{\text{capex}_\text{K}}_\text{c,l}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}$ | expenditures per capacity |
| $\hat{X}^{\text{opex}_\text{K}}_\text{c,l}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}$ | operational expenditures per capacity |
| $\hat{X}^{\text{capex}_\text{B}}_\text{c,l}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}$ | expenditures if capacity is built |
| $\hat{X}^{\text{opex}_\text{B}}_\text{c,l}$ | $\mathbb{R}_0^+$ with $c \in \mathcal{C}, l \in \mathcal{L}_\text{c}$ | operational expenditures per capacity if capacity is built |
| $T^\text{EL}_\text{c}$ | $\mathbb{Z}_0^+$ with $c \in \mathcal{C}$ | economic lifetime of component c |

## Time-Independent Variables and Constraints

A capacity variable $k_\text{c,l}\in\mathbb{R}^{\geq0}$ is declared for all locations
$\text{l}\in\mathcal{L}_\text{c}$ in the energy system at which the component can appear. Implicitly, this
capacity is modeled either as a continuous or discrete value by

$$
\begin{aligned}
k_\text{c,l} =
    \text{\small K}^\text{unit}_\text{c,l} \cdot n_\text{c,l}
\end{aligned}
$$

with $n_\text{c,l} \in \mathbb{R}_0^+$ if the capacity is modeled as a continuous value, and with
$n_\text{c,l} \in \mathbb{N}_0^+$ if the capacity is modeled as discrete value.

Furthermore, the component can be modeled together with a binary design decision variable
$b_\text{c,l}\in\left\{0,1\right\}$ if its boolean parameter $B_\text{c}$ is set to true (=1).
The consideration of the binary decision variables is enforced in the model for all $b_\text{c,l}$ by the constraint

$$
\begin{aligned}
 \text{\small M}_\text{c} \cdot b_\text{c,l} ~\geq~ k_\text{c,l}~,
\end{aligned}
$$

where $\text{M}_\text{c}\in\mathbb{R}_0^{+}$. The constraint enforces that $b_\text{c,l} = 1$ if
$k_\text{c,l} > 1$.

Lower and upper boundaries can be specified for the capacity variables. Lower bounds are enforced by

$$
\begin{aligned}
&&&k_\text{c,l} \geq
\begin{cases}
    K^\text{min}_\text{c,l}\cdot b_\text{c,l} &,~\text{\small if}~ B_\text{c}=1,\\
    K^\text{min}_\text{c,l} &,~\text{\small if}~ B_\text{c}=0.
\end{cases}
\end{aligned}
$$

Upper bounds are enforced by

$$
\begin{aligned}
k_\text{c,l}  ~\leq~
\text{K}^\text{max}_\text{c,l}~~.
\end{aligned}
$$

Fixed values can be individually specified for the capacity and binary decision variables by

$$
\begin{aligned}
k_\text{c,l}  &~=~~ && K^\text{fix}_\text{c,l}~~\text{\small and} \\
k^\text{bin}_\text{c,l}  &~=~~ && K^\text{bin,fix}_\text{c,l}~~.
\end{aligned}
$$

## Basic Time-Dependent Variables and Constraints

Operational variables $o_{\omega \text{,l,} \theta}\in\mathbb{R}^{\geq0}$ are declared for all operation
types of a component $\omega \in \Omega$, for all locations $\text{l}\in\mathcal{L}^\text{c}$ and for all
periods and time steps $\theta \in \Theta$.

Each operation variable of a component modeled with a physical capacity ($K_\text{c} = 1$) is limited in
one of four ways:

1. Limited by capacity and a time-independent factor $\text{a}_{\omega}\in\mathbb{R}^{\geq0}$ (default: 1):

$$
\begin{aligned}
    o_{\omega \text{, l,} \theta}  ~\leq~ \text{\small T}^\text{hours} \cdot \text{\small a}_{\omega} \cdot k_\text{c,l}
\end{aligned}
$$

2. Fixed to a relative operation rate $\text{R}^\text{fix}_{\text{c,l,} \theta}$:

$$
\begin{aligned}
    o_{\omega \text{,l,} \theta}  ~=~ \text{\small T}^\text{hours} \cdot \text{\small R}^\text{fix}_{\text{c,l,} \theta} \cdot k_\text{c,l}
\end{aligned}
$$

3. Limited by a maximum relative operation rate $\text{R}^\text{max}_{\text{c,l,} \theta}$:

$$
\begin{aligned}
    o_{\omega \text{,l,} \theta}  ~\leq~ \text{\small T}^\text{hours} \cdot \text{\small R}^\text{max}_{\text{c,l,} \theta} \cdot k_\text{c,l}
\end{aligned}
$$

4. Bounded below by a minimum relative operation rate $\text{R}^\text{min}_{\text{c,l,} \theta}$:

$$
\begin{aligned}
    o_{\omega \text{,l,} \theta}  ~\geq~ \text{\small T}^\text{hours} \cdot \text{\small R}^\text{min}_{\text{c,l,} \theta} \cdot k_\text{c,l}
\end{aligned}
$$

## Basic Inter-Component Constraint Contributions

Inter-component constraint contributions are defined to model constraints which affect multiple components.
The contribution of a component to a commodity balance equation is labeled $C_{\text{c,g,l,}\theta}$ and has
to be defined for each component which is added to the model.

Moreover, two or more components can compete for a limited capacity potential in an energy system. Components
which share a potential in FINE are provided with an identifier. The share of a component on the maximum
potential at all locations $\text{l} \in\mathcal{L}^\text{c}$ is defined by
$k_\text{c,l}/\text{\small k}^\text{max}_\text{c,l}$.

## Basic Objective Function Contribution

The objective function in the framework is defined as the net present value $NPV$ of all components
$\text{c} \in \mathcal{C}$ and is minimized during optimization. Objective function contributions
$NPV_\text{c}$ [costUnit/a] are specified for each component individually by

$$
\begin{aligned}
    NPV_\text{c} =& \sum\limits_{\text{l}~\in~\mathcal{L}_\text{c}}\hspace{-3pt}
        \left( NPV^\text{K}_\text{c,l} ~+~ NPV^\text{B}_\text{c,l} ~+~ NPV^\text{O}_\text{c,l} \right)
\end{aligned}
$$

The capacity related total annual cost contributions are determined by

$$
\begin{aligned}
    &NPV^\text{K}_\text{c,l} = \text{\small F}^\text{K}_\text{c,l} \cdot \left(\frac{\hat{X}^{\text{capex}_\text{K}}_\text{c,l}}{\text{\small CCF}_\text{c,l}} + \hat{X}^{\text{opex}_\text{K}}_\text{c,l}\right) \cdot k_\text{c,l}
\end{aligned}
$$

The total annual cost contributions related to the binary decision variables are determined by

$$
\begin{aligned}
    &NPV^\text{B}_\text{c,l} = \text{\small F}^\text{B}_\text{c,l} \cdot \left(\frac{\hat{X}^{\text{capex}_\text{B}}_\text{c,l}}{\text{\small CCF}_\text{c,l}} + \hat{X}^{\text{opex}_\text{B}}_\text{c,l}\right) \cdot b_\text{c,l}
\end{aligned}
$$

The total annual cost contributions related to the operation are determined by the individual component
model extensions.
