# Storage Component Model Extension

Components which store a commodity are modeled in FINE as so-called *Storage* components. Examples for
*Storage* components are batteries or underground gas storage facilities. The *Storage* component model
thereby extends the *Basic* component model. In addition to the *Basic* component model functionalities, the
model requires sets of variables and constraints which can model storage inventories. This includes a set of
variables and constraints enabling to transfer the information on storage inventories between typical periods.
This storage formulation makes computationally efficient seasonal storage investigations possible.
The *Storage* component model formulation extends the formulations given by
[Welder et al. (2018)](https://doi.org/10.1016/j.energy.2018.05.059) and
[Kotzur et al. (2018)](https://doi.org/10.1016/j.apenergy.2018.01.023).
In the following, the set of all *Storage* components is labeled
$\mathcal{C}^\text{stor}\subset\mathcal{C}^\text{node}$.

## Specification of Basic Operational Parameters and Associated Commodities

A *Storage* component $\text{c}\in\mathcal{C}^\text{stor}$ has two types of basic operation modes
$\mathcal{M}^\text{c}=\{\text{+,-}\}$. It is associated with one commodity
$\mathcal{G}^\text{c}=\{\text{g}\}$, with $\text{g}\in\mathcal{G}$, which is stored by the component.
`+` indicates the charging operation, `-` indicates the discharging operation.
If a capacity is defined for this component, it is related to this commodity. For example, the capacity of a
battery is related to the nominal electric energy it can store. The rate at which a storage can be charged/
discharged is generally limited. The parameter $\text{j}_\omega$ is in this context used to define the
relative charging/ discharging rate per hour. For example, if it takes six hours to fully charge a storage,
with respect to its nominal capacity, $\text{j}_\text{c,+}$ is equal to $1/6$.

## Specification of Additional Variables and Constraints

An additional set of variables is required to track how much commodity remains in the *Storage* component in
between time steps. These variables are referred to as $s$ (state of charge) variables.

The variable $s_\text{c,l,p,t}\in\mathbb{R}^{\geq0}$ defines for all $\text{c}\in\mathcal{C}^\text{stor}$
and for all $\text{l}\in\mathcal{L}^\text{c}$ the state of charge within a period p at the beginning of time
step t, with $(\text{p, t})\in\mathcal{P}\times\mathcal{T}_\text{inter}$.

If typical periods are considered, an additional set of state of charge variables is declared that accounts
for the state of charge in between periods. In this case,
$s^\text{inter}_\text{c,l,p}\in\mathbb{R}^{\geq0}$ describes the actual, real state of charge in between
periods and is defined for all $\text{c}\in\mathcal{C}^\text{stor}$, for all
$\text{l}\in\mathcal{L}^\text{c}$ and for all $p\in\mathcal{P}^\text{total}_\text{inter}$.
$s_\text{c,l,p,t}$, now in $\mathbb{R}$, functions as a virtual state of charge. The superposition of the
two variables gives, with the consideration of a self-discharge factor, the real state of charge at period p
at the beginning of time step t.

## Linkage of $s$ Variables Across the Investigated Timeframe

The state of charge within a period p at the beginning of time step $\text{t}+1$ results from the state of
charge at the beginning of time step t and the charge and discharge rate during time step t within that period:

$$
\begin{aligned}
    & s_\text{c,l,p,t+1} &~=~& s_\text{c,l,p,t} \cdot \left(1-\text{Q}^{\circ}_\text{c}\right)^{\text{T}^\text{hours}} \nonumber \\
    & && + o^\text{+}_\text{c,l,p,t}\cdot\text{Q}^{+}_\text{c} ~-~ o^\text{-}_\text{c,l,p,t}/\text{Q}^{-}_\text{c}
\end{aligned}
$$

for all $\text{c}\in\mathcal{C}^\text{stor}$, for all $\text{l}\in\mathcal{L}^\text{c}$ and for all
$(\text{p, t})\in\mathcal{P}\times\mathcal{T}$. The parameters
$\text{Q}^{\circ}_\text{c},\text{Q}^{+}_\text{c},\text{Q}^{-}_\text{c}\in(0,1]$
describe the self-discharge during one hour and the charging and discharging efficiency respectively.

If typical periods are considered, the virtual state of charge at the beginning of each typical period
$\text{p}\in\mathcal{P}^\text{typical}$ has to satisfy the condition

$$
\begin{aligned}
    & s^\text{inter}_\text{c,l,p} + s_\text{c,l,p,0} = s^\text{inter}_\text{c,l,p} ~~\rightarrow~~ s_\text{c,l,p,0}=0
\end{aligned}
$$

for all $\text{c}\in\mathcal{C}^\text{stor}$ and for all $\text{l}\in\mathcal{L}^\text{c}$. The state of
charge at the beginning of period $\text{p}+1$ results from the superposition of the state of charge at the
beginning of period p and the state of charge at the end of the period by

$$
\begin{aligned}
    &s^\text{inter}_\text{c,l,p+1} &~=~& && s^\text{inter}_\text{c,l,p} \cdot \left(1-\text{Q}^{\circ}_\text{c}\right)^{\text{T}^\text{per period}~\cdot~\text{T}^\text{hours}} \nonumber \\
    & &&+~ && s_\text{c,l,$map(\text{p})$,T$^\text{per period}$}
\end{aligned}
$$

for all $\text{c}\in\mathcal{C}^\text{stor}$, for all $\text{l}\in\mathcal{L}^\text{c}$ and for all
$p\in\mathcal{P}^\text{total}$. The function $map$ maps a period to a typical period.

The *Storage* component model imposes a constraint which sets the state of charge at the beginning and the
end of the investigated timeframe equal to each other. The energy system is thus modeled as being
self-repetitive. This constraint is given as

$$
\begin{aligned}
    & s_\text{c,l,0,0} &=& s_\text{c,l,0,T$^\text{total}$}~,~&&\text{with full temporal resolution}, ~\text{or} \nonumber \\
    & s^\text{inter}_\text{c,l,0} &=& s^\text{inter}_\text{c,l,P$^\text{total}$}~,~&&\text{with time series aggregation,}
\end{aligned}
$$

for all $\text{c}\in\mathcal{C}^\text{stor}$ and for all $\text{l}\in\mathcal{L}^\text{c}$.

## Consideration of Operating Limits of $s$ Variables

It must be ensured that the state of charge is within the operating limits of the installed storage capacity
for all $\text{c}\in\mathcal{C}^\text{stor}$ if they are modeled with a physical capacity. Here, three
modeling approaches have to be distinguished from one another.

The first modeling approach applies to an energy system which is modeled with a full temporal resolution,
i.e. no typical periods are considered. In this case, the upper and lower operating limits are given by

$$
\begin{aligned}
    & \text{S}^\text{min}_\text{c} \cdot k_\text{c,l} ~\leq~ s_\text{c,l,0,t} ~\leq~ \text{S}^\text{max}_\text{c} \cdot k_\text{c,l}
\end{aligned}
$$

for all $\text{l}\in\mathcal{L}^\text{c}$ and for all $\text{t}\in\mathcal{T}^\text{total}$. Here, the
parameters $0\leq\text{S}^\text{min}_\text{c}<\text{S}^\text{max}_\text{c}\leq1$ model relative lower and
upper limits on the state of charge.

The second modeling approach applies when typical periods are considered, and the *Storage* component should
be modeled with *precise* operating boundaries
($\text{doPreciseTSAmodeling}_\text{c}=\text{True}$). In this case, the lower and upper operating limits
are given by

$$
\begin{aligned}
    & \text{S}^\text{min}_\text{c} \cdot k_\text{c,l} \leq s^\text{sup}_\text{c,l,p,t} \leq \text{S}^\text{max}_\text{c} \cdot k_\text{c,l},~~\text{with} \nonumber \\
    & s^\text{sup}_\text{c,l,p,t}=s^\text{inter}_\text{c,l,p} \cdot \big(1-\text{Q}^{\circ}_\text{c}\big)^{\text{t}~\cdot~\text{T}^\text{hours}} + s_\text{c,l,$map(\text{p})$,t}~,
\end{aligned}
$$

for all $\text{l}\in\mathcal{L}^\text{c}$ and for all $p\in\mathcal{P}^\text{total}$ and for all
$t\in\mathcal{T}^\text{per period}$.

The third modeling approach applies when typical periods are considered, and the *Storage* component should
be modeled with *simplified* operating boundaries
($\text{doPreciseTSAmodeling}_\text{c}=\text{False}$). This approach reduces the computational load in
comparison to the second approach even further and is a good estimate when the self-discharge of the
*Storage* component is small. In this case, the lower and upper operating limits are given by

$$
\begin{aligned}
    & \text{S}^\text{min}_\text{c} \cdot k_\text{c,l} \leq \underline{s}^\text{sup}_\text{c,l,p,t} ~~\wedge~~ \overline{s}^\text{sup}_\text{c,l,p,t} \leq \text{S}^\text{max}_\text{c} \cdot k_\text{c,l}, \nonumber \\
    &~~\text{with}~~ \underline{s}^\text{sup}_\text{c,l,p,t} = s_\text{c,l,p}^\text{inter} \cdot \big(1-\text{Q}^{\circ}_\text{c}\big)^{\text{T}^\text{per period}\cdot~\text{T}^\text{hours}} + s_\text{c,l,$map(\text{p})$}^\text{min} \nonumber \\
    &~~\text{and}~~ \overline{s}^\text{sup}_\text{c,l,p,t} = s_\text{c,l,p}^\text{inter} + S_\text{c,l,$map(\text{p})$}^\text{max}~,
\end{aligned}
$$

for all $\text{l}\in\mathcal{L}^\text{c}$ and for all $p\in\mathcal{P}^\text{total}$. The two variables
$s^\text{min}_{\text{c,l},map(\text{p})}\in\mathbb{R}^{\leq0}$ and
$s^\text{max}_{\text{c,l},map(\text{p})}\in\mathbb{R}^{\geq0}$ are auxiliary variables that describe the
virtual minimum and maximum state of charge within the typical period $\bar{\text{p}}$ obtained by
$map$(p). They are bounded from above/below by all $s_{\text{c,l,}\bar{\text{p}}\text{,t}}$ of the
respective component c within the typical period $\bar{\text{p}}$ by

$$
\begin{aligned}
    & s_\text{c,l,$\bar{\text{p}}$}^\text{min} \leq s_\text{c,l,$\bar{\text{p}}$,t} \leq s_\text{c,l,$\bar{\text{p}}$}^\text{max}
\end{aligned}
$$

for all $\text{c}\in\mathcal{C}^\text{stor}$, for all $\text{l}\in\mathcal{L}^\text{c}$ and for all
$(\bar{\text{p}}, t)\in\mathcal{P}\times\mathcal{T}$. The given equations over- and underestimate the
minimum and maximum real $s$ and therefore always give feasible operating limits.

## Additional Constraints

A cyclic lifetime $\text{T}^\text{CL}_\text{c}\in\mathbb{Z}^{>0}$ can be considered for a storage
component $\text{c}\in\mathcal{C}^\text{stor}$. The cyclic lifetime limits the number of full cycle
equivalents for all $\text{l}\in\mathcal{L}^\text{c}$ by

$$
\begin{aligned}
    & o^\text{+}_\text{c,l,annual} \leq
    \left(\text{S}^\text{max}_\text{c}-\text{S}^\text{min}_\text{c}\right) \cdot k_\text{c,l} \cdot \frac{\text{T}^\text{CL}_\text{c}}{\text{T}^\text{EL}_\text{c,l}}, \nonumber \\
    &\text{with}~~ o^\text{+}_\text{c,l,annual} = \sum\limits_{(\text{p,t})~\in~\mathcal{P}\times\mathcal{T}} o_\text{c,l,p,t}^\text{+} \cdot f\left(\text{p}\right) / \text{T}^\text{years}~,
\end{aligned}
$$

where $f$ is the frequency of the period p within the investigated timeframe. This means that the commodity
amount with which the storage is charged during its economic lifetime divided by the usable storage capacity
(full cycle equivalents) has to be smaller than the cyclic lifetime, e.g. 10,000 cycles. It has to be noted
that a storage can also be associated with a calendric lifetime. This calendric lifetime can be implicitly
enforced in *FINE* by setting the economic lifetime to a value smaller than this calendric lifetime.

## Specification of Commodity Balance Contributions

Contributions to the commodity balance equations are modeled for $\text{c}\in\mathcal{C}^\text{stor}$, for
$\text{g}\in\mathcal{G}^\text{c}$, for all $\text{l}\in\mathcal{L}^\text{c}$ and for all $\theta \in \Theta$ as

$$
\begin{aligned}
    &C_{\text{c,g,l,}\theta} ~=~ o_\text{c,l,$\theta$}^\text{-}-o_\text{c,l,$\theta$}^\text{+}~.
\end{aligned}
$$

The term represents the amount of commodity g which is at location l, period p and time step t injected
($C_{\text{c,g,l,}\theta}<0$) or withdrawn ($C_{\text{c,g,l,}\theta}\geq0$) from the *Storage* component.

## Specification of Objective Function Contributions

The cost factor $\text{F}^\text{O}_{\omega\text{,l}}$ is for a *Storage* component
$\text{c}\in\mathcal{C}^\text{stor}$ given as

$$
\begin{aligned}
    &~~\text{F}^\text{O}_\text{c,+,l} &&~=~ \hat{X}^{\text{opex}_\text{O}}_\text{c,+,l} \nonumber \\
    &~~\text{F}^\text{O}_\text{c,-,l} &&~=~ \hat{X}^{\text{opex}_\text{O}}_\text{c,-,l}~.
\end{aligned}
$$
