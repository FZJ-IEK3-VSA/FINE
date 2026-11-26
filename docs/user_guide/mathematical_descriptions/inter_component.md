# Inter-Component Constraints

Inter-component constraints are constraints that involve variables and parameters from multiple components in
$\mathcal{C}$. The inter-component constraints which are modeled within the framework are commodity balances,
annual commodity inflow/outflow limits and shared potential constraints.

## Commodity Balances

The constraints that provide the basic structure of the energy system are the commodity balances. They are
defined for all commodities $\text{g}\in\mathcal{G}$, at all locations in $\text{l}\in\mathcal{L}$, if the
commodity appears at that location in the model, and, there, for all periods and time steps
$\theta \in \Theta$.

The commodity appears at a location when the set

$$
\begin{aligned}
    &\mathcal{C}_\text{g,l} ~=~ \big\{ &&\text{c} ~\vert~ \forall~ \text{c}\in\mathcal{C}:
     \text{g}\in\mathcal{G}^\text{c}~\wedge~ \big(\text{l}\in\mathcal{L}_\text{c}~\lor \\
    & &&(\exists~\text{l}^*\in\mathcal{L}: (\text{l},\text{l}^*)\in\mathcal{L}_\text{c}~\lor~(\text{l}^*,\text{l})\in\mathcal{L}_\text{c})\big)\big\}
\end{aligned}
$$

is not empty. In this case the commodity balance equation is given for all as

$$
\begin{aligned}
    &\sum\limits_{\text{c}~\in~\mathcal{C}_\text{g,l}} &&C_{\text{c,g,l,}\theta} ~=~ 0~.
\end{aligned}
$$

The definition of $C_{\text{c,g,l,}\theta}$ is given in the component model extensions.

### Pooled Commodities

For some commodities, trade between locations should be possible without an explicit transmission
component, e.g. to represent a market where commodities are freely tradable between a
group of locations. A commodity $\text{g}$ can be declared as pooled commodity, that is, 

$$
g \in \mathcal{G}^{\text{pool}} \subseteq \mathcal{G}.
$$

For each pooled commodity $\text{g}\in\mathcal{G}^\text{pool}$, a set of pools $\mathcal{P}_\text{g}$ is defined, and each pool $\text{p}\in\mathcal{P}_\text{g}$ is associated with a set of locations
$\mathcal{L}_\text{p}\subseteq\mathcal{L}$ at which trading among each other is allowed. A location may belong to at most one pool for a given commodity.

For pooled commodities, the balance is, other than above, not enforced individually for every location
$\text{l}\in\mathcal{L}_\text{p}$ but jointly for all locations within a pool. The commodity balance
equation therefore becomes, for all $\text{p}\in\mathcal{P}_\text{g}$, all $\text{g}\in\mathcal{G}^\text{pool}$
and all $\theta\in\Theta$,

$$
\begin{aligned}
    &\sum\limits_{\text{l}~\in~\mathcal{L}_\text{p}}~\sum\limits_{\text{c}~\in~\mathcal{C}_\text{g,l}}
    &&C_{\text{c,g,l,}\theta} ~=~ 0~.
\end{aligned}
$$

This equation states that individual locations may have nonzero net contributions, but the sum of all contributions across the pool must be zero. Since trade within a pool is already implicitly represented by the pooled balance, transmission components are not permitted for a commodity that has been declared as a pooled commodity.

## Shared Potential Constraints

As already explained in the *Basic* component model, two or more components can share a potential in an
energy system. The framework ensures that for each location/connection where a shared potential is specified,
the share on the maximum capacity of all components with the same identifier does not exceed 100%. Each
component for which a maximum capacity is defined can be associated with the shared potential by setting the
parameter $\text{sharedPotentialID}_\text{c}=\text{sharedPotentialID}$ (default: $\emptyset$).

Let $\mathcal{I}^\text{ID}$ be the set containing all shared potential IDs and let $\mathcal{L}^\text{ID}$ be
the set of locations or connections at which components compete for a maximum potential, respectively. The
shared potential constraints are then given for all $\text{i}\in\mathcal{I}^\text{ID}$ and all
$\text{l}\in\mathcal{L}^\text{ID}$ by

$$
\begin{aligned}
    &\sum\limits_{\text{c}~\in~\mathcal{C}^\text{i}}
    k_\text{c,l}/\text{K}^\text{max}_\text{c,l} ~\leq~ 1,\nonumber \\
    &\text{with}~~\mathcal{C}^\text{i} = \left\{ \text{c} ~\vert~ \forall~ \text{c}\in\mathcal{C}:
     \text{sharedPotentialID}_\text{c}=\text{i} \right\}~.
\end{aligned}
$$

## Component Limits

A component limit bounds a quantity summed over a *group of components* and a *group of locations*. It
answers questions such as "how much wind capacity may stand in these regions together", "how much may
these power plants produce in this year" or "how much CO$_2$ may these regions emit over the whole
pathway".

### Components and locations of a limit

A component joins a limit by naming the limit's identifier in its parameter
$\text{componentLimitID}_\text{c}$. The parameter takes a *list* of identifiers, so one component may
belong to several limits. Let $\mathcal{J}$ be the set of all identifiers in the index of
$\text{componentLimit}$. For each $\text{j}\in\mathcal{J}$,

$$
\mathcal{C}^\text{j} = \left\{ \text{c}\in\mathcal{C} ~\vert~ \text{j}\in\text{componentLimitID}_\text{c} \right\}
$$

is the set of components that carry the identifier.

A limit is summed either over locations or over transmission connections:

- **Over locations.** $\mathcal{L}^\text{j}$ holds the locations marked with 1 for $\text{j}$ in
  $\text{componentLimitEligibility}$.
- **Over connections.** $\mathcal{A}^\text{j}$ holds the connections $(\text{l},\text{l}^*)$ marked
  with 1 for $\text{j}$ in $\text{componentLimitEligibility2dim}$. A limit over connections includes
  only *Transmission* components.

An identifier must appear in exactly one of the two eligibility frames.

### The limit constraint

Each row of $\text{componentLimit}$ belongs to one identifier $\text{j}$ and builds one constraint

$$
\sum\limits_{\text{c}~\in~\mathcal{C}^\text{j}} E^\text{type}_\text{c,j} ~\lesseqgtr~ \text{value}~,
$$

where $\lesseqgtr$ is $\leq$, $\geq$ or $=$ for a $\text{bound}$ of `"upper"`, `"lower"` or `"fixed"`.
$E^\text{type}_\text{c,j}$ is the contribution of component $\text{c}$ over the locations or
connections of $\text{j}$. The column $\text{type}$ selects it from the four quantities below.

The column $\text{commodity}$ restricts a limit to one commodity. If it is set, a component contributes
only if it carries that commodity: a *Source*, *Sink*, *Storage* or *Transmission* component whose
commodity is the given one, or a *Conversion* component whose conversion factors, flexible conversion
groups or emission factors contain it. All other components of $\mathcal{C}^\text{j}$ contribute 0.

### Investment periods

A row applies to the investment period $\text{ip}$. If $\text{ipEnd}$ is set, the row applies to all
investment periods from $\text{ip}$ to $\text{ipEnd}$ and the quantity is summed over them:

- `"commissioning"` is summed as it is, because capacity is commissioned once per investment period.
- `"operation"` and `"balance"` are annual quantities. Each investment period stands for
  $\Delta^\text{ip}$ years, the $\text{investmentPeriodInterval}$ of the model, so its annual value is
  multiplied by $\Delta^\text{ip}$. A row over a range therefore bounds the total over all years of the
  range, as $\text{pathwayBalanceLimit}$ does.
- `"capacity"` cannot span a range, because an installed capacity is a stock, not a quantity that adds
  up over time.

In a stochastic model the investment periods are alternative scenarios, not steps of a pathway, so no
row may span a range there.

### Annual quantities

Operation variables $o_{\omega\text{,l,}\theta}$ hold the amount of one time step $\theta = (p,t)$.
The annual amount of an operation variable is

$$
\bar{o}_{\omega\text{,l}} = \sum\limits_{(p,t)~\in~\Theta} \text{w}_p \cdot o_{\omega\text{,l,}(p,t)}~,
$$

where $\text{w}_p$ is the number of times period $p$ occurs in the year. Without time series
aggregation, $\text{w}_p = 1$.

### Type "capacity"

$E^\text{cap}_\text{c,j}$ is the installed capacity $k_\text{c,l}$ in the investment period $\text{ip}$,
in the capacity unit of the component (the energy capacity for *Storage* components):

- *Source*, *Sink*, *Conversion*, *Storage*: $\sum_{\text{l}\in\mathcal{L}^\text{j}} k_\text{c,l}$.
- *Transmission* over connections: the capacity of each line in $\mathcal{A}^\text{j}$. A line has one
  capacity for both directions, so $(\text{l},\text{l}^*)$ and $(\text{l}^*,\text{l})$ name the same
  line and it is counted once.
- *Transmission* over locations: a line with both ends in $\mathcal{L}^\text{j}$ counts fully, a line
  with one end in $\mathcal{L}^\text{j}$ counts half.

### Type "commissioning"

$E^\text{com}_\text{c,j}$ is the capacity newly commissioned in the investment period, counted over
locations and lines exactly as `"capacity"`.

### Type "operation"

$E^\text{op}_\text{c,j}$ is the annual activity of the component. It is never negative, so a limit of
this type cannot be satisfied by one component offsetting another:

| Component | Annual activity |
|---|---|
| *Source* | the annual output $\sum_{\text{l}\in\mathcal{L}^\text{j}} \bar{o}_\text{c,l}$ |
| *Sink* | the annual intake $\sum_{\text{l}\in\mathcal{L}^\text{j}} \bar{o}_\text{c,l}$ |
| *Conversion* without $\text{commodity}$ | the annual operation in the physical unit of the component |
| *Conversion* with $\text{commodity}$ | the annual amount of the commodity it produces or consumes, $\lvert\text{factor}\rvert \cdot$ operation |
| *Storage* | the annual discharge |
| *Transmission* over connections | the annual flow sent on each directed connection in $\mathcal{A}^\text{j}$, before losses |
| *Transmission* over locations | not defined; such a row is rejected |

For a *Conversion* component, the commodity amount follows the conversion factors in the same way as
the commodity balance does, including conversion factors that depend on the commissioning year,
flexible conversion groups and the emissions given by emission factors.

### Type "balance"

$E^\text{bal}_\text{c,j}$ is the signed annual amount of the commodity that the component adds to the
locations $\mathcal{L}^\text{j}$. It is the commodity balance contribution $C_{\text{c,g,l,}\theta}$
summed over the year:

| Component | Annual contribution |
|---|---|
| *Source* | $+$ annual output |
| *Sink* | $-$ annual intake |
| *Conversion* | annual operation times the conversion factor of the commodity: positive for an output, negative for an input; emissions count as outputs |
| *Storage* | annual discharge $-$ annual charge |
| *Transmission* | annual import into $\mathcal{L}^\text{j}$ after losses $-$ annual export out of $\mathcal{L}^\text{j}$ |

A *Transmission* component counts only lines that cross the boundary of $\mathcal{L}^\text{j}$. Flows
between two locations of $\mathcal{L}^\text{j}$ do not count, so a limit over all locations receives
no contribution from *Transmission* components.

For a pooled commodity, trade between the locations of a pool has no variables and does not count. A
`"balance"` row over some locations of a pool therefore bounds their net production, which is the
amount they supply to the rest of the pool (or draw from it, if negative). The types `"capacity"`,
`"commissioning"` and `"operation"` count component variables only, which pooling does not change.

A `"balance"` row is summed over locations only. Without $\text{commodity}$, each *Source*, *Sink*,
*Storage* and *Transmission* component counts in its own commodity; a *Conversion* component then
cannot be a member, because its balance exists only per commodity.

### Rejected rows

A row is rejected with an error if

- its identifier appears in neither or in both eligibility frames, or no location or connection is
  marked eligible for it,
- its $\text{value}$ is not a finite number,
- $\text{ipEnd}$ lies before $\text{ip}$, or a range is used where the type or the model does not
  allow one (see above),
- no variable contributes to it and 0 does not satisfy its bound (a `"lower"` bound above 0, an
  `"upper"` bound below 0 or a `"fixed"` bound other than 0), because such a constraint cannot be
  satisfied.

A row without contributing variable whose bound 0 satisfies builds no constraint. A row whose
identifier no component carries builds no constraint and produces a warning.

### Component limit and balance limit

A `"balance"` row reproduces a $\text{balanceLimit}$: a row per location reproduces the per-location
limits, and a row over all locations reproduces the column "Total". A `"balance"` row over the range of
all investment periods reproduces a $\text{pathwayBalanceLimit}$.
