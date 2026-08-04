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

A component limit bounds the summed capacity, commissioning or annual operation of a *group of components* over
a *group of locations*. It answers questions such as "how much wind capacity may stand in these regions
together" or "how much may these technologies produce in this year in total".

A component joins a limit by naming the limit's identifier in its own parameter
$\text{componentLimitID}_\text{c}$. Unlike $\text{balanceLimitID}_\text{c}$, this parameter takes a *list* of
identifiers, so one component may belong to several limits at once.

Let $\mathcal{J}^\text{ID}$ be the set of all component limit identifiers. For each
$\text{j}\in\mathcal{J}^\text{ID}$ let

$$
\begin{aligned}
    &\mathcal{C}^\text{j} = \left\{ \text{c} ~\vert~ \forall~ \text{c}\in\mathcal{C}:
     \text{j}\in\text{componentLimitID}_\text{c} \right\}
\end{aligned}
$$

be the components that carry the identifier, and let $\mathcal{L}^\text{j}$ be the locations marked eligible
for it in $\text{componentLimitEligibility}$, or the connections marked eligible in
$\text{componentLimitEligibility2dim}$. The component limit constraint is then, for every row of
$\text{componentLimit}$ and its investment period $\text{ip}$,

$$
\begin{aligned}
    &\sum\limits_{\text{c}~\in~\mathcal{C}^\text{j}} \sum\limits_{\text{l}~\in~\mathcal{L}^\text{j}}
    E^\text{comp}_\text{c,l,ip} ~\lesseqgtr~ \text{value}_\text{j}~,
\end{aligned}
$$

where $\lesseqgtr$ is $\leq$, $\geq$ or $=$ for a $\text{bound}$ of `"upper"`, `"lower"` or `"fixed"`, and
$E^\text{comp}$ is

- the installed capacity $k_\text{c,l,ip}$ for a $\text{type}$ of `"capacity"`,
- the newly commissioned capacity for a $\text{type}$ of `"commissioning"`,
- the annual operation for a $\text{type}$ of `"operation"`.

A row may span a range of investment periods by setting $\text{ipEnd}$, in which case the sum also runs over
the periods from $\text{ip}$ to $\text{ipEnd}$. A `"capacity"` row cannot do this, because an installed
capacity is a stock and not an additive quantity; use `"commissioning"` there.

### Component limit or balance limit?

The two constraints have similar names but bound different things.

| | $\text{balanceLimit}$ | $\text{componentLimit}$ |
|---|---|---|
| Bounded quantity | the signed commodity balance $E^\text{source} - E^\text{sink} + E^\text{imp} - E^\text{exp}$ | the capacity, commissioning or operation of a named set of components |
| Component classes | Source, Sink and Transmission | Source, Sink, Storage, Conversion and Transmission |
| Constraints produced | one per location, plus one for the total | one over the sum of all eligible locations |
| Identifier per component | a single string | a list of strings |

Use $\text{balanceLimit}$ for a commodity budget, such as a CO$_2$ cap or an import volume. Use
$\text{componentLimit}$ for a cap on how much of a set of technologies may be built or run.
