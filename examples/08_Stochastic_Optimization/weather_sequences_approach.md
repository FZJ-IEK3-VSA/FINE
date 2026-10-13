# Weather sequences as branches of the scenario tree

This note describes how weather uncertainty enters the multi-stage stochastic program (MSSP)
through **multi-year weather sequences** selected with tsam. Every representative sequence
becomes a **node of the scenario tree**: the tree branches on weather.

---

## 1. The approach in short

- **Stages.** A stage of the scenario tree covers several yearly investment periods, e.g. 5:
  2035–2039.
- **Sequences.** A weather sequence is a run of consecutive weather years of the same length,
  e.g. 1998, 1999, 2000, 2001, 2002.
- **Selection.** tsam selects a few **representative sequences** (e.g. 5, one of which is
  forced to be the extreme sequence) from candidate sequences cut from the weather record,
  possibly overlapping and possibly extended with climate projections.
- **Every representative sequence becomes one node.** At the start of a stage, the tree
  branches into one child node per representative sequence. A node's 5 investment periods
  get the 5 years of its sequence, **one weather year (8760 hours) per investment period**:
  year 1 of the sequence becomes the weather of 2035, year 2 that of 2036, and so on.
- **Probabilities of the branches:** either the cluster weights from tsam (optionally with a
  mild recency weighting, section 6) or uniform (1/k).
- **For now, the tree branches only on weather.** Demand follows one structural path.
  Structural uncertainty (demand level, electrification, …) could be added later: a node would
  then combine a structural scenario and a weather sequence, with probability
  p_structure · p_weather.

---

## 2. Example

Three representative 5-year sequences with tsam weights (the prototype notebook uses 5):

| Sequence | Years | Probability |
|---|---|---|
| S1 | 1998 – 2002 | 0.50 |
| S2 | 2007 – 2011 | 0.40 |
| S3 | 1985 – 1989 (with a severe Dunkelflaute in 1987) | 0.10 |

**Tree.** Stage 1 (root, 2030–2034) and stage 2 (2035–2039):

```
                        ┌── node "S1"  p = 0.50   2035: 1998, 2036: 1999, 2037: 2000, 2038: 2001, 2039: 2002
root (stage 1)  ────────┼── node "S2"  p = 0.40   2035: 2007, 2036: 2008, 2037: 2009, 2038: 2010, 2039: 2011
2030–2034               └── node "S3"  p = 0.10   2035: 1985, 2036: 1986, 2037: 1987, 2038: 1988, 2039: 1989
```

**What a node holds.** For every investment period, one weather year: the wind and PV
capacity factors of that year, and the demand built from the structural parameters of that
investment period and that year's weather profiles:

```
demand(node, ip) = base(ip) · shape + heatPumps(ip) · heatProfile(weather year of ip) + …
```

Demand and renewable potentials always come from the **same** weather year. Without structural
branching, two nodes differ only in their weather years.

**How it is solved.** Every leaf of the tree is one scenario: one FINE model over the whole
horizon, holding the weather years along its path. mpi-sppy enforces non-anticipativity:
decisions taken in a node are equal in all scenarios that pass through it.

---

## 3. What the tree structure implies: when weather is known

In the MSSP module, **the data of a node are known when that node's decisions are made**.
Non-anticipativity applies between nodes, not within a node.

- **The first-stage decisions (root) hedge against the weather of the next stage.** They have
  to be good for S1, S2 and S3 together, weighted by their probabilities. This is the actual
  purpose of the tree.
- **Within a node, the weather of its own years is known.** The stage-2 node "S3" decides its
  investments for 2035–2039 knowing that 2037 will be a Dunkelflaute year. The node "S1"
  knows it will not see one. So stage-2 decisions are tuned to their sequence, which is
  optimistic: in reality, the weather of 2037 is not known in 2035.
- **The root has one weather sequence for its own years**, so the first-stage decisions also
  know the weather of 2030–2034. Which sequence the root holds is a modeling choice (section
  8): for example, the most probable representative, or recent observed years.

**What this means:**
- **For the rolling-horizon use**, where only the first-stage decisions are implemented: the
  first stage is hedged against the *next* stage's weather, but tuned to *its own* weather.
  Shorter stages reduce this effect. So does an out-of-sample evaluation of the first stage
  under other realized sequences (as in the notebook).
- **Later-stage decisions** are re-optimized in the next rolling-horizon window anyway. Their
  weather foresight therefore only affects the first stage indirectly, through how the model
  values capacity carried over into stage 2.

**Observed in the prototype (without lead times):** the first-stage decisions hardly change
through weather branching. A stage-2 node can build what it needs in its own years, knowing its
weather, so the first stage has little reason to hedge.

**How this is addressed in the planned application:**
- **The weather of the first stage comes from the Pan-European Climate Database (PECD)** in the
  rolling-horizon application, not from the representative sequences. So the question of which
  representative the root holds does not arise there. The PECD weather can differ considerably
  from the representative years.
- **Lead times.** Capacity decided in a stage only becomes available after its lead time, often
  only in a later stage. The first stage then has to anticipate the weather of the following
  stage. This is expected to favor technologies that can be deployed quickly (wind, PV)
  over those with long lead times (e.g. gas plants).
- **The length of the first stage** (possibly shorter than 5 years) will be chosen with the
  actual weather data.

An alternative without within-node weather foresight is sketched in section 10.

---

## 4. Building the candidate sequences

### 4.1 Non-overlapping and overlapping windows

- **Non-overlapping:** with N historical years and sequences of length L, there are only N/L
  candidates. For example, 40 years and L = 5 give 8.
- **Overlapping (sliding window):** every year starts a sequence: N − L + 1 candidates, e.g.
  36. Neighboring windows share L − 1 years, so the *effective* sample size stays about N/L.
  Overlap gives tsam more, and smoother, candidates, but no new information.

### 4.2 How to make overlapping windows work with tsam

tsam cuts one time series into **consecutive, non-overlapping** periods, so the overlap has to
be built into the input:

1. **Compute features per weather year** (per region: demand, wind and PV capacity factors,
   residual load, and persistence features such as 7- and 14-day rolling means of the
   residual load).
2. **Compress every year** to its duration curves sampled at e.g. 365 points.
3. **Build every window explicitly** by concatenating the compressed years of the window
   **in order**. Each year is copied into every window that contains it.
4. **Stack all windows** and call tsam with **one period = one window**
   (`period_duration` = L × points per year).

The distance between two sequences then compares year 1 with year 1, year 2 with year 2, and
so on.

### 4.3 Settings

- **`ClusterConfig(method="hierarchical", representation="medoid")`:** the representatives are
  real sequences. `kmedoids` solves an exact optimization problem and becomes slow.
- **Keep the worst sequence with its true weight.** The worst sequence is the window that
  *contains* the worst week, i.e. the largest 7-day rolling residual load. With tsam that is
  `max_value`; `max_period` would pick the highest *average*. To make sure it really becomes a
  separate representative, take it out of the candidates **before** clustering and add it as its
  own branch. tsam's `ExtremeConfig(method="append")` skips the extreme period when it already
  forms a cluster of its own, which leaves one representative too few (observed in the
  prototype with 4 + 1 representatives).
- **Compress before clustering.** tsam's preprocessing gets slow for long periods: about 80 s
  for a single 8760-hour year × 9 features. With compressed curves it takes seconds.
- **Optional sequence-level features**, e.g. the mean residual load over the whole sequence,
  or the number of years with a severe Dunkelflaute. They capture "several bad years in a row".

---

## 5. The years at the edges of the record

**With a sliding window, the first and last years appear in fewer sequences.** The last year
appears only in the last window, and only in the last position. As a result:

1. **The edge years are under-represented.** The most recent years, probably the most
   relevant ones, count least.
2. **The positions differ.** The first investment period of a stage can never get the last
   historical year.

**Options:**

| Option | How | Pros | Cons |
|---|---|---|---|
| **a) Extend the record with projections** | append climate-projection years (e.g. EURO-CORDEX RCP4.5 from SECURES) after the last historical year | the recent years appear in all positions; the projections bring the trend | mixes reanalysis and model data; depends on the projection chosen |
| **b) Circular windows** | treat the record as a ring: [N−1, N, 1, 2, 3] ("circular block bootstrap", Politis & Romano 1992) | every year appears exactly L times, once in each position; no extra data | artificial transitions from the last year to the first. These matter, because the sequences are now actual weather paths of the nodes |
| **c) Accept it** | complete windows only | simple | bias towards the middle of the record |
| **d) Re-weight the windows** | windows near the edges get more weight | no extra data | only approximately possible; tricky |

**Suggestion:** option a. Because each node now follows its sequence year by year, artificial
transitions from circular windows (option b) end up in the model as real weather paths.

---

## 6. Probabilities and trends

**The probability of a node** is either:
- **the tsam weight** of its sequence: the share of candidate windows in its cluster. Together
  with recency weights, it is the sum of the members' recency weights (see below); or
- **uniform**, 1/k for k representative sequences.

**Trend weighting by recency.** Give each candidate window a base weight that grows with how
recent it is. **The weighting should be mild:** recent sequences get a bit more focus, but no
sequence may become negligible. A simple choice is a linear ramp:

```
w_j = 1 + boost · (startYear_j − oldestStartYear) / (latestStartYear − oldestStartYear)
```

With `boost = 0.5`, the most recent window counts 1.5 times as much as the oldest. A steep
exponential weighting (e.g. a 10-year half-life) is too strong: in the prototype, it reduced
the probability of an old extreme sequence to about 3 %.

**tsam cannot weight periods** (tsam 4.1.1 only has column weights). There are two ways:

| Way | How | Effect |
|---|---|---|
| **1. After clustering** (recommended) | cluster unweighted; each representative's probability = sum of its members' base weights / total | probability redistribution of scenario reduction with non-uniform base probabilities (Dupačová, Gröwe-Kuska and Römisch, 2003); the clusters do not depend on the weights |
| **2. Duplicate windows** | copy windows in proportion to their (integer) weights | the weights also pull the clustering towards recent windows; the input grows |

With way 1, the medoid can be an old window although most of the cluster's weight comes from
recent ones. If that matters, take the member closest to the *weighted* cluster center.

**The extreme sequence** is forced into the representative set (`ExtremeConfig(method="append")`)
and keeps its true weight. With the mild recency weighting, an old extreme sequence keeps a
reasonable probability.

**One representative set per stage** follows the trend over the planning horizon: each stage
selects from windows around its own time, or from projections.

---

## 7. Implementation in FINE and the MSSP module

- **One weather year per investment period.** The model has `numberOfTimeSteps = 8760`, and
  every investment period has its own time series.
- **The weather data are uncertain parameters of the tree.** Per node and investment period:
  `Wind.operationRateMax`, `PV.operationRateMax` (and other weather-driven time series), and
  `Demand.operationRateFix`, built from the structural parameters and the weather year. All of
  these are per-investment-period parameters in FINE, so the module accepts them as uncertain
  values.
- **The weights are the node probabilities.** mpi-sppy weights the scenarios exactly; no
  stacking or repetition is needed.
- **Sequence length = number of investment periods of the stage.** With stages of different
  length, the sequences per stage have different lengths.
- **Time series aggregation** (typical days) runs per scenario and investment period, i.e. per
  weather year.
- **First-stage decisions:** the module's `firstStageDecisions()` only returns the first
  investment period of the first stage. With stages of several investment periods, read the
  `commis_*` variables of all first-stage investment periods (workaround in the notebook).

### 7.1 Storage

**FINE's storage handling fits this approach as it is.** The state of charge runs hour by hour
through the year of an investment period and is closed at the end: level at the end of the
year = level at the start (`cyclicState`, `fine/storage.py:1004-1006`; with typical days
`:1024-1026`). Nothing is carried from one investment period to the next. So every investment
period is a self-contained, cyclic weather year.

- **Seasonal storage within a year works**, also with typical days: the between-day level is
  chained over the days of the year (`connectInterPeriodSOC`, `storage.py:1163-1201`).
  `isPeriodicalStorage` is **not** needed.
- **Storage across years** (carrying energy from 2036 into 2037, e.g. for multi-year reservoirs)
  is not modeled in FINE today. With sequences, it would be meaningful, because a node's years
  are a real consecutive path. It would need a new link between consecutive investment periods
  in FINE core.

### 7.2 Yearly limits

Because each investment period is one year, yearly limits (e.g. a CO₂ cap or full-load hours)
apply **per weather year**, as intended. Budgets over several years apply per node path, i.e.
per weather sequence.

---

## 8. Tree size and open modeling choices

- **Tree size.** With k sequences per stage, the tree has k^(stages − 1) leaves. With k = 5
  and 3 stages, that is 25 scenarios. With structural branching added, it becomes
  (k · s)^(stages − 1). The extensive form grows accordingly; progressive hedging (one scenario
  per process) becomes attractive.
- **Weather of the root.** In the rolling-horizon application, it comes from the PECD
  (section 3). In the stand-alone prototype, the root holds the most probable representative.
- **Overlapping representatives.** Neighboring windows are similar, so two representatives can
  share most of their years (e.g. 2002–2004 and 2003–2005 in an earlier prototype run with 3
  representatives). Requiring the representatives not to overlap, or to keep a minimum distance,
  keeps the branches diverse. To be reviewed with 5 representatives (see the overlap table in
  the notebook).

---

## 9. Conditions and limitations

- **Within-stage weather foresight** (section 3): decisions in a node know that node's weather.
  Weather is uncertain only across stages.
- **Representative sequences** stand in for the whole distribution. With few branches, the
  tails are captured mainly through the extreme sequence.
- **Typical days per weather year** smooth the peaks. Choose the aggregation level with an
  out-of-sample test at full resolution: in an earlier prototype, 12 typical days clearly
  underestimated the firm capacity needed.
- **FINE bug:** with several `extremes` criteria in `aggregateTemporally`, investment periods
  can end up with different numbers of typical periods, which crashes storage models
  (`esM.typicalPeriods` is shared across investment periods, `fine/energySystemModel.py:1178`).
  Use a single extreme criterion until it is fixed.
- **The synthetic weather of the prototype is not realistic enough for results** (compared with
  the real hourly data in `examples/data/multiRegion` in the notebook): only 4–10 % of the hours
  have wind CF < 0.1 (real: 35 %), the mean PV CF is 0.074–0.079 (real: 0.122) and the demand
  peaks are too weak (peak/mean 1.35 vs. 1.59). It underestimates scarcity, so it is good for
  testing the method but not for its numbers. Use real multi-year data (PECD) for any
  quantitative statement.

---

## 10. Further ideas

**All sequences in every investment period ("stacked").** Instead of branching, every
investment period holds all representative weather years at once, as parallel 8760-hour blocks
weighted by their probabilities. Capacity then has to work in all of them, and no node knows its
weather in advance. This avoids the within-stage weather foresight of section 3. The costs:
- the order of years is lost: FINE operates each investment period on its own, so only the set
  of years per investment period matters;
- FINE has no per-block cyclic storage;
- yearly limits only hold on average over the blocks.

It was prototyped earlier. Stacking gave exactly the same optimum as one scenario per sequence
when the decisions are fixed before the sequences branch.

---

## 11. Open points

1. **Probabilities:** tsam weights or uniform (both available in the notebook)?
2. **Edge years:** projections (recommended here) or circular windows (section 5)?
3. **Overlapping representatives:** require non-overlapping representatives (section 8)?
4. **Length of the first stage** and the interplay with lead times: to be explored with the
   actual weather data (section 3).
5. **Storage across years:** is it needed (section 7.1)?
6. **Structural branching:** to be added later?

Decided so far: weather branches only; 5 representatives including one extreme sequence; mild
recency weighting; the first-stage weather comes from the PECD in the rolling-horizon
application.
