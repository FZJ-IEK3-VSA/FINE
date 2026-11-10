# FINE 3 – Refactoring Plan for the Input System

Last updated: 2026-11-10, checked against `develop` at commit `2368d498`.
This is a living plan: it is reviewed and revised before and after every PR (section 0).
It replaces the older "FINE Input Refactor Plan". Line numbers refer to that commit and go stale with every merge.

## 0. Way of working

### Branch and location

- Every PR of this plan targets `develop` (decision of 2026-11-10, replaces the earlier integration-branch approach).
- The plan lives on `fine_refactor` as `FINE3_refactoring_plan.md`. Every revision is recorded there. `fine_refactor` holds only the plan.
- Every PR gets its own branch from `develop`, created only when the user asks for it. The branch holds only the code changes of the PR, never this plan. A PR that needs an unmerged PR branches from that PR's branch.
- `develop` is merged into `fine_refactor` regularly so the line numbers in the plan stay current.
- The plan and all notes, comments and PR texts are written in English.
- The plan is not followed blindly. Before and during every PR, check whether each planned step still makes sense against the current code (e.g. paths that no longer exist, helpers placed where only one test uses them, scope that contradicts the minimal-footprint rule) and raise every doubt with the user before implementing it.

### Status

Values for "Status": open, in progress, in review, merged. The last two columns hold the date of the plan review.

| PR | Status | GitHub | Plan reviewed before | Plan reviewed after |
|---|---|---|---|---|
| 1 | in review | #884 | 2026-11-10 | – |
| 2 | in review | #886 | 2026-11-10 | – |
| 3 | in review | #885 | 2026-11-10 | – |
| 4 | open | – | – | – |
| 5 | open | – | – | – |
| 6 | open | – | – | – |
| 7 | open | – | – | – |
| 8 | open | – | – | – |
| 9 | open | – | – | – |
| 10 | open | – | – | – |
| 11 | open | – | – | – |
| 12 | open | – | – | – |
| 13 | open | – | – | – |
| 14 | open | – | – | – |
| 15 | open | – | – | – |
| 16 | open | – | – | – |
| 17 | open | – | – | – |
| 18 | open | – | – | – |
| 19 | open | – | – | – |
| 20 | open | – | – | – |
| 21 | open | – | – | – |

### Checklist before every PR

- [ ] `fine_refactor` is up to date with `develop`
- [ ] Line numbers and function names of this PR checked against the current `develop`
- [ ] All PRs listed under "Needs" are merged
- [ ] Open decisions that affect this PR (section 10) are made and recorded
- [ ] Scope, file list and acceptance of this PR are concrete enough to implement; missing target signatures or names are added
- [ ] Status table updated

### Checklist after every PR

- [ ] Acceptance criteria met and evidenced (CI run of the PR, or command and result)
- [ ] Status and PR number recorded
- [ ] Deviations from the plan noted in the section of the PR
- [ ] Follow-up PRs reviewed: dependencies, scope and line numbers adjusted
- [ ] Section 2 (numbers), section 10 and section 11 updated
- [ ] New findings recorded as a note on the affected follow-up PR

### Stop rule

If an LP file changes up to and including block C: do not continue, investigate the cause and report it. The reference is not regenerated. The only exceptions are the PRs where the plan explicitly foresees regenerating it (follow-up PR to PR 16, PR 21).

### Environment

- Environment `fine_env`; install packages with `mamba`.
- The test suite is not run locally before a PR is created; the CI on GitHub runs it. Local runs use `pytest -n 4`.
- Commits and PRs follow `CLAUDE.md`: ask before every commit and before every PR.

## 1. Summary

The input system is rebuilt in 21 small PRs onto `develop`. The public constructors stay flat and backward compatible. The rebuild happens behind them:

- The check functions in `fine/utils.py` become pure: values in, result out.
- Each component family gets one frozen input object as the single source for Pyomo, TSA, results and I/O.
- Silent repairs become a warning first, then an error.
- After that, three API extensions come almost for free: constructing components without `esM`, `replaceComponent`, stricter period keys.

Up to and including block C, no PR changes an LP file.

## 2. Starting point on `develop`

| Finding | Number | Evidence |
|---|---|---|
| Constructor parameters `Source` / `Conversion` / `Storage` / `Transmission` | 41 / 45 / 49 / 38 | of which required: 4 / 4 / 3 / 3 (incl. `esM`, `name`) |
| Functions in `fine/utils.py` | 78, of which 65 are input checks by name | Appendix A |
| of which take `esM` as an argument | 35 | Appendix A |
| of which additionally take `comp`/`component` | 8 | Appendix A |
| `raise` sites in `fine/utils.py` | 211 | – |
| distinct `processed*` / `full*` / `preprocessed*` / `aggregated*` attributes that are set | `component.py` 10, `sourceSink.py` 18, `conversion.py` 15, `storage.py` 22, `transmission.py` 17 | – |
| Files outside the components that read these attributes | `energySystemModel.py`, `results/` (3 files), `expansionModules/` (2), `utils.py`, `utilsPWLCF.py` | – |
| I/O via class names and signature inspection | `fine/IOManagement/dictIO.py:41`, `:61`, `:220` | – |
| Silent repairs | `fine/utils.py:239`, `:645`, `:2272-2273`, `:2296`, `discard=True` in `:2264` | – |
| Implicit factor 0.5 on transmission costs | `fine/transmission.py:374`, `:383`, `:392`, `:401` | – |
| Class names as strings | `fine/component.py:642`, `fine/energySystemModel.py:556`, `:685` | – |

Milestone "Fine3" on GitHub: 5 items done, 6 open (#786, #768, #803, #806, #148, #122), no due date.

## 3. Guidelines and decisions made

### Guidelines

- Small PRs onto `develop`, each mergeable on its own and with a green suite.
- Every PR does exactly one thing. No clean-up on the side.
- Up to and including block C: the LP files of the fixtures stay identical.
- Behaviour changes come with one release of advance warning (`FutureWarning`). Exception: spagat is removed without warning (PR 3, decision of 2026-11-10).
- `fine3_working` serves only as inspiration for concepts. No code from it is merged, cherry-picked or ported.

### Concepts taken over

- pure check and normalisation functions
- one frozen input object per component family
- a small, frozen system context instead of `esM` as an argument (only from PR 18 on)
- a structured error with component and field path
- type IDs and explicit field lists in I/O, `schemaVersion`

### Deliberately dropped

| Idea from the old plan | Reason |
|---|---|
| Grouping into `design` / `economics` / `technical` | The number of parameters stays the same; only a level is added. The assignment is not stable: `operationRateMax` was under `design` in the prototype and under `technical` in the old plan. The prototype had to flatten the groups again for I/O and `updateComponent`. Breaks every user script. |
| `LegacyConstructorMixin` with signature tables | One class name with two calling conventions is fragile. In the prototype, positional legacy calls broke. |
| Pydantic | Covers only local checks; pandas fields are checked for type only. Context and model rules need custom functions anyway. New runtime dependency. Dataclasses are enough. |
| Handles, `add_many`, transaction layer | Names are already unique identifiers. `add` is atomic as soon as checking happens before inserting. |
| Plugin registry with codecs and schema migrations | The repo only has the built-in subclasses. A dict from type ID to class is enough. |
| Rejecting settings that have no effect | Gets in the way of scenario scripts that set parameters across the board. |
| Own netCDF input schema | The format is defined by #768 and #803. |
| Migration command line tool | Old files continue to be read; a separate tool is not needed. |

### Deferred

**Many constructor parameters.** The plan does not solve this problem; the decision of 2026-11-10 is: leave it out for now. Recorded for later:

- Only 3–4 parameters per family are required. The burden is in reading and maintaining, not in calling.
- More useful than categories would be optional bundles per function (build decision, transformation pathway and stock, special costs). Such a bundle switches a formulation on.
- Earliest sensible time: after block C, because then each family has an explicit field list.
- Merging the `Min`/`Max`/`Fix` triples is not worth it: small gain, large break.

## 4. Target picture

### Data flow

```text
Constructor        stores the input, checks only local things (type, value range, NaN)
esM.add()          pure check functions + system data → one frozen input object per family
Pyomo, TSA,        read only this object
Results, I/O
optimize()         model-wide check (shared potentials, linked quantities, balance limits)
```

### Three check levels

| Level | Checks | Needs | Runs in |
|---|---|---|---|
| Local | type, value range, NaN, contradictory flags | only the input | constructor |
| Context | locations, periods, time steps, commodities, eligibility, order of bounds, stock | system context | `esM.add()` |
| Model-wide | shared potentials, linked quantities, balance limits, pooled commodities | all components | once before `optimize()` |

Until PR 18 the constructor calls the build function itself at the end. After that, this one call moves to `esM.add()`.

### API after the rebuild

```python
import fine as fn

esM = fn.EnergySystemModel(            # unchanged
    locations={"DE", "FR"},
    commodities={"electricity"},
    commodityUnitsDict={"electricity": "GW"},
    numberOfTimeSteps=8760,
    hoursPerTimeStep=1,
    startYear=2030,
    numberOfInvestmentPeriods=3,
    investmentPeriodInterval=5,
)

wind = fn.Source(                      # new: without esM
    name="Wind",
    commodity="electricity",
    hasCapacityVariable=True,
    capacityMax=pd.Series({"DE": 50, "FR": 30}),
    operationRateMax=cf,
    investPerCapacity={2030: 1.5, 2035: 1.3, 2040: 1.1},
    interestRate=0.05,
    economicLifetime=20,
)

esM.add(wind)                          # checks against locations, periods, time steps here
esM2.add(wind)                         # new: the same definition in a second model

esM.replaceComponent("Wind", windV2)   # new: replaces updateComponent
esM.optimize()                         # unchanged, checks model-wide rules first
```

Old calls of the form `fn.Source(esM=esM, name="Wind", ...)` remain valid.

Errors are reported collectively with a field path (the text is a proposal):

```text
InputValidationError: Source 'Wind' – 2 problems
  capacityMax[CH]: unknown location
  investPerCapacity[2032]: not an investment year (allowed: 2030, 2035, 2040)
```

### Sketch of the build function

```python
def buildSourceInput(raw, <system data>) -> SourceInput:
    errors = InputErrorCollector(raw.name)
    with errors.field("locationalEligibility"):
        elig = ...                                  # exactly once
    with errors.field("capacityMax"):
        capacityMax = utils.checkAndSetBounds(...)
    with errors.field("operationRateMax"):
        operationRateMax = utils.checkAndSetInvestmentPeriodTimeSeries(...)
    ...
    errors.raiseIfAny()
    return SourceInput(capacityMax=capacityMax, operationRateMax=operationRateMax, ...)
```

The collector catches `ValueError` and `TypeError` per field. That way the 211 `raise` sites in `utils.py` do not have to be touched.

## 5. Building blocks and their benefit

| Block | PRs | What we rebuild | What it brings |
|---|---|---|---|
| A. Safety net | 1–4 | LP comparison for fixtures, tests for rejected inputs, remove spagat and `useProcessedValues` | Every later PR can show that it does not change the model. One consumer of the old attributes less. |
| B. Error type and pure checks | 5–10 | `InputValidationError` with collector; `utils` functions get values instead of `esM`/`comp` | All input errors at once. Checks testable without a model. Prerequisite for C and E. |
| C. One input object per family | 11–15 | Frozen dataclass object; old attribute names become read-only properties | One source instead of parallel attributes. Raw and processed value cannot diverge. |
| D. Strictness and I/O | 16–17 | Warning on silent repair; type IDs, field lists, `schemaVersion` | Wrong inputs get noticed. Saved models survive renames. |
| E. Public API | 18–20 | `esM` optional, `replaceComponent`, period keys | Components reusable across models. Replacing is checked and atomic. |
| F. Transmission edges | 21 | One edge instead of two directions | Factor 0.5 disappears, costs are given once. |

C has the largest benefit and the largest effort. B and C together make E cheap.

## 6. The PRs in detail

### Block A – Safety net

#### PR 1 – `test: add lp snapshot check for fixtures`

- **Goal:** Evidence that a rebuild does not change the optimisation model.
- **Changes:** Helper in `test/conftest.py` that declares the Pyomo model for existing fixtures, writes it as an LP file and compares it with a stored reference. New test file, references under `test/data/`.
- **Fixtures:** `minimal_test_esM`, `single_node_test_esM`, `multi_node_test_esM_init`, `dsm_test_esM`, `balanceLimitConstraint_test_esM`, `perfectForesight_test_esM`.
- **First step in the PR:** Clarify the stability of the LP output (symbolic names, ordering): write the same fixture twice and compare, measure the file size for `multi_node_test_esM_init`. If the output is not stable or too large, a substitute criterion is defined and recorded here before continuing. `fine/energySystemModel.py` does not write an LP file itself today; the export goes through Pyomo.
- **Result of the first step (2026-11-10):**
  - Within one process the LP output is byte-identical. Across processes it depended on `PYTHONHASHSEED` (order of rows and terms). Writing with `io_options={"symbolic_solver_labels": True, "file_determinism": 20}` makes it byte-identical across hash seeds; no numeric differences.
  - `balanceLimitConstraint_test_esM` draws unseeded random data; the test calls `np.random.seed(0)` first. Its LP has 5.7 MB and is stored as `.lp.gz` (about 300 KB).
  - `multi_node_test_esM_init` is dropped: 1.15 GB unaggregated (7 min); with time series aggregation still 19 MB and dependent on tsam's clustering (local tsam differs from the CI version, every tsam update would trigger the stop rule).
  - References under `test/data/golden/lp/`, regenerated with `UPDATE_GOLDEN=1` as in `test/IOManagement/test_xarrayio.py`. `.gitignore` excludes them from its `*.lp` rule.
- **Deviations (2026-11-10):** PR #884, branch `test/lp-snapshot-check`. Local evidence before the PR: the test passes under `PYTHONHASHSEED` 1–3 and fails when a coefficient is changed on purpose; the full suite runs in CI only.
- **Note:** `test/data/golden/` already holds a golden mechanism for results (`test/IOManagement/test_xarrayio.py`). Align the location of the LP references with it.
- **Acceptance:** passes on `develop`; fails when a coefficient is changed on purpose.
- **Needs:** nothing.

#### PR 2 – `test: add rejected-input tests per component family`

- **Goal:** Notice lost checks. An LP comparison does not detect them, because rejected inputs produce no model at all.
- **Changes:** For every check function called from a constructor (Appendix A), at least one test with invalid input per family.
- **Scope:** not each of the 211 `raise` sites, but every function and every family.
- **Exception types:** The tests assert today's type (`ValueError` or `TypeError`). Tests on `TypeError` break on purpose in PR 11–15 and are adapted there (decision of 2026-11-10).
- **Acceptance:** Every function from Appendix A with callers in the component files has at least one negative test.
- **Needs:** nothing.
- **State (2026-11-10):** PR #886, branch `test/rejected-input-tests`. `test/test_rejectedInputs.py`, 95 tests. A mutation check (every `raise` of the target function turned into `pass`) kills 91 of 95 cases; the 4 survivors are the `checkAndSetInvestmentPeriodParameters` cases, whose raise sits in `checkInvestmentPeriodParameters`.
- **Not testable with `pytest.raises`:** `setStockCapacityStartYear`, `sortTimeSeriesColumns`, `process2dimCapacityData`, `preprocess2dimData` (no raise of their own); `checkOperationRateForCapacityVariable` (warns only); `checkSimultaneousChargeDischarge` (only after a solve); `checkAndSetFullLoadHoursParameter` for Storage and Transmission (their constructors do not forward `yearlyFullLoadHoursMin/Max`).
- **Found bug (not fixed, out of scope):** for Storage and Transmission, any valid `partLoadMin` raises `TypeError: 'NoneType' object is not subscriptable` in `checkAndSetPartLoadMin`, because `fullOperationMin` is `None` there.

#### PR 3 – `refactor: remove spagat` (issue #806, no PR yet)

- **Goal:** Remove spatial and technology aggregation.
- **Changes:** `fine/aggregations/spatialAggregation/`, `fine/aggregations/technologyAggregation/`, `test/spagat/`, `test/aggregations/`, fixture `test_esM_for_spagat`, the import in `fine/energySystemModel.py:23` and the method containing the call in `:941`, the example `examples/07_Spatial_and_technology_aggregation/`.
- **Stays:** `fine/aggregations/temporalAggregation/` (`deprecatedKeywords` is imported in `fine/energySystemModel.py:26`).
- **Dependencies in `pyproject.toml` and `requirements_dev.yml`:** check `rasterio`, `scikit-learn`, `networkx` and remove them where only spagat needs them. `geopandas` stays (plotting, `fine/IOManagement/standardIO.py`).
- **No advance warning:** spagat is removed without a warning release.
- **Acceptance:** suite green; `import fine` works without the removed packages.
- **Needs:** nothing.
- **Deviations (2026-11-10):** PR #885 (branch `refactor/remove-spagat`) targets `develop` (user decision). `test/spagat/` did not exist; `test/aggregations/temporalAggregation/` stays. Also removed: the unused-after-removal `timer()` in `fine/IOManagement/standardIO.py`, the `xrIO` import in `fine/energySystemModel.py`, docs/README/mkdocs references, the three packages in `.github/workflows/test_pull_min_max_versions.yml`. The geopandas `ImportError` text now names plotting (user decision). `rasterio`, `networkx`, `scikit-learn` removed as direct dependencies; `scikit-learn` still comes in through tsam. Local suite: 367 passed, 2 skipped.
- **Left for later:** `.github/renovate.json:31` still lists `scikit-learn` and `rasterio` (no-op); stale "spatial aggregation" wording in `xarrayIO.py:24` and `dictIO.py:75` (PR 4 touches both).

#### PR 4 – `refactor(io): remove useProcessedValues`

- **Goal:** Remove the special path that exports processed values as input.
- **Changes:** `fine/IOManagement/dictIO.py:32`, `:69-70`; `fine/IOManagement/xarrayIO.py:15-32`; the test using `useProcessedValues=True` in `test/IOManagement/test_xarrayio.py:766`.
- **Acceptance:** I/O tests green.
- **Needs:** PR 3 (the only caller with `True` under `fine/` is `energySystemModel.py:941`).

### Block B – Error type and pure checks

Common rule for PR 6–10: change only signatures and call sites. A function receives the values it needs (e.g. `locations`, `investmentPeriods`, `investmentPeriodNames`) as individual arguments and returns its result. No assignment to `comp` or `esM`. No new module, no context object.

Common acceptance for PR 6–10: LP files identical (PR 1), negative tests green (PR 2).

Order: PR 7 first, then 6, 8 and 9, PR 10 last. The reason is calls across PR boundaries:

- PR 6 calls PR 7: `checkAndSetCostParameter` → `checkRegionalIndex`, `processCommodityCost` → `checkAndSetTimeSeries`
- PR 8 calls PR 7: `checkLocationSpecficDesignInputParams` → `checkRegionalIndex`
- PR 10 calls PR 7 and 8: `checkAndSetCommodityConversionFactor` → `checkRegionalIndex`, `checkAndSetTimeSeriesConversionFactors`; `checkAndSetMaintenanceParameters` → `processBoundParams`

To clarify before PR 7: one worked before/after example of a signature as a pattern, and whether `esM.verboseLogLevel` is passed through as an argument (five functions read it).

#### PR 5 – `feat(utils): add structured input error`

- **Goal:** One error type with component name and field path that collects independent errors.
- **Changes:** `InputValidationError` as a subclass of `ValueError` and a collector with field context in `fine/utils.py`. No callers yet.
- **Acceptance:** Tests for collecting, field path and message text.
- **Needs:** nothing.
- **Note:** Existing tests with `pytest.raises(TypeError)` break as soon as a collector catches `TypeError`. This becomes visible per family in PR 11–15; the tests break there on purpose and are adapted there.

#### PR 6 – `refactor(utils): make cost and period checks pure`

- `checkAndSetInvestmentPeriodParameters` (`:681`)
- `checkAndSetInterestRate` (`:869`)
- `checkAndSetCostParameter` (`:1436`)
- `checkAndSetInvestmentPeriodCostParameter` (`:1623`, 10 calls)
- `processCommodityCost` (`:1657`)
- `checkAndSetFullLoadHoursParameter` (`:1864`)
- `checkAndSetPartLoadMin` (`:1519`; calls `checkAndSetInvestmentPeriodParameters`, therefore here instead of in PR 7)

#### PR 7 – `refactor(utils): make time series checks pure`

- `checkAndSetInvestmentPeriodTimeSeries` (`:1214`, 14 calls)
- `checkAndSetTimeSeries` (`:1245`)
- `checkAndSetTimeSeriesConversionFactors` (`:1715`)
- `checkTimeSeriesIndex` (`:132`), `checkRegionalColumnTitles` (`:142`), `checkRegionalIndex` (`:196`)

#### PR 8 – `refactor(utils): make design, bound, eligibility and stock checks pure`

- `checkLocationSpecficDesignInputParams` (`:399`, 191 lines, takes `comp` and `esM`)
- `checkDesignVariableModelingParameters` (`:1376`)
- `checkAndSetBounds` (`:612`), `processBoundParams` (`:592`)
- `setLocationalEligibility` (`:1072`, 140 lines)
- `checkAndSetStock` (`:2438`, 149 lines, takes `component` and `esM`)
- `setStockCapacityStartYear` (`:2589`, takes `component` and `esM`)

Largest PR of the block. If too large: split stock off into its own PR.

**To decide before the PR:** `checkLocationSpecficDesignInputParams` reads other components via `esM.getComponent` and `esM.sharedPotentialDict`. That is a model-wide check and cannot be made pure by passing individual values. Decide whether this part is extracted and where it goes.

#### PR 9 – `refactor(utils): make 2d checks pure`

- `checkAndSetDistances` (`:303`)
- `process2dimCapacityData` (`:2221`)
- `preprocess2dimInvestmentPeriodData` (`:2229`)

`preprocess2dimData` (`:2264`) and `checkAndSetTransmissionLosses` (`:326`) already take no `esM`.

#### PR 10 – `refactor(utils): make conversion and subclass checks pure`

- `checkCommodities` (`:244`), `checkCommodityUnits` (`:257`), `checkRampRates` (`:883`)
- `checkConversionFactorProperties` (`:2706`), `checkAndSetCommodityConversionFactor` (`:2822`, 226 lines), `checkEmissionFactors` (`:3050`), `checkAndSetFlowShares` (`:3120`) – all four take `comp` and `esM`
- `checkConversionDynamicSpecficDesignInputParams` (`:905`), `checkAndSetMaintenanceParameters` (`:982`, takes `comp` and `esM`)
- piecewise linear cost function in `fine/utilsPWLCF.py` and `fine/expansionModules/piecewiseLinearCostFunction.py`, which today receives the whole component

This is where the prototype stopped. PR 18 depends on these being done.

### Block C – One input object per family

Common content for PR 11–15:

- one frozen dataclass object with all normalised fields of the family, listed explicitly, not inherited from a common base
- one build function that is the only place that normalises and uses the collector from PR 5
- the old attribute names stay as read-only properties on the object, no copies
- the constructor calls the build function at the end

Common acceptance:

- LP files identical, negative tests green
- eligibility is computed exactly once
- period fields are never a bare `None`, but `{ip: None}`
- existing attribute names keep their meaning; raw input gets new names
- no attribute is overwritten after construction

#### PR 11 – `refactor(sourceSink): single normalized input object`

- **Affects:** `fine/sourceSink.py`, common part in `fine/component.py`.
- **Additionally to decide:** relationship between full, aggregated and effective time series. Today `full*` → `aggregated*` → `processed*` applies; time series aggregation writes after construction (`setAggregatedTimeSeriesData`). Proposal: the input object holds only the full time series. The aggregated ones live in a separate state of the component. `processed*` becomes a property that returns one or the other depending on `hasTSA`.
- **Additionally to decide:** handling of mutable dicts and Series in the frozen object (copy on build or a documented convention).

#### PR 12 – `refactor(conversion): single normalized input object`

- **Affects:** `fine/conversion.py`.
- **Caution:** The check `operationRateMin <= operationRateMax` (`fine/conversion.py:344`) must not get lost; it was missing in the prototype.

#### PR 13 – `refactor(storage): single normalized input object`

- **Affects:** `fine/storage.py` (22 attributes, the most of all families).

#### PR 14 – `refactor(transmission): single normalized input object`

- **Affects:** `fine/transmission.py`.
- **Special case:** The distance-weighted costs (`:362-401`) live in the object and are not overwritten afterwards. `losses` and `distances` keep their current meaning.
- **Needed on the side:** replace the class name check `str(type(self))[-14:-2] != "Transmission"` in `fine/component.py:642` with a property of the family; it does not match `LinearOptimalPowerFlow`.

#### PR 15 – `refactor(subclasses): single normalized input object`

- **Affects:** `fine/subclasses/conversionDynamic.py`, `conversionPartLoad.py`, `lopf.py`.
- **Special case:** The subclass parameters (e.g. `downTimeMin`, `nSegments`, `reactances`) belong in the field list of their family.

#### Optional follow-up PR – `refactor: read input object directly`

Results, expansion modules and I/O read the object directly instead of via the properties. Not a must; the properties can stay.

### Block D – Strictness and I/O

#### PR 16 – `fix(utils): warn on silently repaired input`

- **Goal:** Every silent repair becomes visible without changing behaviour.
- **Cases:**
  - NaN becomes 0 in 2D data (`fine/utils.py:2272-2273`); already warns today with `UserWarning`, only the warning class changes here
  - Diagonal is adjusted automatically (`:2296`); already warns today with `UserWarning`, only the warning class changes here
  - Zero capacities are discarded (`discard=True`, `:2264`; #122); silent today
  - Filling with 0 after `reindex` (`:239`, `:645`); silent today
- **To decide:** meaning of NaN in `capacityMax` (today "unbounded" or error).
- **Warning class:** `FutureWarning`. Python does not show `DeprecationWarning` outside `__main__` by default.
- **Acceptance:** one test per case for the warning; LP unchanged.
- **Needs:** PR 5.

#### Follow-up PR one release later – `fix(utils)!: reject invalid input instead of repairing it`

Turns the warnings from PR 16 into errors. Closes #122. The LP references of affected fixtures are regenerated deliberately.

#### PR 17 – `refactor(io): use type ids and explicit field lists`

- **Goal:** I/O no longer depends on class names and constructor signatures.
- **Changes:** in `fine/IOManagement/dictIO.py` a dict from type ID to class instead of `getattr(fn, classname)` (`:61`, `:220`); field list from the dataclasses instead of `inspect.getfullargspec` (`:41`); `schemaVersion` in the export.
- **Requirements:** fixed field order; subclass parameters included; no filtering via `hasattr`.
- **Acceptance:** files without `schemaVersion` load as before; export and import without data loss.
- **Needs:** PR 11–15.
- **Out of scope:** The structure of the xarray datasets belongs to #768, Zarr to #803.

### Block E – Public API

#### PR 18 – `feat(component): allow construction without esM`

- **Goal:** A component can be defined without a model and added to several models.
- **Changes:** `esM` in the constructor becomes optional. Without `esM` the constructor stores the input and runs only local checks; `esM.add()` (`fine/energySystemModel.py:537`) calls the build function. A small frozen system context is introduced here that contains exactly the fields the build functions read.
- **Needed before the PR:** inventory of the direct `esM` accesses in the constructors that do not go through `utils` (e.g. `esM.investmentPeriods` in `fine/transmission.py:366`). Block B does not cover them.
- **Not in the context:** quantities that change through time series aggregation (`periodsOrder`, `timeStepsPerPeriod`).
- **Acceptance:** Old calls with `esM` unchanged, also positional. One component usable in two models. A failed `add` leaves the model unchanged.
- **Needs:** PR 10–15.

#### PR 19 – `feat(esM): add replaceComponent`

- **Goal:** Checked, atomic replacement of a component.
- **Changes:** new method in `fine/energySystemModel.py`; `updateComponent` (`:648`) is marked as deprecated.
- **Two legacy issues in `updateComponent`:** The subclass list contains `'ConversionPartLoadModel'` instead of `'ConversionPartLoad'` (`:685`). Unknown keys in `updateAttrs` are not rejected.
- **Acceptance:** A failed replacement leaves the old component in the model.
- **Needs:** PR 18.

#### PR 20 – `feat: unify period keys`

- **Finding:** For normal parameters `develop` already requires calendar years: `checkInvestmentPeriodParameters` (`fine/utils.py:667`) compares dict keys with `esM.investmentPeriodNames`. The PR is therefore smaller than assumed in the old plan.
- **Goal:** Inventory of all places with period keys, in particular stock years (`stockCommissioning`), and exactly one conversion year ↔ index that raises an error for unknown years.
- **Acceptance:** An unknown year results in an error with a field path.
- **Needs:** PR 11–15.

### Block F – Transmission

#### PR 21 – `refactor(transmission): model edges instead of doubled arcs`

- **Goal:** Capacity and fixed costs belong to the physical line, operation to directed arcs.
- **Changes:** factor 0.5 in `fine/transmission.py:362-401` disappears; Pyomo sets for edges and arcs.
- **To decide:** input form (symmetric matrix as today or pair list) and the keys of the connections. Today they are strings of the form `loc1_loc2`, which are ambiguous for location names containing `_`.
- **Acceptance:** objective values equal within tolerance; LP references regenerated deliberately.
- **Needs:** PR 14.

## 7. Order and dependencies

```text
1, 2 ──────────────┐
3 → 4              │
5 ─────────────────┼→ 7 → 6, 8, 9 → 10 → 11 → 12, 13, 14, 15 ─┬→ 17
                   │                                           ├→ 18 → 19
5 → 16 → (follow-up PR after one release)                      ├→ 20
                                                      14 ──────┴→ 21
```

- PR 1 and 2 come before anything that could change behaviour.
- PR 3 and 4 are independent of the rest.
- In block B, PR 7 comes first and PR 10 last (after 7 and 8). PR 6, 8 and 9 are independent of each other.
- PR 11 comes before 12–15, because the time series question is decided there.
- Order of the families by simplicity: source/sink, conversion, storage, transmission, subclasses.

## 8. Impact on users

| Block | Visible to users |
|---|---|
| A–C | nothing, except that spagat is gone (PR 3) and error messages arrive collected (from PR 11) |
| D | first warnings, one release later errors for inputs that were silently repaired so far; export files carry `schemaVersion` |
| E | new possibilities without a break; `updateComponent` is marked as deprecated |
| F | changed formulation of transmission |

## 9. Lessons from `fine3_working`

The branch was read completely (six areas, every statement cross-checked, nothing executed). It is an unfinished layer on top of the old attribute state. What it provides as a warning:

- **Parallel paths diverge.** Next to `utils.py`, rewritten variants with deviating behaviour appeared (different eligibility algorithm, different defaults). Therefore: rebuild in place.
- **A snapshot that gets copied is not a source.** The frozen object was built, around 45 values were copied onto `self.*`, and after that nobody read it. Therefore: properties instead of copies.
- **Only the common part was moved.** Time series, conversion factors, storage parameters, TSA and subclasses stayed on the old path. Therefore: fields per family and separate PRs for the hard cases (PR 10, PR 15).
- **Checks get lost when rewriting.** The conversion check min ≤ max was missing. Therefore: PR 2.
- **New silent repairs appear easily.** NaN in bounds became 0 instead of "unbounded". Therefore: PR 16 with an explicit decision per case.
- **Eligibility was computed two to three times**, with two algorithms. Hence the acceptance criterion in block C.
- **A context object grows.** Of 14 fields, 7 were read. Therefore: only in PR 18 and only with fields that are read.

## 10. Open decisions

| No. | Decision | Due before |
|---|---|---|
| 3 | Pattern signature for pure check functions; handling of `verboseLogLevel` | PR 7 |
| 4 | Model-wide part of `checkLocationSpecficDesignInputParams` (shared potential): where does it go? | PR 8 |
| 5 | Scope of the piecewise linear cost function (`utilsPWLCF.py`) | PR 10 |
| 6 | Time series in the frozen object: accept the proposal from PR 11? | PR 11 |
| 7 | Mutable dicts and Series in the frozen object | PR 11 |
| 8 | Module, names and field lists of the input objects; new names for the raw input | PR 11 |
| 9 | `TypeError` or uniformly `InputValidationError` for input errors | PR 11 |
| 10 | Meaning of NaN in `capacityMax` | PR 16 |
| 11 | How long do the warnings from PR 16 run before they become errors? | PR 16 |
| 12 | Name `replaceComponent` | PR 19 |
| 13 | Input form and keys for transmission connections | PR 21 |
| 14 | Re-evaluate parameter bundles per function (section 3) | after block C |

### Decided on 2026-11-10

- All PRs target `develop`; `fine_refactor` holds only the plan. Decision 2 (merging `fine_refactor` back) is therefore dropped.
- The plan and all notes are written in English.
- The stability of the LP output is clarified in PR 1, not beforehand.
- Negative tests (PR 2) assert today's exception type and break on purpose in PR 11–15.
- PR 3: the example notebook 07 is removed. `geopandas` stays for plotting. spagat is removed without a warning release.
- If an LP file changes, work stops and the cause is investigated (section 0).
- Decision 1 (PR 1): the LP files are compared byte-exactly, written with `file_determinism=20`. `multi_node_test_esM_init` is dropped from the snapshot set; `balanceLimitConstraint_test_esM` is stored gzip-compressed.

## 11. Not verified

- The assignment of the `utils` functions to PR 6–10 is based on signatures, callers (Appendix A) and the calls between the functions, not on fully read function bodies.
- The piecewise linear cost function was not read (PR 10).
- The handling of stock years as keys (PR 20).
- The test suite was not run as part of this planning. According to information from 2026-11-10 it is green on `develop`.

## Appendix A – Input functions in `fine/utils.py`

Column "Context": arguments that pass a model or a component. Empty means: already without. Column "Callers": files under `fine/` outside `utils.py` with the number of calls; "internal" means: called only within `utils.py` or not at all.

| Function | Line | Context | Callers | PR |
|---|---|---|---|---|
| `checkCommodities` | 244 | `esM` | sourceSink 1, storage 1, transmission 1, conversionPartLoad 1 | 10 |
| `checkCommodityUnits` | 257 | `esM` | conversion 1 | 10 |
| `checkTimeSeriesIndex` | 132 | `esM` | internal | 7 |
| `checkRegionalColumnTitles` | 142 | `esM` | internal | 7 |
| `checkRegionalIndex` | 196 | `esM` | internal | 7 |
| `checkAndSetDistances` | 303 | `esM` | transmission 1 | 9 |
| `checkLocationSpecficDesignInputParams` | 399 | `comp`, `esM` | component 1 | 8 |
| `processBoundParams` | 592 | `esM` | internal | 8 |
| `checkAndSetBounds` | 612 | `esM` | component 2 | 8 |
| `checkAndSetInvestmentPeriodParameters` | 681 | `esM` | component 1 | 6 |
| `checkAndSetInterestRate` | 869 | `esM` | component 1 | 6 |
| `checkRampRates` | 883 | `esM` | conversion 1 | 10 |
| `checkConversionDynamicSpecficDesignInputParams` | 905 | `esM` | conversionDynamic 1 | 10 |
| `checkAndSetMaintenanceParameters` | 982 | `comp`, `esM` | conversionDynamic 1 | 10 |
| `setLocationalEligibility` | 1072 | `esM` | conversion 1, sourceSink 1, storage 1, transmission 1 | 8 |
| `checkAndSetInvestmentPeriodTimeSeries` | 1214 | `esM` | conversion 3, sourceSink 3, storage 6, transmission 2 | 7 |
| `checkAndSetTimeSeries` | 1245 | `esM` | internal | 7 |
| `checkDesignVariableModelingParameters` | 1376 | `esM` | component 1 | 8 |
| `checkAndSetCostParameter` | 1436 | `esM` | component 2 | 6 |
| `checkAndSetPartLoadMin` | 1519 | `esM` | conversion 1, sourceSink 1, storage 1, transmission 1 | 6 |
| `checkAndSetInvestmentPeriodCostParameter` | 1623 | `esM` | component 5, conversion 1, sourceSink 1, storage 2, transmission 1 | 6 |
| `processCommodityCost` | 1657 | `esM` | sourceSink 2 | 6 |
| `checkAndSetTimeSeriesConversionFactors` | 1715 | `esM` | internal | 7 |
| `checkAndSetFullLoadHoursParameter` | 1864 | `esM` | component 2 | 6 |
| `process2dimCapacityData` | 2221 | `esM` | transmission 2 | 9 |
| `preprocess2dimInvestmentPeriodData` | 2229 | `esM` | transmission 4 | 9 |
| `checkAndSetStock` | 2438 | `component`, `esM` | component 1 | 8 |
| `setStockCapacityStartYear` | 2589 | `component`, `esM` | component 1 | 8 |
| `checkConversionFactorProperties` | 2706 | `comp`, `esM` | conversion 1 | 10 |
| `checkAndSetCommodityConversionFactor` | 2822 | `comp`, `esM` | conversion 1 | 10 |
| `checkEmissionFactors` | 3050 | `comp`, `esM` | conversion 1 | 10 |
| `checkAndSetFlowShares` | 3120 | `comp`, `esM` | conversion 1 | 10 |

Model-wide functions with `esM` that are not part of block B: `checkAndSetPathwayBalanceLimit` (1794), `checkAndSetBalanceLimit` (1809), `checkSinkCompCO2toEnvironment` (2626).

Already without context and called by components (no change needed): `checkAndSetBalanceLimitID`, `checkCapacityOrCommissioningTransmission`, `isInRange`, `isString`, `isStrictlyPositiveNumber`, `isPositiveNumber`, `sortTimeSeriesColumns`, `checkAndSetTransmissionLosses`, `checkCapacityDevelopmentWithStock`, `checkOperationRateForCapacityVariable`, `checkEconomicAndTechnicalLifetime`, `checkFlooringParameter`, `preprocess2dimData`, `checkNumberOfConversionFactors`, `checkStockYears`, `checkSimultaneousChargeDischarge`.

## Appendix B – Relation to existing issues

| Issue | Topic | Relation |
|---|---|---|
| #806 | remove spagat (issue, open) | PR 3 |
| #122 | zero capacities in transmission are discarded | PR 16 and follow-up PR |
| #768 | restructure xarray dataset format | builds on PR 17; not part of this plan |
| #803 | Zarr as storage format | builds on PR 17; not part of this plan |
| #786 | reconstruct `_rawResults` on netCDF load | results side; independent |
| #148 | flexible limit for location selection | independent |
| PR #791 | refactoring of `convertOptimizationOutputToDatasets` | results side; independent |
