"""Assembly of the optimization summary.

The summary is a *view* of the frames the result pipeline has already produced: the raw
solved design variables from ``extractRawResults`` and the cost frames from
``deriveEconomics``. Nothing is computed here that is not already in those frames, and
nothing downstream may derive values from the summary.

Kept free of pyomo and of the component model, so it can be exercised on a hand-built
raw-results dict.
"""

import warnings

import pandas as pd

from fine.results.frames import economicSummaryUnits


def buildOptimizationSummary(
    componentsDict,
    rawResults,
    rawResults1dim,
    esM,
    indexColumns,
    plantUnit,
    unitApp="",
):
    r"""Assemble the optimization summary from the already computed result frames.

    Writes the frames into a ``(Component, Property, Unit) x indexColumns`` DataFrame per
    investment period. It also emits the big-M proximity warning for capacities close to
    the chosen big-M value.

    **Required arguments**

    :param componentsDict: the modeling class' components, by name.
    :type componentsDict: dict

    :param rawResults: raw and derived frames per investment period name, as produced by
        ``extractRawResults`` and ``deriveEconomics``.
    :type rawResults: dict

    :param rawResults1dim: the 1-dim companions of the raw design frames.
    :type rawResults1dim: dict

    :param esM: EnergySystemModel instance representing the energy system in which the
        components are modeled.
    :type esM: EnergySystemModel instance

    :param indexColumns: set of strings with the column indices of the summary (locations
        or connections between locations).
    :type indexColumns: set

    :param plantUnit: attribute of the component that describes the unit of the plants
        (e.g. "commodityUnit" or "physicalUnit").
    :type plantUnit: string

    **Default arguments**

    :param unitApp: string appended to the capacity unit in the summary (e.g. '\*h' for
        storage). |br| * the default value is ''.
    :type unitApp: string

    :return: summary of the optimized values, keyed by investment period name.
    :rtype: dict
    """
    compDict = componentsDict

    # Single source of truth for the summary's (Property -> Unit) rows. The design rows
    # (capacity/commissioning/decommissioning) carry a per-component plant unit resolved
    # below and are marked with ``None``; every other unit is fixed. The economic units
    # are shared with the export via :func:`fine.results.frames.economicSummaryUnits`. The
    # same mapping drives both the MultiIndex and the economic-frame write loop below.
    summaryUnits = {
        "capacity": None,
        "commissioning": None,
        "decommissioning": None,
        "isBuilt": "[-]",
        **economicSummaryUnits(esM.costUnit),
    }
    # A rolling-horizon run re-discounts NPVcontribution back to the true overall start
    # year: esM.startYear is only the current window's start year, so
    # esM.rollingHorizonStartYear (set once, on the first window) is needed to express the
    # additional discounting distance. getattr, not esM.rollingHorizonStartYear directly:
    # this module is exercised on hand-built esM doubles (see the module docstring) that
    # need not carry every EnergySystemModel attribute.
    rollingHorizonStartYear = getattr(esM, "rollingHorizonStartYear", None)
    if rollingHorizonStartYear is not None:
        summaryUnits["NPVcontributionRH"] = "[" + esM.costUnit + "]"
    # Design rows are written explicitly below (from the 1-dim frames, with their own
    # conditionals); the remaining rows are the economic frames derived by deriveEconomics.
    designProps = ("capacity", "commissioning", "decommissioning", "isBuilt")

    def resolveUnit(compName, prop):
        # ``None`` marks a capacity-like row whose unit is the component's plant unit.
        unit = summaryUnits[prop]
        if unit is None:
            unit = "[" + getattr(compDict[compName], plantUnit) + unitApp + "]"
        return unit

    mIndex = pd.MultiIndex.from_tuples(
        [
            (compName, prop, resolveUnit(compName, prop))
            for compName in compDict.keys()
            for prop in summaryUnits
        ],
        names=["Component", "Property", "Unit"],
    )

    optSummary = {}
    for ip in esM.investmentPeriods:
        ipName = esM.investmentPeriodNames[ip]
        optSummary_ip = pd.DataFrame(
            index=mIndex, columns=sorted(indexColumns)
        ).sort_index()
        # raw + derived economic frames produced by extractRawResults / deriveEconomics
        results_ip = rawResults[ipName]

        # Read raw solved design variables (1-dim) extracted by extractRawResults
        capOptVal = rawResults1dim[ipName]["capacity"]
        commisOptVal = rawResults1dim[ipName]["commissioning"]
        decommisOptVal = rawResults1dim[ipName]["decommissioning"]
        binCapOptVal = rawResults1dim[ipName]["isBuilt"]

        if capOptVal is not None:
            # Check if the installed capacities are close to a bigM val
            # ue for components with design decision variables but
            # ignores cases where bigM was substituted by capacityMax parameter (see bigM constraint
            for compName, comp in compDict.items():
                if (
                    comp.hasIsBuiltBinaryVariable
                    and (comp.processedCapacityMax is None)
                    and capOptVal.loc[compName].max() >= comp.bigM * 0.9
                    and esM.verboseLogLevel < 2
                ):
                    warnings.warn(
                        "the capacity of component "
                        + compName
                        + " is in one or more locations close "
                        + "or equal to the chosen Big M. Consider rerunning the simulation with a higher"
                        + " Big M."
                    )

            # Fill the optimization summary with the optimal capacities.
            optSummary_ip.loc[
                [
                    (
                        ix,
                        "capacity",
                        "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                    )
                    for ix in capOptVal.index
                ],
                capOptVal.columns,
            ] = capOptVal.values

        # Fill the optimization summary with the isBuilt decisions.
        if binCapOptVal is not None:
            optSummary_ip.loc[
                [(ix, "isBuilt", "[-]") for ix in binCapOptVal.index],
                binCapOptVal.columns,
            ] = binCapOptVal.values

        # Get and set optimal values for commissioning and decommissioning
        # not applicable for singleyear optimization, hence dropped from summary
        # either decommissioning or capacity exists
        # (years can have decommissioning, leading to no left capacity)
        if decommisOptVal is not None or capOptVal is not None:
            # commissioning
            optSummary_ip.loc[
                [
                    (
                        ix,
                        "commissioning",
                        "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                    )
                    for ix in commisOptVal.index
                ],
                commisOptVal.columns,
            ] = commisOptVal.values
            # decommissioning
            optSummary_ip.loc[
                [
                    (
                        ix,
                        "decommissioning",
                        "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                    )
                    for ix in decommisOptVal.index
                ],
                decommisOptVal.columns,
            ] = decommisOptVal.values

        # Fill the optimization summary with the derived economic frames (invest,
        # capexCap, opexCap, capexIfBuilt, opexIfBuilt, lifetime corrections, TAC and
        # NPVcontribution) computed by deriveEconomics.
        # The lifetime correction rows are written cell-wise to keep their numpy scalar
        # dtype (as in the former inline implementation).
        perCellProps = (
            "investLifetimeExtension",
            "revenueLifetimeShorteningResale",
        )
        for prop, unit in summaryUnits.items():
            if prop in designProps or prop not in results_ip:
                continue
            frame = results_ip[prop]
            if frame.empty:
                continue
            if prop in perCellProps:
                for component in frame.index:
                    for loc in frame.columns:
                        optSummary_ip.loc[(component, prop, unit), loc] = frame.loc[
                            component, loc
                        ]
            else:
                optSummary_ip.loc[
                    [(ix, prop, unit) for ix in frame.index],
                    frame.columns,
                ] = frame.values

        # The former inline implementation wrote the TAC and NPVcontribution rows as a
        # groupby sum over this summary frame (the base class for TAC, the four component
        # classes for NPVcontribution). That groupby also turned the all-NaN cells of the
        # entries the derived frames do not cover (e.g. a location the component is not
        # eligible in, or a location pair without a transmission connection) into 0. The
        # fold itself now runs on the raw frames in deriveEconomics, which hold the covered
        # entries only, so reproduce the NaN -> 0 normalization here. The 0 is not
        # cosmetic: the transmission summary drops NaN cells when it splits the connection
        # index, so those rows would otherwise lose the uncovered entries.
        # ``where`` rather than ``fillna``, so the object dtype of the summary survives
        # (fillna would downcast it and warn).
        foldedRows = optSummary_ip.index.get_level_values("Property").isin(
            ("TAC", "NPVcontribution")
        )
        folded = optSummary_ip.loc[foldedRows]
        optSummary_ip.loc[foldedRows] = folded.where(folded.notna(), 0)

        # Discount NPVcontribution back to the true overall start year for a rolling
        # horizon run (see the summaryUnits note above for why esM.startYear alone is not
        # enough). Reads the already NaN -> 0 normalized NPVcontribution row above, so the
        # division below never hits a NaN.
        if rollingHorizonStartYear is not None:
            rhExponent = esM.startYear - rollingHorizonStartYear
            unit = "[" + esM.costUnit + "]"
            for compName in compDict:
                interestRate = compDict[compName].interestRate
                for loc in optSummary_ip.columns:
                    npvValue = optSummary_ip.loc[
                        (compName, "NPVcontribution", unit), loc
                    ]
                    optSummary_ip.loc[(compName, "NPVcontributionRH", unit), loc] = (
                        npvValue / (1 + interestRate[loc]) ** rhExponent
                    )

        optSummary[ipName] = optSummary_ip

    return optSummary
