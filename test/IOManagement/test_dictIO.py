import pytest

import fine as fn


def test_export_to_dict_minimal(minimal_test_esM):
    # EXPECTED
    expected_esm_dict = dict(
        zip(
            (
                "locations",
                "commodities",
                "stochasticModel",
                "commodityUnitsDict",
                "numberOfTimeSteps",
                "hoursPerTimeStep",
                "startYear",
                "numberOfInvestmentPeriods",
                "investmentPeriodInterval",
                "costUnit",
                "lengthUnit",
                "verboseLogLevel",
                "balanceLimit",
                "pathwayBalanceLimit",
                "annuityPerpetuity",
            ),
            (
                minimal_test_esM.locations,
                minimal_test_esM.commodities,
                minimal_test_esM.stochasticModel,
                minimal_test_esM.commodityUnitsDict,
                minimal_test_esM.numberOfTimeSteps,
                minimal_test_esM.hoursPerTimeStep,
                minimal_test_esM.startYear,
                minimal_test_esM.numberOfInvestmentPeriods,
                minimal_test_esM.investmentPeriodInterval,
                minimal_test_esM.costUnit,
                minimal_test_esM.lengthUnit,
                minimal_test_esM.verboseLogLevel,
                minimal_test_esM.balanceLimit,
                minimal_test_esM.pathwayBalanceLimit,
                minimal_test_esM.annuityPerpetuity,
            ),
        )
    )

    expected_Electrolyzers_investPerCapacity = minimal_test_esM.getComponentAttribute(
        "Electrolyzers", "investPerCapacity"
    )
    expected_Electricitymarket_operationRateMax = (
        minimal_test_esM.getComponentAttribute("Electricity market", "operationRateMax")
    )
    expected_Industrysite_operationRateFix = minimal_test_esM.getComponentAttribute(
        "Industry site", "operationRateFix"
    )

    # FUNCTION CALL
    output_esm_dict, output_comp_dict = fn.dictIO.exportToDict(minimal_test_esM)

    output_Conversion_investPerCapacity = (
        output_comp_dict.get("Conversion").get("Electrolyzers").get("investPerCapacity")
    )
    output_Source_operationRateMax = (
        output_comp_dict.get("Source").get("Electricity market").get("operationRateMax")
    )
    output_Sink_operationRateFix = (
        output_comp_dict.get("Sink").get("Industry site").get("operationRateFix")
    )

    # ASSERTION
    assert output_esm_dict == expected_esm_dict
    assert (
        expected_Electrolyzers_investPerCapacity == output_Conversion_investPerCapacity
    )
    assert expected_Electricitymarket_operationRateMax.equals(
        output_Source_operationRateMax
    )
    assert expected_Industrysite_operationRateFix.equals(output_Sink_operationRateFix)


def test_export_to_dict_singlenode(single_node_test_esM):
    # EXPECTED
    expected_esm_dict = dict(
        zip(
            (
                "locations",
                "commodities",
                "stochasticModel",
                "commodityUnitsDict",
                "numberOfTimeSteps",
                "hoursPerTimeStep",
                "startYear",
                "numberOfInvestmentPeriods",
                "investmentPeriodInterval",
                "costUnit",
                "lengthUnit",
                "verboseLogLevel",
                "balanceLimit",
                "pathwayBalanceLimit",
                "annuityPerpetuity",
            ),
            (
                single_node_test_esM.locations,
                single_node_test_esM.commodities,
                single_node_test_esM.stochasticModel,
                single_node_test_esM.commodityUnitsDict,
                single_node_test_esM.numberOfTimeSteps,
                single_node_test_esM.hoursPerTimeStep,
                single_node_test_esM.startYear,
                single_node_test_esM.numberOfInvestmentPeriods,
                single_node_test_esM.investmentPeriodInterval,
                single_node_test_esM.costUnit,
                single_node_test_esM.lengthUnit,
                single_node_test_esM.verboseLogLevel,
                single_node_test_esM.balanceLimit,
                single_node_test_esM.pathwayBalanceLimit,
                single_node_test_esM.annuityPerpetuity,
            ),
        )
    )

    expected_Electrolyzers_investPerCapacity = (
        single_node_test_esM.getComponentAttribute("Electrolyzers", "investPerCapacity")
    )
    expected_Electricitymarket_operationRateMax = (
        single_node_test_esM.getComponentAttribute(
            "Electricity market", "operationRateMax"
        )
    )
    expected_Industrysite_operationRateFix = single_node_test_esM.getComponentAttribute(
        "Industry site", "operationRateFix"
    )

    # FUNCTION CALL
    output_esm_dict, output_comp_dict = fn.dictIO.exportToDict(single_node_test_esM)

    output_Conversion_investPerCapacity = (
        output_comp_dict.get("Conversion").get("Electrolyzers").get("investPerCapacity")
    )
    output_Source_operationRateMax = (
        output_comp_dict.get("Source").get("Electricity market").get("operationRateMax")
    )
    output_Sink_operationRateFix = (
        output_comp_dict.get("Sink").get("Industry site").get("operationRateFix")
    )

    # ASSERTION
    assert output_esm_dict == expected_esm_dict
    assert (
        expected_Electrolyzers_investPerCapacity == output_Conversion_investPerCapacity
    )
    assert expected_Electricitymarket_operationRateMax.equals(
        output_Source_operationRateMax
    )
    assert expected_Industrysite_operationRateFix.equals(output_Sink_operationRateFix)


def test_export_to_dict_multinode(multi_node_test_esM_init):
    # EXPECTED
    expected_esm_dict = dict(
        zip(
            (
                "locations",
                "commodities",
                "stochasticModel",
                "commodityUnitsDict",
                "numberOfTimeSteps",
                "hoursPerTimeStep",
                "numberOfInvestmentPeriods",
                "investmentPeriodInterval",
                "startYear",
                "costUnit",
                "lengthUnit",
                "verboseLogLevel",
                "balanceLimit",
                "pathwayBalanceLimit",
                "annuityPerpetuity",
            ),
            (
                multi_node_test_esM_init.locations,
                multi_node_test_esM_init.commodities,
                multi_node_test_esM_init.stochasticModel,
                multi_node_test_esM_init.commodityUnitsDict,
                multi_node_test_esM_init.numberOfTimeSteps,
                multi_node_test_esM_init.hoursPerTimeStep,
                multi_node_test_esM_init.numberOfInvestmentPeriods,
                multi_node_test_esM_init.investmentPeriodInterval,
                multi_node_test_esM_init.startYear,
                multi_node_test_esM_init.costUnit,
                multi_node_test_esM_init.lengthUnit,
                multi_node_test_esM_init.verboseLogLevel,
                multi_node_test_esM_init.balanceLimit,
                multi_node_test_esM_init.pathwayBalanceLimit,
                multi_node_test_esM_init.annuityPerpetuity,
            ),
        )
    )

    expected_Windonshore_operationRateMax = (
        multi_node_test_esM_init.getComponentAttribute(
            "Wind (onshore)", "operationRateMax"
        )
    )
    expected_CCGTplantsmethane_investPerCapacity = (
        multi_node_test_esM_init.getComponentAttribute(
            "CCGT plants (methane)", "investPerCapacity"
        )
    )
    expected_Saltcavernshydrogen_capacityMax = (
        multi_node_test_esM_init.getComponentAttribute(
            "Salt caverns (hydrogen)", "capacityMax"
        )
    )
    expected_ACcables_reactances = multi_node_test_esM_init.getComponentAttribute(
        "AC cables", "reactances"
    )
    expected_Hydrogendemand_operationRateFix = (
        multi_node_test_esM_init.getComponentAttribute(
            "Hydrogen demand", "operationRateFix"
        )
    )

    # FUNCTION CALL
    output_esm_dict, output_comp_dict = fn.dictIO.exportToDict(multi_node_test_esM_init)

    output_Windonshore_operationRateMax = (
        output_comp_dict.get("Source").get("Wind (onshore)").get("operationRateMax")
    )
    output_CCGTplantsmethane_investPerCapacity = (
        output_comp_dict.get("Conversion")
        .get("CCGT plants (methane)")
        .get("investPerCapacity")
    )
    output_Saltcavernshydrogen_capacityMax = (
        output_comp_dict.get("Storage")
        .get("Salt caverns (hydrogen)")
        .get("capacityMax")
    )
    output_ACcables_reactances = (
        output_comp_dict.get("LinearOptimalPowerFlow")
        .get("AC cables")
        .get("reactances")
    )
    output_Hydrogendemand_operationRateFix = (
        output_comp_dict.get("Sink").get("Hydrogen demand").get("operationRateFix")
    )

    # ASSERTION
    assert output_esm_dict == expected_esm_dict
    assert expected_Windonshore_operationRateMax.equals(
        output_Windonshore_operationRateMax
    )
    assert (
        expected_CCGTplantsmethane_investPerCapacity
        == output_CCGTplantsmethane_investPerCapacity
    )
    assert expected_Saltcavernshydrogen_capacityMax.equals(
        output_Saltcavernshydrogen_capacityMax
    )
    assert expected_ACcables_reactances.equals(output_ACcables_reactances)
    assert expected_Hydrogendemand_operationRateFix.equals(
        output_Hydrogendemand_operationRateFix
    )


@pytest.mark.parametrize(
    "test_esM_fixture", ["minimal_test_esM", "multi_node_test_esM_init"]
)
def test_import_from_dict(test_esM_fixture, request):
    test_esM = request.getfixturevalue(test_esM_fixture)

    # FUNCTION CALL
    ## get dicts
    esm_dict, comp_dict = fn.dictIO.exportToDict(test_esM)
    ## call the function on dicts
    output_esM = fn.dictIO.importFromDict(esm_dict, comp_dict)

    # EXPECTED (AND OUTPUT)
    expected_locations = test_esM.locations
    expected_commodityUnitsDict = test_esM.commodityUnitsDict

    if test_esM_fixture == "minimal_test_esM":
        ## expected
        expected_df = test_esM.getComponentAttribute(
            "Electricity market", "operationRateMax"
        )
        expected_series = None
        expected_value = test_esM.getComponentAttribute(
            "Electrolyzers", "investPerCapacity"
        )
        ## output
        output_df = output_esM.getComponentAttribute(
            "Electricity market", "operationRateMax"
        )
        output_df.reset_index(level=0, drop=True, inplace=True)

        output_value = output_esM.getComponentAttribute(
            "Electrolyzers", "investPerCapacity"
        )
        output_series = None

    else:
        ## expected
        expected_df = test_esM.getComponentAttribute(
            "Hydrogen demand", "operationRateFix"
        )
        expected_series = test_esM.getComponentAttribute(
            "AC cables", "reactances"
        ).sort_index()
        expected_value = test_esM.getComponentAttribute(
            "Existing run-of-river plants", "investPerCapacity"
        )
        ## output
        output_df = output_esM.getComponentAttribute(
            "Hydrogen demand", "operationRateFix"
        )
        output_df.reset_index(level=0, drop=True, inplace=True)

        output_series = output_esM.getComponentAttribute(
            "AC cables", "reactances"
        ).sort_index()
        output_value = output_esM.getComponentAttribute(
            "Existing run-of-river plants", "investPerCapacity"
        )
        assert output_series.equals(expected_series)

    # ASSERTION
    assert output_esM.locations == expected_locations
    assert output_esM.commodityUnitsDict == expected_commodityUnitsDict

    assert output_df.equals(expected_df)
    assert output_value == expected_value
