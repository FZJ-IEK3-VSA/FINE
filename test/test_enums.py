from fine.enums import (
    ComponentAbbreviation,
    CostType,
    Dimension,
    FncType,
    RampingType,
    VarType,
)


def test_enums_keep_legacy_string_values():
    assert ComponentAbbreviation.STORAGE == "stor"
    assert ComponentAbbreviation.CONVERSION == "conv"
    assert ComponentAbbreviation.TRANSMISSION == "trans"
    assert ComponentAbbreviation.SOURCE_SINK == "srcSnk"
    assert ComponentAbbreviation.LOPF == "lopf"
    assert ComponentAbbreviation.PART_LOAD == "partLoad"
    assert ComponentAbbreviation.CONVERSION_DYNAMIC == "conv_dyn"
    assert ComponentAbbreviation.PWLCF == "pwlcf"

    assert Dimension.ONE == "1dim"
    assert Dimension.TWO == "2dim"
    assert VarType.DESIGN == "designVariables"
    assert VarType.OPERATION == "operationVariables"
    assert CostType.TAC == "TAC"
    assert CostType.NPV == "NPV"
    assert FncType.TD == "TD"
    assert FncType.TIME_SERIES == "TimeSeries"
    assert RampingType.DOWN_MAX == "rampDownMax"
    assert RampingType.UP_MAX == "rampUpMax"


def test_enums_preserve_string_operations():
    assert str(Dimension.ONE) == "1dim"
    assert "operationVarSet_" + ComponentAbbreviation.CONVERSION == (
        "operationVarSet_conv"
    )
    assert f"ConstrInterPeriod_{RampingType.UP_MAX}_conv" == (
        "ConstrInterPeriod_rampUpMax_conv"
    )


def test_component_abbreviations_are_unique():
    values = [member.value for member in ComponentAbbreviation]
    assert len(values) == len(set(values))
