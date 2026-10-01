"""Central definitions for string-based FINE options."""

from enum import Enum, unique


class FineEnum(str, Enum):
    """String-backed enum that keeps legacy string behavior."""

    def __str__(self):
        return self.value


@unique
class ComponentAbbreviation(FineEnum):
    """Abbreviations used in component model names and serialized output."""

    STORAGE = "stor"
    CONVERSION = "conv"
    TRANSMISSION = "trans"
    SOURCE_SINK = "srcSnk"
    LOPF = "lopf"
    PART_LOAD = "partLoad"
    CONVERSION_DYNAMIC = "conv_dyn"
    PWLCF = "pwlcf"


@unique
class Dimension(FineEnum):
    """Supported dimensionality labels for component data."""

    ONE = "1dim"
    TWO = "2dim"


@unique
class VarType(FineEnum):
    """Optimization variable categories used for result formatting."""

    DESIGN = "designVariables"
    OPERATION = "operationVariables"


@unique
class CostType(FineEnum):
    """Cost result types used in economic contribution calculations."""

    TAC = "TAC"
    NPV = "NPV"


@unique
class FncType(FineEnum):
    """Function input types for operation-dependent economic calculations."""

    TD = "TD"
    TIME_SERIES = "TimeSeries"


@unique
class RampingType(FineEnum):
    """Ramping constraint parameter names."""

    DOWN_MAX = "rampDownMax"
    UP_MAX = "rampUpMax"
