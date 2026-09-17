from __future__ import annotations

import re
from typing import Any

import pandas as pd


class FineUnitError(ValueError):
    """Raised when a unit cannot be interpreted or converted."""


class FineUnitRegistry:
    """ESM-scoped unit registry for ETHOS.FINE.

    Pint is imported lazily and is therefore not required for models that do
    not use the optional ``units`` argument.
    """

    def __init__(self, esM):
        self.commodityUnitsDict = dict(esM.commodityUnitsDict)
        self.costUnit = esM.costUnit
        self.lengthUnit = esM.lengthUnit
        self.timeUnit = esM.timeUnit

        self._ureg = None
        self._pint = None

    # ------------------------------------------------------------------
    # Pint loading
    # ------------------------------------------------------------------

    def _require_pint(self):
        if self._ureg is not None:
            return self._ureg

        try:
            import pint  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Unit-aware inputs require the optional 'pint' dependency. "
                "Install ETHOS.FINE with `pip install fine[units]` or "
                "`pip install -e .[units]`."
            ) from exc

        self._pint = pint
        self._ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

        self._define_fine_units()

        return self._ureg

    def _define_fine_units(self):
        """Register FINE-specific aliases without introducing FX conversion."""
        definitions = (
            "EUR = [currency_EUR]",
            "euro = EUR",
            "Euro = EUR",
            "USD = [currency_USD]",
            "Dollar = USD",
            "Mio = 1e6",
        )

        for definition in definitions:
            try:
                self._ureg.define(definition)
            except Exception:
                # Alias may already exist in the Pint installation.
                pass

    # ------------------------------------------------------------------
    # Legacy FINE unit strings
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_fine_unit_string(unit: str) -> str:
        """Translate common legacy FINE labels into Pint-readable syntax.

        Examples
        --------
        GW_el          -> GW
        GWh_H2_LHV     -> GWh
        kg_H2/h        -> kg/h
        Mio.t_CO2/h    -> 1e6 * t/h
        1e9 Euro       -> 1e9 * EUR

        """
        if not isinstance(unit, str):
            raise TypeError("FINE base units must be strings.")

        value = unit.strip()

        value = value.replace("Mio.", "1e6 * ")
        value = value.replace("Mio ", "1e6 * ")

        value = re.sub(r"\bEuro\b", "EUR", value)
        value = re.sub(r"\bDollar\b", "USD", value)

        # Explicit multiplication between numeric scale and unit.
        value = re.sub(r"(?<=\d)\s+(?=[A-Za-z])", " * ", value)

        # Convert FINE commodity-labelled unit symbols such as GW_el, GWh_H2_LHV, kg_H2, t_CO2.
        tagged_units = (
            "TWh",
            "GWh",
            "MWh",
            "kWh",
            "TW",
            "GW",
            "MW",
            "kW",
            "kg",
            "mg",
            "t",
            "g",
            "L",
        )

        pattern = r"\b(" + "|".join(tagged_units) + r")_[A-Za-z0-9_]+\b"

        value = re.sub(pattern, r"\1", value)

        return value.replace("^", "**")

    def parse_fine_unit(self, unit: str):
        """Return a Pint Unit/Quantity expression for a FINE unit string."""
        ureg = self._require_pint()
        normalized = self._normalize_fine_unit_string(unit)

        try:
            return ureg.parse_expression(normalized)
        except Exception as exc:
            raise FineUnitError(
                f"Could not interpret FINE unit '{unit}'. Normalized representation was '{normalized}'."
            ) from exc

    # ------------------------------------------------------------------
    # Value scaling
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_value(value: Any, factor: float):
        """Scale supported FINE numeric input containers."""
        if value is None or isinstance(value, bool):
            return value

        if isinstance(value, (int, float, pd.Series, pd.DataFrame)):
            return value * factor

        if isinstance(value, dict):
            return {
                key: FineUnitRegistry._scale_value(item, factor)
                for key, item in value.items()
            }

        return value

    def convert(self, value, source_unit, target_unit, parameter_name=None):
        """Convert a FINE parameter while preserving its container type."""
        if value is None:
            return None

        ureg = self._require_pint()

        # Accept pint.Unit objects or strings as source_unit descriptors.
        if self._pint is not None and not isinstance(source_unit, self._pint.Unit):
            # allow strings too
            if not isinstance(source_unit, str):
                raise TypeError(
                    f"Unit for '{parameter_name or 'parameter'}' must be a pint.Unit object or string, not a pint.Quantity or other type."
                )

        source_expression = str(source_unit)

        target_expression = self._normalize_fine_unit_string(target_unit)

        try:
            factor = (
                (1 * ureg.parse_units(source_expression))
                .to(ureg.parse_expression(target_expression))
                .magnitude
            )
        except Exception:
            name = parameter_name or "parameter"

            raise FineUnitError(
                f"Unit '{source_unit}' supplied for '{name}' is not compatible with FINE base unit '{target_unit}'."
            )

        return self._scale_value(value, factor)

    # ------------------------------------------------------------------
    # Canonical target units
    # ------------------------------------------------------------------

    def commodity_unit(self, commodity: str) -> str:
        """Return the commodity's canonical unit string stored in the ESM."""
        try:
            return self.commodityUnitsDict[commodity]
        except KeyError as exc:
            raise FineUnitError(f"Unknown commodity '{commodity}'.") from exc

    def storage_capacity_unit(self, commodity: str) -> str:
        """Return the canonical storage capacity unit for a commodity.

        The storage capacity unit combines the commodity unit with the model's
        time unit (e.g. MWh for electricity when commodity unit is MW and time
        unit is hours).
        """
        return f"({self.commodity_unit(commodity)}) * ({self.timeUnit})"

    def operation_unit(self, physical_unit: str) -> str:
        """Return an operation unit for a given physical unit (per timestep)."""
        return f"({physical_unit}) * ({self.timeUnit})"

    def cost_per_operation_unit(self, physical_unit: str) -> str:
        """Return the unit used for cost per operation (e.g. EUR / (MW * h))."""
        return f"({self.costUnit}) / (({physical_unit}) * ({self.timeUnit}))"

    def cost_per_capacity_unit(
        self, physical_unit: str, distance_dependent: bool = False
    ) -> str:
        """Return the unit used for cost per capacity, optionally distance-dependent.

        If `distance_dependent` is True, the denominator includes the model's
        length unit.
        """
        if distance_dependent:
            return f"({self.costUnit}) / (({self.lengthUnit}) * ({physical_unit}))"

        return f"({self.costUnit}) / ({physical_unit})"

    def cost_if_built_unit(self, distance_dependent: bool = False) -> str:
        """Return the unit for fixed (if-built) costs, optionally per distance."""
        if distance_dependent:
            return f"({self.costUnit}) / ({self.lengthUnit})"

        return self.costUnit

    # ------------------------------------------------------------------
    # Shared Component arguments
    # ------------------------------------------------------------------

    def convert_component_parameters(
        self,
        values: dict,
        units: dict | None,
        physical_unit: str,
        distance_dependent: bool = False,
    ) -> dict:
        """Convert parameters common to all Component subclasses."""
        if not units:
            return values

        capacity_parameters = {
            "capacityPerPlantUnit",
            "bigM",
            "capacityMin",
            "capacityMax",
            "capacityFix",
            "commissioningMin",
            "commissioningMax",
            "commissioningFix",
            "stockCommissioning",
        }

        per_capacity_cost_parameters = {"investPerCapacity", "opexPerCapacity"}

        fixed_cost_parameters = {"investIfBuilt", "opexIfBuilt"}

        converted = dict(values)

        for parameter in capacity_parameters:
            if parameter in units and parameter in converted:
                converted[parameter] = self.convert(
                    converted[parameter], units[parameter], physical_unit, parameter
                )

        for parameter in per_capacity_cost_parameters:
            if parameter in units and parameter in converted:
                converted[parameter] = self.convert(
                    converted[parameter],
                    units[parameter],
                    self.cost_per_capacity_unit(
                        physical_unit, distance_dependent=distance_dependent
                    ),
                    parameter,
                )

        for parameter in fixed_cost_parameters:
            if parameter in units and parameter in converted:
                converted[parameter] = self.convert(
                    converted[parameter],
                    units[parameter],
                    self.cost_if_built_unit(distance_dependent=distance_dependent),
                    parameter,
                )

        return converted

    @staticmethod
    def check_unit_keys(units, supported_parameters):
        """Validate that provided `units` keys are among supported parameters."""
        if units is None:
            return

        if not isinstance(units, dict):
            raise TypeError(
                "'units' must be None or a dictionary mapping parameter names to pint.Unit objects."
            )

        unsupported = set(units) - set(supported_parameters)

        if unsupported:
            raise ValueError(
                "Unsupported unit specification for parameter(s): "
                + ", ".join(sorted(unsupported))
            )
