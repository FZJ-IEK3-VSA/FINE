import pytest
import pyomo.environ as pyo
from pyomo.core import NonNegativeReals

# --- Dummy Classes for Testing ---

class DummyComponent:
    def __init__(self, name, material_intensity, processed_stock_years, has_capacity):
        self.name = name
        self.MaterialIntensity = material_intensity  # Either a dict or None
        self.processedStockYears = processed_stock_years  # A list (e.g., [2020, 2025])
        self.hasCapacityVariable = has_capacity  # Boolean

class DummyESM:
    def __init__(self, investment_periods, materials_available):
        self.investmentPeriods = investment_periods
        self.materialsAvailable = materials_available
        self.componentsDict = {}

        
    def add(self, comp):
        # Use the component's name as key
        self.componentsDict[comp.name] = comp

class DummySelf:
    def __init__(self, componentsDict, abbrvName):
        self.componentsDict = componentsDict
        self.abbrvName = abbrvName

# --- Local Versions of the Functions to Test ---

def declareMaterialIntensityVarSet(self, pyM, esM):
    """
    This function attaches a Pyomo Set to the model with elements (compName, ip, resource)
    generated from the components in self.componentsDict.
    """
    compDict, abbrvName = self.componentsDict, self.abbrvName

    def init_material_intensity_var_set(pyM):
        return (
            (compName, ip, resource)
            for compName, comp in compDict.items()
            if comp.MaterialIntensity is not None
            for ip in comp.processedStockYears + esM.investmentPeriods
            for resource in comp.MaterialIntensity.keys()
            if comp.hasCapacityVariable
        )

    # Attach the set to the Pyomo model
    setattr(
        pyM,
        "MaterialIntensityVarSet" + abbrvName,
        pyo.Set(dimen=3, initialize=init_material_intensity_var_set),
    )
    # For testing, return the evaluated set as a list.
    return list(getattr(pyM, "MaterialIntensityVarSet" + abbrvName))

def declareMaterialIntensityVars(self, pyM, esM):
    """
    This function attaches a Pyomo Var to the model which is indexed by the previously
    declared MaterialIntensityVarSet.
    """
    abbrvName = self.abbrvName
    setattr(
        pyM,
        "MaterialIntensity" + abbrvName,
        pyo.Var(
            getattr(pyM, "MaterialIntensityVarSet" + abbrvName),
            domain=NonNegativeReals,
        ),
    )

# --- Pytest Test Functions ---

def test_declare_material_intensity_var_set():
    # Create dummy energy system model with investment periods
    investment_periods = [2030, 2040]
    esm = DummyESM(investment_periods)
    
    # Create a dummy component (e.g., "Wind (onshore)") with proper attributes.
    # For instance, processedStockYears are given and MaterialIntensity is non-empty.
    comp = DummyComponent(
        name="Wind (onshore)",
        material_intensity={"cobalt": 5},
        processed_stock_years=[2020, 2025],
        has_capacity=True
    )
    esm.add(comp)
    
    # Create dummy self holding the components and an abbreviation string.
    dummy_self = DummySelf(componentsDict=esm.componentsDict, abbrvName="_test")
    
    # Create a Pyomo model
    pyM = pyo.ConcreteModel()
    
    # Call the function that attaches the set to the model and return its elements.
    result = declareMaterialIntensityVarSet(dummy_self, pyM, esm)
    
    # Expected: for the single component "Wind (onshore)", we expect the set to be built from:
    # processedStockYears: [2020, 2025]
    # investmentPeriods: [2030, 2040]
    # Combined, these yield [2020, 2025, 2030, 2040] for ip.
    # For each ip, we iterate over the keys of MaterialIntensity: ["cobalt"].
    expected = [
        ("Wind (onshore)", 2020, "cobalt"),
        ("Wind (onshore)", 2025, "cobalt"),
        ("Wind (onshore)", 2030, "cobalt"),
        ("Wind (onshore)", 2040, "cobalt"),
    ]
    assert result == expected

def test_declare_material_intensity_vars():
    # Create dummy energy system model with investment periods
    investment_periods = [2030, 2040]
    esm = DummyESM(investment_periods)
    
    # Create a dummy component that will be indexed in the set.
    comp = DummyComponent(
        name="Expensive",
        material_intensity={"lithium": 5},
        processed_stock_years=[2020, 2025],
        has_capacity=True
    )
    esm.add(comp)
    
    # Create dummy self and a Pyomo model.
    dummy_self = DummySelf(componentsDict=esm.componentsDict, abbrvName="_test")
    pyM = pyo.ConcreteModel()
    
    # First, attach the set.
    declareMaterialIntensityVarSet(dummy_self, pyM, esm)
    
    # Now, attach the variable.
    declareMaterialIntensityVars(dummy_self, pyM, esm)
    
    # Check that the variable is attached.
    var_name = "MaterialIntensity" + dummy_self.abbrvName
    assert hasattr(pyM, var_name)
    material_intensity_var = getattr(pyM, var_name)
    
    # Retrieve the set for indexing.
    set_name = "MaterialIntensityVarSet" + dummy_self.abbrvName
    material_intensity_set = list(getattr(pyM, set_name))
    
    # Expected index set for the "Expensive" component:
    expected_set = [
        ("Expensive", 2020, "lithium"),
        ("Expensive", 2025, "lithium"),
        ("Expensive", 2030, "lithium"),
        ("Expensive", 2040, "lithium"),
    ]
    assert material_intensity_set == expected_set
    
    # For additional verification, assign a value to one index and check it.
    # (Pyomo variables are mutable and you can set a value.)
    idx = expected_set[0]
    material_intensity_var[idx] = 1.0
    # Create an instance of the model to compute (if needed).
    instance = pyM.create_instance()
    # In a ConcreteModel, values should be available immediately.
    assert material_intensity_var[idx].value == 1.0

import pytest

# Dummy component class (if not already imported)
class DummyComponent:
    def __init__(self, name, material_intensity, processed_stock_years, has_capacity):
        self.name = name
        self.MaterialIntensity = material_intensity  # Should be a dict or None
        self.processedStockYears = processed_stock_years  # E.g., [2020, 2025]
        self.hasCapacityVariable = has_capacity  # Boolean

def test_material_intensity_keys_direct():
    # Create a dummy component with a valid MaterialIntensity dictionary
    material_intensity = {"cobalt": 5, "lithium": 10}
    comp = DummyComponent(
        name="TestComponent",
        material_intensity=material_intensity,
        processed_stock_years=[2020, 2025],
        has_capacity=True
    )
    
    # Iterate over the keys and check they are valid.
    for key in comp.MaterialIntensity.keys():
        assert key is not None, "MaterialIntensity key should not be None"
        assert isinstance(key, str), f"MaterialIntensity key should be a string, got {type(key)}"

def test_processed_stock_years():
    # Create a dummy component with valid processedStockYears
    comp = DummyComponent(
        name="TestComponent",
        material_intensity={"cobalt": 5},
        processed_stock_years=[2020, 2025],
        has_capacity=True
    )
    # Check that processedStockYears is not None and is a list of integers
    assert comp.processedStockYears is not None, "processedStockYears should not be None"
    assert isinstance(comp.processedStockYears, list), "processedStockYears should be a list"
    for year in comp.processedStockYears:
        assert year is not None, "Each year in processedStockYears should not be None"
        assert isinstance(year, int), "Each year should be an integer"


def test_investment_periods():
    # Create a dummy ESM with valid investment periods
    esm = DummyESM([2030, 2040])

    assert esm.investmentPeriods is not None, "investmentPeriods should not be None"
    assert isinstance(esm.investmentPeriods, list), "investmentPeriods should be a list"
    for period in esm.investmentPeriods:
        assert period is not None, "Each investment period should not be None"
        assert isinstance(period, int), "Each investment period should be an integer"

import pytest
import pyomo.environ as pyo
from pyomo.environ import value
import pyomo.core as pyomo

# --- Dummy Classes for Testing ---

class DummyComponent:
    def __init__(self, name, material_intensity, processed_stock_years, has_capacity):
        self.name = name
        self.MaterialIntensity = material_intensity  # e.g., {"cobalt": 5}
        self.processedStockYears = processed_stock_years  # Not used in this constraint
        self.hasCapacityVariable = has_capacity  # Boolean

class DummyLocElig:
    @property
    def index(self):
        # Dummy processed locational eligibility index (list of locations)
        return ["Loc1", "Loc2"]

class DummyESM:
    def __init__(self, investment_periods, materials_available):
        self.investmentPeriods = investment_periods  # e.g. [2030, 2040]
        self.materialsAvailable = materials_available  # e.g. {"cobalt": 100}
        self.componentsDict = {}
        
    def add(self, comp):
        self.componentsDict[comp.name] = comp

class DummySelf:
    def __init__(self, componentsDict, abbrvName):
        self.componentsDict = componentsDict
        self.abbrvName = abbrvName

# --- Function Under Test ---

def declareMaterialAvailabilityConstraints(self, pyM, esM):
    """
    Ensures that material usage for commissioning does not exceed available resources.
    """
    compDict, abbrvName = self.componentsDict, self.abbrvName
    matCommisVar = getattr(pyM, "materialsCommis_" + abbrvName)

    # Material availability constraint
    def material_availability(pyM, materialType):
        return sum(
            matCommisVar[loc, compName, ip]
            for compName in compDict
            for loc in compDict[compName].processedLocationalEligibility.index
            for ip in esM.investmentPeriods
            if materialType in esM.materialsAvailable
        ) <= esM.materialsAvailable[materialType]

    setattr(
        pyM,
        "ConstrMaterialAvailability_" + abbrvName,
        pyomo.Constraint(esM.materialsAvailable.keys(), rule=material_availability),
    )

# --- Test Function ---

def test_material_availability_constraint():
    # Setup dummy ESM: investment periods and available materials.
    investment_periods = [2030, 2040]
    materials_available = {"cobalt": 100}
    esm = DummyESM(investment_periods, materials_available)
    
    # Create a dummy component.
    # (Note: processedStockYears is not used in the constraint; we rely on processedLocationalEligibility.)
    comp = DummyComponent(
        name="TestComp",
        material_intensity={"cobalt": 5},
        processed_stock_years=[2020, 2025],
        has_capacity=True
    )
    # Attach a dummy processedLocationalEligibility attribute.
    comp.processedLocationalEligibility = DummyLocElig()
    esm.add(comp)
    
    # Create dummy 'self' with components and abbreviation.
    dummy_self = DummySelf(componentsDict=esm.componentsDict, abbrvName="_test")
    
    # Create a Pyomo ConcreteModel.
    pyM = pyo.ConcreteModel()
    
    # Define the variable materialsCommis_{abbrvName} with indices (loc, compName, ip).
    # Build the index set based on the component's processedLocationalEligibility.index and esm.investmentPeriods.
    indices = []
    for compName, comp in esm.componentsDict.items():
        for loc in comp.processedLocationalEligibility.index:
            for ip in esm.investmentPeriods:
                indices.append((loc, compName, ip))
    indices = set(indices)
    
    pyM.materialsCommis__test = pyo.Var(indices, domain=pyo.NonNegativeReals)
    
    # Assign dummy values to the variable.
    # For 2 locations and 2 investment periods, we'll have 4 indices.
    # Set each variable to 10 so that total usage is 4*10 = 40.
    for idx in indices:
        pyM.materialsCommis__test[idx] = 10
    
    # Now call the constraint declaration function.
    declareMaterialAvailabilityConstraints(dummy_self, pyM, esm)
    
    # Check that the constraint is attached.
    constr_name = "ConstrMaterialAvailability_" + dummy_self.abbrvName
    assert hasattr(pyM, constr_name), f"Constraint {constr_name} is not attached to the model."
    
    constr = getattr(pyM, constr_name)
    
    # The constraint is indexed by esm.materialsAvailable.keys() (i.e., "cobalt").
    # For "cobalt", the sum should be:
    # For TestComp: For each loc in ["Loc1", "Loc2"] and for each ip in [2030, 2040]:
    # sum = materialsCommisVar["Loc1","TestComp",2030] + ... (4 terms)
    # In our case, that is 10*4 = 40.
    expr = constr["cobalt"].body
    total_usage = value(expr)
    
    # Check that the total usage does not exceed available material.
    assert total_usage <= materials_available["cobalt"], (
        f"Total usage {total_usage} exceeds available cobalt {materials_available['cobalt']}."
    )
    
    # Optionally, check the actual computed sum is as expected.
    assert total_usage == 40, f"Expected total usage of 40, but got {total_usage}."
