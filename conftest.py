from fine.utils import ImplementedSolvers


def _gurobi_available():
    """Check if Gurobi is installed with a valid full (non-size-limited) license.

    Creates a Gurobi model that exceeds the 2000-variable limit of the
    restricted license bundled with the gurobipy pip package, then tries
    to optimize it.  If creating the environment fails, no license is
    available at all; if optimize fails, only the restricted license is
    present.  Model and environment are properly disposed of so that
    license tokens are released.

    See https://support.gurobi.com/hc/en-us/articles/4424054948881
    """
    try:
        import gurobipy as gp  # noqa: PLC0415
    except ImportError:
        return False

    env = None
    model = None
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        model.addVars(2001)
        model.optimize()
        return True
    except gp.GurobiError:
        return False
    finally:
        if model is not None:
            model.close()
        if env is not None:
            env.close()


if _gurobi_available() is True:
    ImplementedSolvers.STANDARD_SOLVER.value = ImplementedSolvers.GUROBI.value
else:
    ImplementedSolvers.STANDARD_SOLVER.value = ImplementedSolvers.GLPK.value

print(
    f"\n=== FINE test suite: using solver '{ImplementedSolvers.STANDARD_SOLVER.value}' ===\n"
)
