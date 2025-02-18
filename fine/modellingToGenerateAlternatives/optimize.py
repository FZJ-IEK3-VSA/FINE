import random
import time
import importlib.util
import warnings
import os
import pandas as pd
import pyomo.environ as pyomo
import pyomo.opt as opt
import fine as fn

def declareOptimalCostConstraint(esM, pyM):

    """ Optimum cost should not be more than the objective value obtained in the original optimization + slack value. 
    """
    fn.utils.output("Declaring cost constraint...", esM.verbose, 0)
    # slack = slack

    def optimalCostConstraint(pyM):
        return (
            sum(
                mdl.getObjectiveFunctionContribution(esM, pyM)
                for mdl in esM.componentModelingDict.values()
            )
            <= esM.objectiveValue*(1+esM.slack)
        )
    pyM.optimalCostConstraint = pyomo.Constraint(rule=optimalCostConstraint) 

def declareMGAObjective(esM, pyM,iteration,sense):

    fn.utils.output("Declaring MGA objective function...", esM.verbose, 0)

    def mgaOperation(
        mdl,
        pyM,
        esM,
        iteration,
        opVarName,
        isOperationCommisYearDepending=False,
        ):
        """
        Declare the objective function by obtaining the opertaion rate variables abd capacity variables of the components. The  objective function is the sum of the operation and capacity variables 
        of the componenets multiplied by the Beta value.

            .. math::
                    \\{min/max}\:  {\beta}_{loc,comp,iteration} * ({op}^{comp,opType}_{loc,ip,p,t} + {cap}^{comp}_{loc})

        """

        abbrvName = mdl.abbrvName
        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        capVar = getattr(pyM, "cap_" + abbrvName)
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)
        capVarSet = getattr(pyM, "designDimensionVarSet_" + abbrvName) 

        if isOperationCommisYearDepending:

            opsum = sum(opVar[loc, compName, commis, ip, p, t]  * esM.beta[loc][iteration][compName]
                for loc,compName,commis,ip in opVarSet for p,t in pyM.intraYearTimeSet
            )

        else:

            opsum = sum(opVar[loc, compName, ip, p, t] * esM.beta[loc][iteration][compName]
                for loc,compName,ip in opVarSet for p,t in pyM.intraYearTimeSet 
            )
            
        capsum = sum(capVar[loc, compName, ip]  * esM.beta[loc][iteration][compName]
                            for loc, compName, ip in capVarSet)
        
        return (opsum + capsum)
    def mgaObjective(pyM):
        mgaContribution  = 0
        for key,mdl in esM.componentModelingDict.items():
            if key != 'StorageModel':
                mgaContribution += mgaOperation(mdl,pyM, esM, iteration, "op")
            else:
                variables = ["chargeOp","dischargeOp"]
                storageContribution = sum(mgaOperation(mdl,pyM, esM, iteration, var) for var in variables)
                mgaContribution += storageContribution
        return mgaContribution
    if sense == "minimize":
        pyM.Obj = pyomo.Objective(rule=mgaObjective, sense=pyomo.minimize)
    else:
        pyM.Obj = pyomo.Objective(rule=mgaObjective, sense=pyomo.maximize)

def declareMGAOptimizationProblem(
    esM,
    iteration,
    sense,
    timeSeriesAggregation=False,
    relaxIsBuiltBinary=False,
    relevanceThreshold=None,
):

    """
    Declare the optimization problem belonging to the specified energy system for which a pyomo concrete model
    instance is built and filled with

    * basic time sets,
    * sets, variables and constraints contributed by the component modeling classes,
    * basic, component overreaching constraints, and
    * an objective function.

    **Default arguments:**

    :param timeSeriesAggregation: states if the optimization of the energy system model should be done with

        (a) the full time series (False) or
        (b) clustered time series data (True).

        |br| * the default value is False
    :type timeSeriesAggregation: boolean

    :param relaxIsBuiltBinary: states if the optimization problem should be solved as a relaxed LP to get the lower
        bound of the problem.
        |br| * the default value is False
    :type declaresOptimizationProblem: boolean

    :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
        |br| * the default value is None
    :type relevanceThreshold: float (>=0) or None
    """
    fn.utils.output(f"MGA Iteration {iteration} {sense} .....", esM.verbose, 0 )

    # Get starting time of the optimization to, later on, obtain the total run time of the optimize function call
    timeStart = time.time()

    # Check correctness of inputs
    fn.utils.checkDeclareOptimizationProblemInput(
        timeSeriesAggregation, esM.isTimeSeriesDataClustered
    )

    # Set segmentation value if time series aggregation is True
    if timeSeriesAggregation:
        segmentation = esM.segmentation
    else:
        segmentation = False

    ################################################################################################################
    #                           Initialize mathematical model (ConcreteModel) instance                             #
    ################################################################################################################

    # Initialize a pyomo ConcreteModel which will be used to store the mathematical formulation of the model.
    # The ConcreteModel instance is stored in the EnergySystemModel instance, which makes it available for
    # post-processing or debugging. A pyomo Suffix with the name dual is declared to make dual values associated
    # to the model's constraints available after optimization.

    """
    :param pyM: a pyomo ConcreteModel instance which contains parameters, sets, variables,
        constraints and objective required for the optimization set up and solving.
    :type pyM: pyomo ConcreteModel
    """
    esM.pyM = pyomo.ConcreteModel()
    pyM = esM.pyM
    pyM.dual = pyomo.Suffix(direction=pyomo.Suffix.IMPORT)

    # Set time sets for the model instance
    esM.declareTimeSets(pyM, timeSeriesAggregation, segmentation)

    ################################################################################################################
    #                         Declare component specific sets, variables and constraints                           #
    ################################################################################################################

    for key, mdl in esM.componentModelingDict.items():
        _t = time.time()
        fn.utils.output("Declaring sets, variables and constraints for " + key, esM.verbose, 0)
        fn.utils.output("\tdeclaring sets... ", esM.verbose, 0), mdl.declareSets(esM, pyM)
        fn.utils.output("\tdeclaring variables... ", esM.verbose, 0), mdl.declareVariables(esM, pyM,relaxIsBuiltBinary, relevanceThreshold)
        fn.utils.output("\tdeclaring constraints... ", esM.verbose, 0), mdl.declareComponentConstraints(esM, pyM)
        fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    ################################################################################################################
    #                              Declare cross-componential sets and constraints                                 #
    ################################################################################################################

    # Declare constraints for enforcing shared capacities
    _t = time.time()
    esM.declareSharedPotentialConstraints(pyM)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Declare constraints for linked quantities
    _t = time.time()
    esM.declareComponentLinkedQuantityConstraints(pyM)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Declare commodity balance constraints (one balance constraint for each commodity, location and time step)
    _t = time.time()
    esM.declareCommodityBalanceConstraints(pyM)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Declare constraint for balanceLimit
    _t = time.time()
    esM.declareBalanceLimitConstraint(pyM, timeSeriesAggregation)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Declare constraint for optimal cost
    _t = time.time()
    declareOptimalCostConstraint(esM, pyM)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    ###############################################################################################################
    #                                         Declare MGA objective function                                           #
    ################################################################################################################

    # Declare objective function by obtaining the contributions to the objective function from all modeling classes
    _t = time.time()
    declareMGAObjective(esM, pyM,iteration,sense)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Store the build time of the optimize function call in the EnergySystemModel instance
    esM.solverSpecs["buildtime"] = time.time() - timeStart  

def optimalValues(esM, iteration):

    esM.solutions[iteration] = {}
    esM.optimalValueParameters = [
    "op_",
    "cap_",
    ]
    esM.storageParameters = ["chargeOp_","dischargeOp_"]

    for key, mdl in esM.componentModelingDict.items():
        esM.solutions[iteration][key] = {}
        for parameter in esM.optimalValueParameters:
            if not (parameter == "op_" and mdl.abbrvName == "stor"):
                if esM.numberOfInvestmentPeriods == 1:
                    esM.solutions[iteration][key][parameter] = getattr(esM.pyM, parameter + mdl.abbrvName).get_values()
                else:
                    # This needs to adjust
                    esM.solutions[iteration][key][parameter] = getattr(esM.pyM, parameter + mdl.abbrvName).get_values()
            else:
                for action in esM.storageParameters:
                    esM.solutions[iteration][key][action] = getattr(esM.pyM, action + mdl.abbrvName).get_values()
def mgaOptimize(
            esM,
            declaresOptimizationProblem=True,
            timeSeriesAggregation=False,
            logFileName="",
            threads=3,
            solver="None",
            timeLimit=None,
            optimizationSpecs="",
            warmstart=False,
            relevanceThreshold=None,
            slack=0.1,
            iterations = 10,
            random_seed = False,
            operationRateinOutput = False,
            writeSolutionsasExcels = False
):
    """
    Optimize the specified energy system for which a pyomo ConcreteModel instance is built or called upon.
    A pyomo instance is optimized with the specified inputs, and the optimization results are further
    processed.

    **Default arguments:**

    :param declaresOptimizationProblem: states if the optimization problem should be declared (True) or not (False).

        (a) If true, the declareOptimizationProblem function is called and a pyomo ConcreteModel instance is built.
        (b) If false a previously declared pyomo ConcreteModel instance is used.

        |br| * the default value is True
    :type declaresOptimizationProblem: boolean

    :param timeSeriesAggregation: states if the optimization of the energy system model should be done with

        (a) the full time series (False) or
        (b) clustered time series data (True).

        |br| * the default value is False
    :type timeSeriesAggregation: boolean

    :param logFileName: logFileName is used for naming the log file of the optimization solver output
        if gurobi is used as the optimization solver.
        If the logFileName is given as an absolute path (e.g. logFileName = os.path.join(os.getcwd(),
        'Results', 'logFileName.txt')) the log file will be stored in the specified directory. Otherwise,
        it will be stored by default in the directory where the executing python script is called.
        |br| * the default value is 'job'
    :type logFileName: string

    :param threads: number of computational threads used for solving the optimization (solver dependent
        input) if gurobi is used as the solver. A value of 0 results in using all available threads. If
        a value larger than the available number of threads are chosen, the value will reset to the maximum
        number of threads.
        |br| * the default value is 3
    :type threads: positive integer

    :param solver: specifies which solver should solve the optimization problem (which of course has to be
        installed on the machine on which the model is run).
        |br| * the default value is 'gurobi'
    :type solver: string

    :param timeLimit: if not specified as None, indicates the maximum solve time of the optimization problem
        in seconds (solver dependent input). The use of this parameter is suggested when running models in
        runtime restricted environments (such as clusters with job submission systems). If the runtime
        limitation is triggered before an optimal solution is available, the best solution obtained up
        until then (if available) is processed.
        |br| * the default value is None
    :type timeLimit: strictly positive integer or None

    :param optimizationSpecs: specifies parameters for the optimization solver (see the respective solver
        documentation for more information). Example: 'LogToConsole=1 OptimalityTol=1e-6'
        |br| * the default value is an empty string ('')
    :type optimizationSpecs: string

    :param warmstart: specifies if a warm start of the optimization should be considered
        (not always supported by the solvers).
        |br| * the default value is False
    :type warmstart: boolean

    :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
        |br| * the default value is None
    :type relevanceThreshold: float (>=0) or None

    :param slack: slack parameter for the MGA optimization algorithm. The slack parameter decides the upper limit of the system total cost should be. For e.g. if slack is 0.2, the system total cost should not be more than 1.2 times the original optimal cost.
    :type slack: float (>0)

    :param iterations: number of iterations of the MGA optimization algorithm
    :type iterations: strictly positive integer

    :param random_seed: random seed for the MGA optimization algorithm. If random seed is set to True, the results shall be the same each time the code is executed.
    :type random_seed: boolean

    :param operationRateinOutput: If operationRateinOutput is set to True, the operation rate of the alternate solutions will be written to the output file.
    :type operationRateinOutput: boolean

    :param writeSolutionsasExcels: If writeSolutionsasExcels is set to True, the alternate solutions will be written to excel files
    :type writeSolutionsasExcels: boolean

    """
    
    esM.objectiveValue = esM.pyM.Obj()
    esM.solutions = {}
    esM.iterations = iterations
    esM.slack = slack

    optimalValues(esM, 0)

    if esM.solutions[0] is None:
        raise TypeError(
        "The optimization problem for optimal solution doesn't have an optimal solution"
        "Cannot perofrm a MGA optimization if the optimization problem doesn't have an optimal solution."
        )
    
    else:
        components = []
        sinkComponents = []
        transmissionComponents = []

        for item in esM.componentModelingDict.values():
            for key,_item in item.componentsDict.items():
                components.append(key)
                if isinstance(_item, fn.sourceSink.Sink):
                    sinkComponents.append(key)
                elif isinstance(_item, fn.transmission.Transmission):
                    transmissionComponents.append(key)
        # print("components: ", components)
        # print("sinkComponents: ", sinkComponents)
        # print("transmissionComponents: ", transmissionComponents)
        
        if random_seed:
            random.seed(10)
        
        """Beta is a random value between 0 and 1 and it changes with location, time and iteration. This Beta value
        is used to build the objective function of the MGA optimization.
        """
        transmission_locations = []
        for loc1 in esM.locations:
            for loc2 in esM.locations:
                transmission_locations.append(loc1 + "_" + loc2)

        esM.beta = {location: 
                {iteration+1: 
                {component: random.random() if component not in sinkComponents and component not in transmissionComponents else 1 
                    if component in sinkComponents else None for component in components if component not in transmissionComponents
                }  
                for iteration in range(esM.iterations)
                } 
                for location in esM.locations
                }
        
        new_data = {location: 
                {iteration+1: 
                {component: random.random() for component in transmissionComponents
                }  
                for iteration in range(esM.iterations)
                } 
                for location in transmission_locations
                }

        esM.beta.update(new_data)

        if not timeSeriesAggregation:
            esM.segmentation = False
        
        _t = time.time()

        """ 
        MGA optimization is an iterative process. It starts with the first iteration and ends with the last iteration (self.iterations). Each iteration has a minimization and a maximization of the optimization problem.
        therefore, each iteration provides 2 solutions. After the last iteration, there will be a maximum of 2*(self.iterations) times final solutions. The optimization problem is defined in the declareMGAOptimizationProblem function."
        """
        iteration =1
        while iteration <= esM.iterations:
            for sense in ["minimize","maximize"]:    

                if declaresOptimizationProblem:
                    declareMGAOptimizationProblem(
                        esM,
                        iteration,
                        sense,
                        timeSeriesAggregation=timeSeriesAggregation,
                        relevanceThreshold=relevanceThreshold,
                        )
                elif esM.pyM is None:
                        raise TypeError(
                            "The optimization problem is not declared yet. Set the argument declaresOptimization"
                            " problem to True or call the declareOptimizationProblem function first."
                        )

                # Get starting time of the optimization to, later on, obtain the total run time of the optimize function call
                timeStart = time.time()

                # Check correctness of inputs
                fn.utils.checkOptimizeInput(
                    timeSeriesAggregation,
                    esM.isTimeSeriesDataClustered,
                    logFileName,
                    threads,
                    solver,
                    timeLimit,
                    optimizationSpecs,
                    warmstart,
                )

                # Store keyword arguments in the EnergySystemModel instance
                esM.solverSpecs["logFileName"], esM.solverSpecs["threads"] = (
                    logFileName,
                    threads,
                )
                esM.solverSpecs["solver"], esM.solverSpecs["timeLimit"] = solver, timeLimit
                esM.solverSpecs["optimizationSpecs"], esM.solverSpecs["hasTSA"] = (
                    optimizationSpecs,
                    timeSeriesAggregation,
                )

                # Check which solvers are available and choose default solver if no solver is specified explicitely
                # Order of possible solvers in solverList defines the priority of chosen default solver.
                solverList = ["gurobi", "glpk", "cbc"]

                if solver != "None":
                    try:
                        opt.SolverFactory(solver).available()
                    except Exception:
                        solver = "None"

                if solver == "None":
                    for nSolver in solverList:
                        if solver == "None":
                            try:
                                if opt.SolverFactory(nSolver).available():
                                    solver = nSolver
                                    fn.utils.output(
                                        "Either solver not selected or specified solver not available."
                                        + str(nSolver)
                                        + " is set as solver.",
                                        esM.verbose,
                                        0,
                                    )
                            except Exception:
                                pass

                if solver == "None":
                    raise TypeError(
                        "At least one solver must be installed."
                        " Have a look at the FINE documentation to see how to install possible solvers."
                        " https://vsa-fine.readthedocs.io/en/latest/"
                    )
                
                ################################################################################################################
                #                                  Solve the specified optimization problem                                    #
                ################################################################################################################

                # Set which solver should solve the specified optimization problem
                if solver == "gurobi" and importlib.util.find_spec('gurobipy'):
                    # Use the direct gurobi solver that uses the Python API.
                    optimizer = opt.SolverFactory(solver, solver_io="python")
                else:
                    optimizer = opt.SolverFactory(solver)

                # Set, if specified, the time limit
                if esM.solverSpecs["timeLimit"] is not None and solver == "gurobi":
                    optimizer.options["timelimit"] = timeLimit

                # Set the specified solver options
                if "LogToConsole=" not in optimizationSpecs and solver == "gurobi":
                    if esM.verbose == 2:
                        optimizationSpecs += " LogToConsole=0"

                # Solve optimization problem. The optimization solve time is stored and the solver information is printed.
                if solver == "gurobi":
                    optimizer.set_options(
                        "Threads="
                        + str(threads)
                        + " logfile="
                        + logFileName
                        + " "
                        + optimizationSpecs
                    )
                    solver_info = optimizer.solve(
                        esM.pyM,
                        warmstart=warmstart,
                        tee=True,
                    )
                elif solver == "glpk":
                    optimizer.set_options(optimizationSpecs)
                    solver_info = optimizer.solve(esM.pyM, tee=True)
                else:
                    solver_info = optimizer.solve(esM.pyM, tee=True)
                esM.solverSpecs["solvetime"] = time.time() - timeStart
                fn.utils.output(solver_info.solver(), esM.verbose, 0), fn.utils.output(
                    solver_info.problem(), esM.verbose, 0
                )
                fn.utils.output(
                    "Solve time: " + str(esM.solverSpecs["solvetime"]) + " sec.",
                    esM.verbose,
                    0,
                )

                # Post-process the optimization output by differentiating between different solver statuses and termination
                # conditions. First, check if the status and termination_condition of the optimization are acceptable.
                # If not, no output is generated.
                # TODO check if this is still compatible with the latest pyomo version
                status, termCondition = (
                    solver_info.solver.status,
                    solver_info.solver.termination_condition,
                )
                esM.solverSpecs["status"] = str(status)
                esM.solverSpecs["terminationCondition"] = str(termCondition)
                if (
                    status == opt.SolverStatus.error
                    or status == opt.SolverStatus.aborted
                    or status == opt.SolverStatus.unknown
                ):
                    fn.utils.output(
                        "Solver status:  "
                        + str(status)
                        + ", termination condition:  "
                        + str(termCondition)
                        + ". No output is generated.",
                        esM.verbose,
                        0,
                    )
                elif (
                    solver_info.solver.termination_condition
                    == opt.TerminationCondition.infeasibleOrUnbounded
                    or solver_info.solver.termination_condition
                    == opt.TerminationCondition.infeasible
                    or solver_info.solver.termination_condition
                    == opt.TerminationCondition.unbounded
                ):
                    fn.utils.output(
                        "Optimization problem is "
                        + str(solver_info.solver.termination_condition)
                        + ". No output is generated.",
                        esM.verbose,
                        0,
                    )
                else:
                    # If the solver status is not okay (hence either has a warning, an error, was aborted or has an unknown
                    # status), show a warning message.
                    if (
                        not solver_info.solver.termination_condition
                        == opt.TerminationCondition.optimal
                        and esM.verbose < 2
                    ):
                        warnings.warn("Output is generated for a non-optimal solution.")
                    # fn.utils.output("\nProcessing optimization output...", self.verbose, 0)
                    # Declare component specific sets, variables and constraints
                    # w = str(len(max(self.componentModelingDict.keys())) + 6)

                    """
                    MGA solutions consist of the operation rate variables and capacity variables of the components. MGA solutions are stored in self.solutions.
                    """

                    if sense == "minimize":
                        optimalValues(esM,iteration)
                        # self.solutions[iteration] = getattr(self.pyM, "op_" + "srcSnk").get_values() 
                    else:
                        optimalValues(esM, esM.iterations + iteration)
                        # self.solutions[self.iterations + iteration] = getattr(self.pyM, "op_" + "srcSnk").get_values() 
                print(esM.pyM.optimalCostConstraint.display())###################################
            iteration +=1
        # print(self.solutions)
        # self.get_solutions()
        fn.utils.output("\n\t\tMGA optimization completed after %.4f" % (time.time() - _t) + " sec\n", esM.verbose, 0)
        # self.fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", self.verbose, 0)
        
        # ############################################################################################################
        # # #                                      Identify maximally different solutions                                       
        # # ################################################################################################################
        """
        MGA optimization provides 2*(self.iterations) times different solutions. From these solutions, solutions which are maximally different to the optimal solutions should be identified.
        For this, largest squared Euclidian distance between the solutions are calculated.   
        """
        def supremum(i):
            m = 10**4
            x_sum = 0

            for iteration in range(len(set_solutions)):
                sel_sum = 0

                sel_sum += sum((esM.solutions[i][key][parameter][item]-set_solutions[iteration][key][parameter][item])**2 for key in esM.solutions[i] 
                                for parameter in esM.solutions[i][key] for item in esM.solutions[i][key][parameter]) 
                if sel_sum == 0:
                    x_sum += m
                else:
                    x_sum += 1/sel_sum

            return 1/x_sum

        # def get_solutions(self):

        set_solutions = {}
        set_solutions[0] = esM.solutions[0]
        # print(self.set_solutions[0])

        fn.utils.output("\nIdentifying maximally different solutions....\n", esM.verbose, 0)
        for k in range(esM.iterations):
            previous_max = 0
            highest_distance = 0
            # for i in range(2*self.iterations):
            for i in range(len(esM.solutions)):
                get_max = supremum(i)
                if get_max > previous_max:
                    highest_distance = i
                    previous_max = get_max
            # if highest_distance not in used_solutions:
            fn.utils.output (f"Maximally different solution {k+1} identified... Solution {highest_distance}", esM.verbose, 0)
            set_solutions[k+1] = esM.solutions[highest_distance]  
            # used_solutions.append(highest_distance)  

        #################################################################################################################
        # #                                      Post-process optimization output                                        #
        ###########################################################################################################

        # iterate over investment periods, to get yearly results
        # for key, mdl in self.componentModelingDict.items():

        if writeSolutionsasExcels:
            fn.utils.output("\nWriting optimization output to Excel files\n", esM.verbose, 0)

            cwd = os.getcwd()
            # outdir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "OutputData")
            outdir = os.path.join(cwd, "OutputData")

            if not os.path.exists(outdir):
                os.mkdir(outdir)

            if not operationRateinOutput:   # if optimalValueParameters is True, we do not require operation rate variables in the output anymore.
                esM.optimalValueParameters = ["cap_"]

            for ip in esM.investmentPeriods:    # Currently a single investment period is consdiered.           
                for key, mdl in esM.componentModelingDict.items():
                    _t = time.time()
                    fn.utils.output(f"\tWriting {key} output....", esM.verbose, 0)
                    outputData = {}
                    file_name = f"{key}.xlsx"
                    outputFile = os.path.join(outdir, file_name)
                    with pd.ExcelWriter(outputFile) as writer: 
                        for parameter in esM.optimalValueParameters: 
                            for k in range((esM.iterations+1)):   
                                if parameter == "op_":
                                    if key != "TransmissionModel" and key != "StorageModel":
                                        outputData[f'{parameter}_{k}'] = fn.utils.formatOptimizationOutput(
                                            set_solutions[k][key][parameter],
                                            "operationVariables",
                                            "1dim",
                                            ip,
                                            esM.periodsOrder[ip],
                                            esM=esM,
                                        )
                                        outputData[f'{parameter}_{k}'].to_excel(writer, sheet_name=f'{parameter}_{k}')

                                    elif key == "StorageModel":  
                                        for action in esM.storageParameters:
                                            outputData[f'{action}_{k}'] = fn.utils.formatOptimizationOutput(
                                            set_solutions[k][key][action],
                                            "operationVariables",
                                            "1dim",
                                            ip,
                                            esM.periodsOrder[ip],
                                            esM=esM,
                                        )
                                            outputData[f'{action}_{k}'].to_excel(writer, sheet_name=f'{action}_{k}')
                                            
                                    else:
                                        outputData[f'{parameter}_{k}'] = fn.utils.formatOptimizationOutput(
                                            set_solutions[k][key][parameter],
                                            "operationVariables",
                                            "2dim",
                                            ip,
                                            esM.periodsOrder[ip],
                                            compDict=mdl.componentsDict,
                                            esM=esM,
                                        )
                                        outputData[f'{parameter}_{k}'].to_excel(writer, sheet_name=f'{parameter}_{k}')

                                else:
                                    outputData[f'{parameter}_{k}'] = fn.utils.formatOptimizationOutput(
                                        set_solutions[k][key][parameter],
                                        "designVariables",
                                        mdl.dimension,
                                        ip,
                                        compDict=mdl.componentsDict,
                                    )
                                    outputData[f'{parameter}_{k}'].to_excel(writer, sheet_name=f'{parameter}_{k}')
                    fn.utils.output("\t\t (%.4f)" % (time.time() - _t) + " sec\n", esM.verbose, 0)
                            
            print("\nClutsering output to single sheets")
            # if not self.operationRateinOutput:
            for key, mdl in esM.componentModelingDict.items():
                if key!= "TransmissionModel":
                    print(f"\tfor {key}....")
                    _t = time.time()
                    file_name = f"{key}.xlsx"
                    new_file_name = f"{key}_capacity_clustered.xlsx"
                    inputFile = os.path.join(outdir, file_name)
                    outputFile = os.path.join(outdir, new_file_name)
                    data = pd.read_excel( inputFile, sheet_name="cap__0",index_col=0)
                    column_list = list(esM.locations)
                    index_list = data.index  
                    column_list.sort()
                    multi_index = pd.MultiIndex.from_product([index_list,column_list])  
                    row_index = [iteration for iteration in range(esM.iterations)] 
                    df = pd.DataFrame(index=row_index, columns=multi_index)

                    for item in index_list:
                        for location in column_list:
                            items = []
                            for iteration in range(esM.iterations):
                                input_data = pd.read_excel( inputFile, sheet_name=f"cap__{iteration}",index_col=0)
                                items.append(input_data.loc[item][location])
                            df.loc[:, (item,location)] = items
                    df.to_excel(outputFile)
                    fn.utils.output("\t\t (%.4f)" % (time.time() - _t) + " sec\n", esM.verbose, 0)
            print(f"Output is saved in {outdir}\n")
            
            # else:
            #     print("Selected operation can be provided only for capacity variables. Run the mgaOptimize method as operationRateinOutput as false")
        fn.utils.output("\n\t MGA optimization completed", esM.verbose, 0) 

        