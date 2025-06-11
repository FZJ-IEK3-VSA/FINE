
iterations = 3
keys = {
    'SourceSink': ['wind', 'solar'],
    'Conversion': ['heatpump']
}
values = ['cap','or']
data = [i for i in range(25)]

esM_solutions = {}
i = 0  # Index in data

for iteration in range(4):
    esM_solutions[iteration] = {}
    for key, params in keys.items():
        esM_solutions[iteration][key] = {}
        for param in params:
            esM_solutions[iteration][key][param] = {}
            for val in values:
                esM_solutions[iteration][key][param][val] = data[i]
                i += 1

# esM_solutions is basically all the MGA solutions thatb have been obtained in the MGA optimization process.
def supremum(i):
        m = 10**4
        x_sum = 0
        x_sum_list = []

        print(f'Number of total solutions in set_solutions is {len(set_solutions)} \n')
        for iteration in range(len(set_solutions)):
            sel_sum = 0
            print(f'Checking solution {i} in esM_solutions with solution {iteration} in set_solutions....')
            sel_sum = sum((esM_solutions[i][key][parameter][item]-set_solutions[iteration][key][parameter][item])**2
                        for key in esM_solutions[i]
                        for parameter in esM_solutions[i][key]
                        for item in esM_solutions[i][key][parameter])
            if sel_sum == 0:
                x_sum += m
                x_sum_list.append(m)
            else:
                x_sum += 1/sel_sum
                x_sum_list.append(1/sel_sum)
            print(f'For solution {iteration} in set_solutions, sel_sum is {sel_sum} and x_sum is {x_sum_list}')
            print('--------')

        return 1/x_sum, x_sum_list

# set_solutions is a dictionary that stores the maximally different MGA solutions. Always, set_solutions[0] is the optimal
# solution in the original optimization, which is also the first solution in esM_solutions. therefore, initially, set_solution
# has only 1 solution.
# Each solution in esM_solutions is compared with each solution in set_solutions. For example, the 4 solutions in esm_solutions
# are compared with the solution in set_solutions. The solution in esM_solutions with the highest squared mean Euclidian distance
# to the solution in set_solutions is the maximally different solution. This will become the new solution in set_solutions. Now,
# set_solutions has 2 solutions. This process contunues, untill set_solutions is filled.

set_solutions = {}
set_solutions[0] = esM_solutions[0]

for k in range(iterations):
    previous_max = 0
    highest_distance = 0

    print(f'Getting maximally different solution {k+1}-------- \n\n')
    for i in range(len(esM_solutions)):
        print(f'Sending solution {i} in esM_solutions to get the supremum')
        get_max, x_sum_list = supremum(i)
        print(f'For solution {i} in esM_solutions received a max value of {get_max}, which is the 1/sum{x_sum_list}')
        if get_max >= previous_max:
            highest_distance = i
            print(f'Max value of {get_max} is greater than or equal to previous_max {previous_max}. Maximally different solution so far is, solution {i}')
            previous_max = get_max
        else:
            print(f'Max value of {get_max} is less than to previous_max {previous_max}. ')
        print('------------------------------------')
    print (f"Maximally different solution {k+1} identified... Solution {highest_distance} \n\n")
    set_solutions[k+1] = esM_solutions[highest_distance]

print(set_solutions)
