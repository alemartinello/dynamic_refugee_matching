import numpy as np
import pandas as pd
import time
import dynamic_refugee_matching.flowgen as flg
import dynamic_refugee_matching.assignment as dfa
from dynamic_refugee_matching.utilities import f_writetime

def initialize_results(n_asylum_seekers, errorlist, country_properties):
    results = {}
    for key in country_properties.keys():
        results[key] = {
            'Sequential': np.zeros((n_asylum_seekers, 3))
        }
        for err in errorlist:
            results[key]['DRM, {}% error'.format(err)] = np.zeros((n_asylum_seekers, 3))
    return results

def initialize_simulation_cache(dictionary):
    simulation_cache = {}
    for key in dictionary.keys():
        simulation_cache[key] = {}
        simulation_cache[key]['assignment'] = None
        simulation_cache[key]['results'] = np.zeros((dictionary[key].shape[0],3))
    
    return simulation_cache

# def initialize_over_errors(n_asylum_seekers, errorlist):
#     # Initialize empty result dictionary
#     results = initialize_results(n_asylum_seekers, errorlist)

#     # Initialize empty simulation dictionary
#     simulation_cache = initialize_simulation_cache(results)

#     return results, simulation_cache

def compute_measures_by_k(assignment, demand_matrix, envy1_limit=2): # Can't I return 3 column vectors? If no quotas...
    
    n_demanded_refugees = np.cumsum(np.amax(demand_matrix, axis=1, keepdims=True), axis=0)
    assignment_matrix = assignment.assignment
    # Efficiency: # misallocated acceptable/n_demanded.refugees
    misallocated = ((1-demand_matrix)*assignment_matrix*np.amax(demand_matrix, axis=1, keepdims=True))
    misallocated = np.cumsum(np.amax(misallocated, axis=1, keepdims=True), axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        inefficiency = misallocated/n_demanded_refugees
    inefficiency[inefficiency == np.inf] = 0
    inefficiency = inefficiency.flatten()
    
    # Envy
    n_refugees = assignment_matrix.shape[0]
    envy0 = []
    envy1 = []
    for k in np.arange(n_refugees):
        max_envy = np.amax(assignment.get_envy(refugee=k,correct_scores=demand_matrix), axis=1)
        envy0.append(np.mean((max_envy>0)))
        envy1.append(np.mean((max_envy>envy1_limit)))
        
    envy0 = np.array(envy0)
    envy1 = np.array(envy1)
    
    return inefficiency, envy0, envy1
    

def simulate_over_errors(n_asylum_seekers, n_simulations, country_properties, errorlist, envy1_limit=2):
    # Initialize empty result & simulation cache dictionary
    results = initialize_results(n_asylum_seekers, errorlist, country_properties)

    for country, country_prop in country_properties.items():
        for _ in np.arange(n_simulations):
            # Simulate demand matrix
            demand_matrix = flg.simulate_flow(
                n_asylum_seekers, 
                country_prop['n_localities'],
                country_prop['AA'],
                country_prop['NA'],
                country_prop['autocorrelation'],
            )

            # for each assignment type in results[country], update results
            for key in results[country].keys():
                # calculate assignment
                if key == 'Sequential':
                    assignment = dfa.assign_seq(demand_matrix)
                else:
                    error = int(key[-9:-7])
                    # scramble with error
                    scrambled_matrix = flg.add_error(demand_matrix, error)
                    assignment = dfa.assign(scrambled_matrix)
                # compute measures
                inefficiency, envy0, envy1 = compute_measures_by_k(assignment, demand_matrix, envy1_limit)

                # update results
                results[country][key][:,0] = results[country][key][:,0] + inefficiency/n_simulations
                results[country][key][:,1] = results[country][key][:,1] + envy0/n_simulations
                results[country][key][:,2] = results[country][key][:,2] + envy1/n_simulations

        # turn into dataframe
        for key in results[country].keys():
            results[country][key] = pd.DataFrame(results[country][key], columns=['inefficiency', 'envy0', 'envy1'])


    return results        

def simulate_over_quota_splits(vector_quotas, n_simulations, errorlist, envy1_limit):
    
    pass