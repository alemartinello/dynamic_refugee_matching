import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
import time
from collections import Counter
import dynamic_refugee_matching.flowgen as flg
import dynamic_refugee_matching.assignment as dfa
import dynamic_refugee_matching.evaluate as evl
from dynamic_refugee_matching.utilities import f_writetime


def initialize_dynamic_results(n_asylum_seekers, errorlist, country_properties):
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
#     results = initialize_dynamic_results(n_asylum_seekers, errorlist)

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
    results = initialize_dynamic_results(n_asylum_seekers, errorlist, country_properties)

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

def simulate_over_quota_splits(vector_quotas, n_simulations, errorlist, splits, envy1_limit=2):
    
    for sim in range(n_simulations):
        # Initialize empty results list
        results = []

        # Compute useful measures
        n_refugees = np.int(np.sum(vector_quotas))
        n_localities = vector_quotas.shape[0]
        # Simulate matrix
        demand_matrix = flg.simulate_flow(n_refugees, n_localities, 0.28, 0.45, 0.5)
        for error in errorlist:
            # Add error
            observed_matrix = flg.add_error(demand_matrix,err=error)
            for split in splits:
                # Calculate period quotas
                split_quotas =  np.int32(vector_quotas/split)

                # Initialize empty assignment arrays
                assignments = {
                    'DRM': np.array([], dtype=np.int8).reshape(0, n_localities),
                    'DRM with legacy': np.array([], dtype=np.int8).reshape(0, n_localities),
                    'Sequential': np.array([], dtype=np.int8).reshape(0, n_localities)
                }

                # Initialize intermediate utility measures
                n_ref_start = 0
                n_ref_end = np.int(n_refugees/split)
                envy_init = np.zeros((n_localities, n_localities))

                for _ in range(split):
                    # Assign refugees within lap
                    observed_submatrix = observed_matrix[n_ref_start:n_ref_end, :]
                    drm = dfa.assign(observed_submatrix , vector_quotas=split_quotas)
                    drm_legacy = dfa.assign(observed_submatrix , vector_quotas=split_quotas, envy_init=envy_init)
                    seq = dfa.assign_seq(observed_submatrix , vector_quotas=split_quotas)
                    
                    # Append lap assigment to assignment arrays
                    assignments['DRM'] = np.append(assignments['DRM'], drm.assignment, axis=0)
                    assignments['DRM with legacy'] = np.append(assignments['DRM with legacy'], drm_legacy.assignment, axis=0)
                    assignments['Sequential'] = np.append(assignments['Sequential'], seq.assignment, axis=0)
                    
                    # Update initial envy (for legacy)
                    envy_init = np.int32(evl.calc_envy(assignments['DRM with legacy'], demand_matrix[:n_ref_end,:]))

                    # Update utilities
                    n_ref_start += np.int(n_refugees/split)
                    n_ref_end += np.int(n_refugees/split)

                # calculate end-of-period measures
                for method in assignments.keys():
                    max_envy = np.amax(evl.calc_envy(assignments[method], demand_matrix), axis=1)
                    fair0 = np.mean(max_envy>0)
                    fair1 = np.mean(max_envy>=envy1_limit)
                    misallocated = ((1-demand_matrix)*assignments[method]*np.amax(demand_matrix, axis=1, keepdims=True))
                    effic = np.sum(misallocated)/np.sum(np.sum(demand_matrix, axis=1)>0)

                    # append to result list
                    results.append([sim, error, split, method, fair0, fair1, effic])
    
    # convert results into dataframe
    results = pd.DataFrame(
        results, 
        columns=['wave','error','splits', 'method', 'Envy by 0', 'Envy by {}'.format(envy1_limit) , 'Inefficiency']
        )
    return results


def simulate_over_quota_splits_mp_func(sim, vector_quotas, errorlist, splits, envy1_limit):
    # Initialize empty results list
    results = []

    # Compute useful measures
    n_refugees = np.int(np.sum(vector_quotas))
    n_localities = vector_quotas.shape[0]
    # Simulate matrix
    demand_matrix = flg.simulate_flow(n_refugees, n_localities, 0.28, 0.45, 0.5)
    for error in errorlist:
        # Add error
        observed_matrix = flg.add_error(demand_matrix,err=error)
        for split in splits:
            # Calculate period quotas
            split_quotas =  np.int32(vector_quotas/split)

            # Initialize empty assignment arrays
            assignments = {
                'DRM': np.array([], dtype=np.int8).reshape(0, n_localities),
                'DRM with legacy': np.array([], dtype=np.int8).reshape(0, n_localities),
                'Sequential': np.array([], dtype=np.int8).reshape(0, n_localities)
            }

            # Initialize intermediate utility measures
            n_ref_start = 0
            n_ref_end = np.int(n_refugees/split)
            envy_init = np.zeros((n_localities, n_localities))

            for _ in range(split):
                # Assign refugees within lap
                observed_submatrix = observed_matrix[n_ref_start:n_ref_end, :]
                drm = dfa.assign(observed_submatrix , vector_quotas=split_quotas)
                drm_legacy = dfa.assign(observed_submatrix , vector_quotas=split_quotas, envy_init=envy_init)
                seq = dfa.assign_seq(observed_submatrix , vector_quotas=split_quotas)
                
                # Append lap assigment to assignment arrays
                assignments['DRM'] = np.append(assignments['DRM'], drm.assignment, axis=0)
                assignments['DRM with legacy'] = np.append(assignments['DRM with legacy'], drm_legacy.assignment, axis=0)
                assignments['Sequential'] = np.append(assignments['Sequential'], seq.assignment, axis=0)
                
                # Update initial envy (for legacy)
                envy_init = np.int32(evl.calc_envy(assignments['DRM with legacy'], demand_matrix[:n_ref_end,:]))

                # Update utilities
                n_ref_start += np.int(n_refugees/split)
                n_ref_end += np.int(n_refugees/split)

            # calculate end-of-period measures
            for method in assignments.keys():
                max_envy = np.amax(evl.calc_envy(assignments[method], demand_matrix), axis=1)
                fair0 = np.mean(max_envy>0)
                fair1 = np.mean(max_envy>=envy1_limit)
                misallocated = ((1-demand_matrix)*assignments[method]*np.amax(demand_matrix, axis=1, keepdims=True))
                effic = np.sum(misallocated)/np.sum(np.sum(demand_matrix, axis=1)>0)

                # append to result list
                results.append([sim, error, split, method, fair0, fair1, effic])
    
    # pass results to output
    return results

def simulate_over_quota_splits_mp(vector_quotas, n_simulations, errorlist, splits, envy1_limit=2, parallel_processes=None):

    # Setup a list of processes that we want to run
    if parallel_processes is None:
        parallel_processes = mp.cpu_count()
    print('Running over {} processes'.format(parallel_processes))
    pool = mp.Pool(parallel_processes)
    
    args = (vector_quotas, errorlist, splits, envy1_limit)
    output = [pool.apply_async(simulate_over_quota_splits_mp_func, (sim,) + args) for sim in range(n_simulations)]
    results = [p.get() for p in output]

    pool.close()

    results = list(itertools.chain.from_iterable(results))
    results = pd.DataFrame(
        results, 
        columns=['wave','error','splits', 'method', 'Envy by 0', 'Envy by {}'.format(envy1_limit) , 'Inefficiency']
        )
    return results

def simulate_conjecture_mp(n_simulations, n_refugees, n_municipalities, seed=0, parallel_processes=None):
    
    # Setup a list of processes that we want to run
    if parallel_processes is None:
        parallel_processes = mp.cpu_count()
    print('Running over {} processes'.format(parallel_processes))
    pool = mp.Pool(parallel_processes)

    # Collect function arguments
    args = (n_refugees, n_municipalities, seed)
    
    # Simulation
    output = [pool.apply_async(simulate_conjecture_mp_func, (sim,) + args) for sim in range(n_simulations)]
    results_list = [p.get() for p in output]
    
    pool.close()
    results = {}
    results['demanded'] = [r[0] for r in results_list]
    results['nondemanded'] = [r[1] for r in results_list]
    
    # Aggregate frequency dictionaries
    for res_type in results.keys():
        results[res_type] = [
            sum_dicts([iteration[refugee] for iteration in results[res_type]]) 
            for refugee in range(len(results[res_type][0]))
            ]
        # Turn into frequency dataframes
        results[res_type] = pd.DataFrame(results[res_type]).fillna(0).astype(int)
    
    return results

def simulate_conjecture_mp_func(sim, n_refugees, n_municipalities, seed):
    np.random.seed(sim + seed)

    # Generate random refugee flow
    p1 = np.random.uniform()
    p0 = np.random.uniform(0, 1-p1)
    which = np.random.randint(2)
    p_nond= which*p1 + (1-which)*p0
    p_over= which*p0 + (1-which)*p1
    autocorrelation = np.random.uniform()
    demand_matrix = flg.simulate_flow(n_refugees, n_municipalities, 
                                           p_nond = p_nond, 
                                           p_over = p_over, 
                                           autocorrelation = autocorrelation
                                          )

    # assign refugees
    assignment = dfa.assign(demand_matrix)

    # Calculate max envy
    results = {}
    for ref_type in ["demanded", "nondemanded"]:
        results_by_refugee = []
        for k in range(n_refugees):
            envy = assignment.get_envy(refugee=k, ref_type=ref_type)
            np.fill_diagonal(envy, -n_refugees)
            envy_max = np.amax(envy, axis=1)
            # transform maximum envies in distribution dictionary
            distribution = Counter(envy_max)
            # append distribution dictionary to results_by_refugee
            results_by_refugee.append(distribution)
        results[ref_type] = results_by_refugee
    

    return results['demanded'], results['nondemanded']

def sum_dicts(list_of_dicts):
    final_dict = {}
    for d in list_of_dicts:
        for k, v in d.items():
            try:
                final_dict[k] += v
            except:
                final_dict[k] = v
    return final_dict