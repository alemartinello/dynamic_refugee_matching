import numpy as np 
import time
import dynamic_refugee_matching.flowgen as flg
import dynamic_refugee_matching.assignment as af
from dynamic_refugee_matching.utilities import f_writetime


def time_print(country, sim, interval, time):
    if sim%interval==0:
        print("Country:", country, "; Simulation", sim,". Elapsed time:", f_writetime(time))

def evaluate_allocations(true_scores, assignments, envy_limit_fraction):
    # Initialize misallocated counts
    mis_dem = {}
    for a_name in assignments.keys():
        mis_dem[a_name] = 0
    
    n_refugees = true_scores.shape[0]
    n_localities = true_scores.shape[1]
    n_demanded_refugees = np.cumsum(np.amax(true_scores, axis=1))

    # foreach refugee
    for k in np.arange(n_refugees):
        for name, assignment in assignments.items():
            # update misallocated count
            if (np.sum(true_scores[k])>0) and (np.sum(assignment[k][:]*true_scores[k][:])==0):
                mis_dem[name] += 1
            # calculate measures
            max_envy = np.amax(assignment.get_envy(refugee=k,real_acceptance=true_scores), axis=1)
            envy0 = np.mean((max_envy>0))
            envy1 = np.mean(max_envy>=np.int32(np.round((n_refugees/n_municipalities)*envy_limit_fraction)))
            
            #HERE!!!!#

            if (n_demanded_refugees[k] > 0) :
                effic = mis_dem[name]/n_demanded_refugees[k]
            else:
                effic = 0

def simulate_performance_witherror(n_refugees, n_simulations, sim_types, country_properties, errorslist, 
    print_time=True, benchmark = ['sequential']
    ):

    time_start = time.time()
    output = np.zeros((n_refugees, len(sim_types)*3))

    for country, countryinfo in country_properties.items():
        if print_time:
            time_print(country, sim, round(n_simulations/10), time.time() - time_start)
            n_localities = countryinfo[0]
            flowinfo = countryinfo[1]

        for sim in np.arange(n_simulations):
            
            #############################
            ### SIMULATE REFUGEE FLOW ###
            #############################
            # simulate demand/refugee flow
            true_scores = flg.simulate_flow(n_refugees, n_localities[country], 
                                            p_nond = flowinfo[0], 
                                            p_over = flowinfo[1], 
                                            autocorrelation = flowinfo[2]
                                            )

            #############################
            ##### ALLOCATE REFUGEES #####
            #############################
            assignments = {}
            # Benchmark allocations
            for mdl in benchmark:
                if mdl == 'sequential':
                    assignments[mdl] = af.assign_seq(true_scores)
                elif mdl == 'random':
                    assignments[mdl] = af.assign_random(true_scores)
            # AEM allocations
            for error in errorslist:
                assignments['err_{0}'.format(error)] = af.assign(flg.add_error(true_scores, error))


            #############################
            #### EVALUATE ALLOCATION ####
            #############################
            
            # Initialize misallocated counts
            mis_dem = {}
            for mdl in benchmark:
                mis_dem[mdl] = 0
            for error in errorslist:
                mis_dem['err_{0}'.format(error)] = 0
            
            n_demanded_refugees = np.cumsum(np.amax(true_scores, axis=1))

            # foreach refugee
            for k in np.arange(n_refugees):
                for val, atype in enumerate(sim_types):
                    # update misallocated count
                    if (np.sum(true_scores[k])>0) and (np.sum(assignments[atype].assignment[k][:]*true_scores[k][:])==0):
                        mis_dem[atype] += 1
                    # calculate measures
                    max_envy = np.amax(assignments[atype].get_envy(refugee=k,real_acceptance=true_scores), axis=1)
                    envy0 = np.mean((max_envy>0))
                    envy1 = np.mean((max_envy>=envy_limit))
                    if n_demanded_refugees[k]>0:
                        effic = mis_dem[atype]/n_demanded_refugees[k]
                    else:
                        effic = 0

                    # Update averages
                    output[k,val*3 + 0] = (output[k,val*3 + 0]*(sim) + envy0)/(sim+1)
                    output[k,val*3 + 1] = (output[k,val*3 + 1]*(sim) + envy1)/(sim+1)
                    output[k,val*3 + 2] = (output[k,val*3 + 2]*(sim) + effic)/(sim+1)

        nameslist = []
        for atype in sim_types:
            nameslist.append('envy0_'+atype)
            nameslist.append('envy1_'+atype)
            nameslist.append('effic_'+atype)