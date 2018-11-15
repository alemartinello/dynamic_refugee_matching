import numpy as np
import pandas as pd


#####################
### Define Object ###
#####################
class ReturnAssignment(object):
    def __init__(self, assignment, pi, sigma, begin_quotas, end_quotas,
                diagnostic, matrix_acceptability, envy_final, envy_init):

        self.assignment = assignment
        self.pi = pi
        self.sigma = sigma
        self.begin_quotas = begin_quotas
        self.end_quotas = end_quotas
        self.diagnostic = diagnostic
        self.matrix_acceptability = matrix_acceptability
        self.envy_final = envy_final
        self.envy_init = envy_init

    ##### Object method(s) #####
    def get_envy(self, refugee=None, correct_scores=None, ref_type=None):
        """
        Calculates the envy matrix of municipalities at the time of assigment of refugee k.
        Inputs: 

        refugee: The index of the refugee at which to compute the utility matrix
        correct_scores: The scoring matrix against which envy should be computed. If equal to None, 
                        it assumes the assigment has been performed according to the correct scoring matrix 
        ref_type: 
        """
        if refugee is None:
            refugee = self.assignment.shape[0]
        if correct_scores is None:
            acceptance = self.matrix_acceptability
        else:
            acceptance = correct_scores

        if ref_type is None:
            Utility = U(acceptance[:refugee,:])
        elif ref_type=="demanded":
            Utility = acceptance[:refugee,:]
        elif ref_type=="nondemanded":
            Utility = acceptance[:refugee,:] - 1

        Assignment_at_k = np.copy(self.assignment[:refugee,:])
        Utility_at_k =np.dot( 
            np.transpose(Utility),Assignment_at_k)
        envy_at_k = Utility_at_k - np.transpose(np.diag(Utility_at_k)[np.newaxis]) + self.envy_init
        return envy_at_k


#####################
##### Functions #####
#####################

def U(num):
  return 2 * num - 1

def calc_envy(assignment, scores, refugee=None):
    """
    Calculates envy matrix for a particular assignment given a scoring matrix, optionally at a specific refugee arrival

    Inputs: 
    assignment: A NxM matrix describing assignment
    scores:     The scoring matrix describing refugee-locality integration scores
    refugee:    the index of the refugee at which the envy matrix should be calculated. If equal to None, 
                thenthe envy is calculated after the assigment of the last refugee in the flow
    """
    if refugee is None:
        refugee = assignment.shape[0]

    Utility = U(scores[:refugee,:])

    Assignment_at_k = np.copy(assignment[:refugee,:])
    Utility_at_k =np.dot( 
    np.transpose(Utility),Assignment_at_k)
    envy_at_k = Utility_at_k - np.transpose(np.diag(Utility_at_k)[np.newaxis])
    return envy_at_k


def assign_random(matrix_acceptability, vector_quotas=None, pi_init=None, sigma_init=None, envy_init=None):
    """
    Calculates envy matrix for a particular assignment given a scoring matrix, optionally at a specific refugee arrival

    Inputs: 
    assignment: A NxM matrix describing assignment
    scores:     The scoring matrix describing refugee-locality integration scores
    refugee:    the index of the refugee at which the envy matrix should be calculated. If equal to None, 
                thenthe envy is calculated after the assigment of the last refugee in the flow
    """
    # Initialize number of asylum seekers and municipalities
    no_asylum_seekers = matrix_acceptability.shape[0]
    no_municipalities = matrix_acceptability.shape[1]

    # Assign default values for optional arguments
    if vector_quotas is None:
        # Assign a non-binding quotas equal to the number of refugees
        vector_quotas = np.ones(
            (matrix_acceptability.shape[1]), dtype=np.int32) * matrix_acceptability.shape[0]

    # Identify municipalities belonging to the economy: creates a vector 1xM
    # where each element is equal to 1 if the municipality is still in the economy
    remaining_municipalities = np.array(1 * (vector_quotas > 0))

    if pi_init is None:
        # Assign a priority ranking according to position in matrix of remaining municipalities
        pi_init = np.cumsum(remaining_municipalities) * \
            remaining_municipalities
        if sigma_init is None:
            # Irrelevant
            sigma_init = np.zeros(no_municipalities)

    # Initialize rankings
    pi = np.copy(pi_init)
    sigma = None
    if envy_init is None:
        # No municipality envies any other
        envy_init = np.zeros(
            (no_municipalities, no_municipalities), dtype=np.int32)

    # Assignment phase
    matrix_assignment = np.int8(np.zeros((no_asylum_seekers, no_municipalities)))
    random_mun = np.random.randint(no_municipalities, size=no_asylum_seekers)
    matrix_assignment[np.arange(no_asylum_seekers), random_mun] = 1

    ###############################
    ######## RECORD OUTPUT ########
    ###############################
    begin_quotas = vector_quotas
    end_quotas = vector_quotas
    assignment = matrix_assignment
    diagnostic = None
    envy_final = envy_init
    envy_init = envy_init
    return ReturnAssignment(assignment, pi, sigma, begin_quotas, end_quotas, diagnostic,
                            matrix_acceptability, envy_final, envy_init)
