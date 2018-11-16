import numpy as np

#####################
### Define Object ###
#####################
class ReturnAssignment(object):
    def __init__(self, assignment, pi, sigma, begin_quotas, end_quotas,
                diagnostic, scores, envy_final, envy_init):

        self.assignment = assignment
        self.pi = pi
        self.sigma = sigma
        self.begin_quotas = begin_quotas
        self.end_quotas = end_quotas
        self.diagnostic = diagnostic
        self.scores = scores
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
            acceptance = self.scores
        else:
            acceptance = correct_scores

        if ref_type is None:
            Utility = U(acceptance[:refugee,:])
        elif ref_type=="demanded":
            Utility = acceptance[:refugee,:]
        elif ref_type=="nondemanded":
            Utility = acceptance[:refugee,:] - 1

        Assignment_at_k = np.copy(self.assignment[:refugee,:])
        Utility_at_k =np.dot(np.transpose(Utility),Assignment_at_k)
        envy_at_k = Utility_at_k - np.transpose(np.diag(Utility_at_k)[np.newaxis]) + self.envy_init
        return envy_at_k


#####################
##### Functions #####
#####################

def U(num):
  return 2 * num - 1

def assign_random(scores, vector_quotas=None, pi_init=None, sigma_init=None, envy_init=None):
    """
    Assigns refugee at random 

    Inputs: 

    assignment: A NxM matrix describing assignment
    scores:     The scoring matrix describing refugee-locality integration scores
    refugee:    The index of the refugee at which the envy matrix should be calculated. If equal to None, then the envy is calculated after the assigment of the last refugee in the flow
    """
    ################################
    ########## INITIALIZE ##########
    ################################
    # Initialize number of asylum seekers and municipalities
    no_asylum_seekers = scores.shape[0]
    no_municipalities = scores.shape[1]

    # Assign default values for optional arguments
    if vector_quotas is None:
        # Assign a non-binding quotas equal to the number of refugees
        vector_quotas = np.ones((no_municipalities), dtype=np.int32) * no_asylum_seekers
        fast_version = True
    else:
        fast_version = False

    assert (np.sum(vector_quotas) >= no_asylum_seekers), "Quotas total ({}) is insufficient \
        to accomodate all asylum seekers ({})".format(np.sum(vector_quotas), no_asylum_seekers)

    # Initialize rankings
    if envy_init is None:
        # No municipality envies any other
        envy_init = np.zeros((no_municipalities, no_municipalities), dtype=np.int32)

    # Initialize matrix assignment (NxM). Initially, nobody is assigned
    matrix_assignment = np.zeros((no_asylum_seekers, no_municipalities), dtype=np.int8)

    quotas = np.copy(vector_quotas)

    ##############################
    ###### Assignment Phase ######
    ##############################
    if fast_version:
        random_mun = np.random.randint(no_municipalities, size=no_asylum_seekers)
        matrix_assignment[np.arange(no_asylum_seekers), random_mun] = 1

    else:
        for k in range(0, no_asylum_seekers):
            m_assigned = np.random.choice(np.nonzero(quotas)[0])

            matrix_assignment[k,m_assigned] = 1
            quotas[m_assigned] += -1

    ###############################
    ######## RECORD OUTPUT ########
    ###############################
    begin_quotas = vector_quotas
    end_quotas = vector_quotas
    assignment = matrix_assignment
    diagnostic = None
    envy_final = envy_init
    envy_init = envy_init
    pi=None
    sigma= None
    scores = scores
    return ReturnAssignment(assignment, pi, sigma, begin_quotas, end_quotas, diagnostic,
                            scores, envy_final, envy_init)


def assign_seq(scores, vector_quotas=None, envy_init=None):
    """
    Assigns refugees sequentially until quotas are satisfied 

    Inputs: 

    assignment: A NxM matrix describing assignment
    scores:     The scoring matrix describing refugee-locality integration scores
    refugee:    The index of the refugee at which the envy matrix should be calculated. If equal to None, then the envy is calculated after the assigment of the last refugee in the flow
    """
    
    ################################
    ########## INITIALIZE ##########
    ################################
    # Initialize number of asylum seekers and municipalities
    no_asylum_seekers = scores.shape[0]
    no_municipalities = scores.shape[1]

    # Assign default values for optional arguments
    if vector_quotas is None:
        # Assign a non-binding quotas equal to the number of refugees
        vector_quotas = np.ones((no_municipalities), dtype=np.int32) * no_asylum_seekers

    assert (np.sum(vector_quotas) >= no_asylum_seekers), "Quotas total ({}) is insufficient \
        to accomodate all asylum seekers ({})".format(np.sum(vector_quotas), no_asylum_seekers)

    # Initialize rankings
    if envy_init is None:
        # No municipality envies any other
        envy_init = np.zeros((no_municipalities, no_municipalities), dtype=np.int32)

    # Initialize matrix assignment (NxM). Initially, nobody is assigned
    matrix_assignment = np.zeros((no_asylum_seekers, no_municipalities), dtype=np.int8)

    quotas = np.copy(vector_quotas)

    ##############################
    ###### Assignment Phase ######
    ##############################
    m_assigned = 0
    for k in range(0, no_asylum_seekers):
        while quotas[m_assigned]== 0:
            m_assigned += 1
            if m_assigned == no_municipalities:
                m_assigned = 0

        matrix_assignment[k,m_assigned] = 1
        quotas[m_assigned] += -1
        m_assigned += 1
        if m_assigned == no_municipalities:
            m_assigned = 0

    ###############################
    ######## RECORD OUTPUT ########
    ###############################
    pi = None
    sigma = None
    begin_quotas = vector_quotas
    end_quotas = quotas
    assignment = matrix_assignment
    diagnostic = None
    envy_final = envy_init
    envy_init = envy_init
    return ReturnAssignment(assignment, pi, sigma, begin_quotas, end_quotas, diagnostic,
                            scores, envy_final, envy_init)


def assign(scoring_matrix, vector_quotas=None, pi_init=None, sigma_init=None, envy_init=None):
    ################################
    ########## INITIALIZE ##########
    ################################
    # Initialize number of asylum seekers and municipalities
    no_asylum_seekers = scoring_matrix.shape[0]
    no_municipalities = scoring_matrix.shape[1]

    # Assign default values for optional arguments
    if vector_quotas is None:
        # Assign a non-binding quotas equal to the number of refugees
        vector_quotas = np.ones((no_municipalities), dtype=np.int32) * no_asylum_seekers

    # Identify municipalities belonging to the economy: creates a vector 1xM
    # where each element is equal to 1 if the municipality is still in the economy
    remaining_municipalities = np.array(1 * (vector_quotas > 0))

    if pi_init is None:
        # Assign a priority ranking according to position in matrix of remaining municipalities
        pi_init = np.cumsum(remaining_municipalities) * remaining_municipalities
    if sigma_init is None:
        # Assign a rejection ranking according to position in matrix of remaining municipalities
        sigma_init = np.cumsum(remaining_municipalities) * remaining_municipalities
    if envy_init is None:
        # No municipality envies any other
        envy_init = np.zeros((no_municipalities, no_municipalities), dtype=np.int32)

    # Initialize dynamic rankings
    sigma = np.copy(sigma_init)
    pi = np.copy(pi_init)

    # Initialize matrix assignment (NxM). Initially, nobody is assigned
    matrix_assignment = np.zeros((no_asylum_seekers, no_municipalities), dtype=np.int32)

    # Create copy of scoring_matrix - modifiable if M exists from the economy
    scoring_matrix_work = np.copy(scoring_matrix)
    quotas = np.copy(vector_quotas)

    # For convenience, calculate the utility of each refugee to each municipality 
    # According to the function U() defined above
    Utility_refugees = U(scoring_matrix)

    # initialize dynamic envy matrix
    envy = np.copy(envy_init)

    ################################
    ######## BEGIN THE LOOP ########
    ################################
    for k in range(0, no_asylum_seekers):
        # cathegorize asylum seeker
        if np.sum(scoring_matrix_work[k]*remaining_municipalities) == 0:
            refugee_demand = 0
        elif np.sum(scoring_matrix_work[k]*remaining_municipalities) == 1:
            refugee_demand = 1
        else:
            refugee_demand = 2

        ##############################
        ###### Assignment phase ######
        ##############################

        # If asylum seeker overdemanded or demanded, find the lucky municipality who both
        # wants him and is first in the priority ranking pi (excluding zeros)
        if refugee_demand > 0:
            pi_accept = scoring_matrix_work[k] * pi
            nz_pi_accept_ind = np.nonzero(pi_accept)
            temp = np.argmin(pi_accept[nz_pi_accept_ind])
            m_assigned = nz_pi_accept_ind[0][temp]

        # If asylum seeker not demanded, assign asylum seeker to first municipality
        # in rejection list sigma (excluding zeros)
        if refugee_demand == 0:
            nz_sigma_ind = np.nonzero(sigma)
            temp = np.argmin(sigma[nz_sigma_ind])
            m_assigned = nz_sigma_ind[0][temp]

        # Assign refugee k to municipality m_assigned
        matrix_assignment[k][m_assigned] = 1

        ###################################
        ######## Update envy phase ########
        ###################################

        # Next, the envy matrix for the municipalities will be updated given the above assignment
        # The envy matrix is to be interpreted as municipality in row a envies
        # municipality in row b by envy[a][b]

        # I am using two tricks here to vectorize the code.
        # First, I use the function U(x) = 2x-1 to calculate the
        # "utility" of each refugee for each municipality, Utility_refugees.
        # The function returns 1 if the refugee is acceptable (x=1)
        # and -1 if the refugee is not acceptable (x=-1)
        # Second, given the allocation of refugees stored in matrix_assignment, a
        # utility matrix, representing each municipality's utility given their allocation on the
        # diagonal and each municipality's utility given the allocation of other municipalities
        # on points off the diagonal, can be calculated as
        # Utility_assignments =  Utility_refugees'matrix_assignment
        # The envy matrix is then Utility_assignments minus
        # the diagonal Utility_assignments (element-wise) - positive envy in row
        # i and column j means that municipality i envies j (would like to swap)
        # by a positive amount
        vector_utility = Utility_refugees[k]
        vector_assignment = matrix_assignment[k]
        Utilities_k = np.dot(np.transpose(vector_utility[np.newaxis]), vector_assignment[np.newaxis])
        
        envy_k = Utilities_k - np.transpose(np.diag(Utilities_k)[np.newaxis])

        # add k-1 envy
        envy = envy + envy_k

        # set envy (by and of) excluded municipalities to zero by multiplying rows
        # and columns of excluded municipalities by 0
        envy = envy * remaining_municipalities * remaining_municipalities[np.newaxis].T

        ###################################
        ###### Update rankings phase ######
        ###################################
        quotas[m_assigned] = quotas[m_assigned] - 1
        # If the quota for the municipality is filled, remove it from the economy and
        # rearrange the rankings accordingly

        if quotas[m_assigned] == 0:
            remaining_municipalities[m_assigned] = 0
            pi[pi > pi[m_assigned]] += -1
            pi[m_assigned] = 0
            sigma[sigma > sigma[m_assigned]] += -1
            sigma[m_assigned] = 0

        # If the quota is not filled, rankings needs to be updated according
        # to the mechanisms described in the paper, unless the refugee is demanded,
        # in which case the ranks stay the same
        else:
            if refugee_demand == 1:
                pass
            # If refugee is not demanded, we need to identify \pi^N(k), called pi_constant,
            # the rank of the municipality with the highest position in pi[k] which is
            # envied (positive numbers - should be at most 1) by municipality m_assigned
            # Then municipality m_assigned gets the rank pi_constant if pi_constant<m_assigned, 
            # And keeps the same rank otherwise. That is, pi_constant<pi[m_assigned] every 
            # municipality with rank between (including) pi_constant and the rank of
            # m_assigned (excluded) scales down a position (+= +1) and m_assigned gets pi_constant.
            # If there's no municipality that m_assigned envies with a better rank
            # , m_assigned keeps its rank
            # m_assign also gets lowest position in rejection order sigma
            # Meaning everyone with rank strictly higher than that of m_assigned gets += -1
            elif refugee_demand == 0:
                # Find pi_constant
                pi_envied = pi * np.array(envy[m_assigned] > 0)
                # If no envied municipality
                if pi_envied[np.nonzero(pi_envied)].size == 0:
                    pi_constant = pi[m_assigned]
                else:
                    pi_constant = np.min(pi_envied[np.nonzero(pi_envied)])
                # If pi_constant<pi[m_assigned], rearrange rank pi
                if pi_constant<pi[m_assigned]:
                    pi[np.where(np.logical_and(pi >= pi_constant, pi < pi[m_assigned]))] += +1
                    pi[m_assigned] = pi_constant
                # Rearrange rank sigma
                sigma_max = np.max(sigma)
                sigma[np.where(sigma > sigma[m_assigned])] += -1
                sigma[m_assigned] = sigma_max

            # If refugee is overdemanded, needs to identify \sigma^O(k), called sigma_constant,
            # the rank of the municipality with the highest position in sigma[k]
            # envying municipality m_assigned (the m_assigned COLUMN of the envy matrix)
            # (if positive number, municipality in row i would like to get the allocation
            # of municipality in column j).
            # Then municipality m_assigned gets the rank sigma_constant if 
            # sigma_constant<sigma[m_assigned], and stays the same otherwise. That is, every municipality
            # with rank between (including) sigma_constant and the rank of m_assigned  (excluded)
            # scales down a position (+= +1) and m_assigned gets sigma_constant
            # m_assign also gets lowest position in priority order pi
            # Meaning everyone with rank strictly higher than that of m_assigned gets += -1
            elif refugee_demand == 2:
                # Find sigma_constant
                sigma_envying = sigma * np.array(envy[:,m_assigned] > 0)
                # If no envying municipality
                if sigma_envying[np.nonzero(sigma_envying)].size == 0:
                    sigma_constant = sigma[m_assigned]
                else:
                    sigma_constant = np.min(sigma_envying[np.nonzero(sigma_envying)])
                # If sigma_constant<sigma[m_assigned], rearrange rank sigma
                if sigma_constant<sigma[m_assigned]:
                    sigma[np.where(np.logical_and(sigma >= sigma_constant, sigma < sigma[m_assigned]))] += +1
                    sigma[m_assigned] = sigma_constant
                # Rearrange rank pi
                pi_max = np.max(pi)
                pi[np.where(pi > pi[m_assigned])] += -1
                pi[m_assigned] = pi_max

    ###############################
    ######## RECORD OUTPUT ########
    ###############################
    begin_quotas = vector_quotas
    end_quotas = quotas
    assignment = matrix_assignment
    diagnostic = None
    envy_init = envy_init
    envy_final = envy
    return ReturnAssignment(assignment, pi, sigma, begin_quotas, end_quotas, diagnostic,
                            scoring_matrix, envy_final, envy_init)
