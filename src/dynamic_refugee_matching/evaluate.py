"""
Description
"""
import numpy as np
import pandas as pd
def evaluate_efficiency_case(scores, AssignmentObject):
    print(
        "Sum of demanded and over-demanded asylum seekers : {}".format(np.sum(np.max(scores, axis=1))),
        "\nSum of well-matched asylum seekers               : {}".format(np.sum(scores*AssignmentObject.assignment))
    )

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
    Utility_at_k =np.dot(np.transpose(Utility),Assignment_at_k)
    envy_at_k = Utility_at_k - np.transpose(np.diag(Utility_at_k)[np.newaxis])
    return envy_at_k
    

def characterize_envy(assignments_dict, scores):
    # #localities envying another by more than 1 refugee
    # maximum envy
    # #localities envying another by more that 50% of average bundle size
    outlist = [[],[]]
    for key in assignments_dict.keys():
        outlist[0].append(np.sum(np.max(assignments_dict[key].get_envy(), axis=1)>0))
        outlist[1].append(np.max(assignments_dict[key].get_envy()))

    df = pd.DataFrame(outlist, columns=assignments_dict.keys(), \
        index = ['# localities envying another by more than 1 AS', 'Maximum envy']
        )

    return df

def characterize_assignments(assignments_dict):
    # #localities envying another by more than 1 refugee
    # maximum envy
    # #localities envying another by more that 50% of average bundle size
    outlist = [[],[],[],[]]
    for key in assignments_dict.keys():
        outlist[0].append(np.sum(np.max(assignments_dict[key].scores, axis=1)))
        outlist[1].append(np.sum(assignments_dict[key].scores*assignments_dict[key].assignment))
        outlist[2].append(np.sum(np.max(assignments_dict[key].get_envy(), axis=1)>0))
        outlist[3].append(np.max(assignments_dict[key].get_envy()))

    df = pd.DataFrame(outlist, columns=assignments_dict.keys(), \
        index = [
            'Sum of demanded and over-demanded asylum seekers', 
            'Sum of well-matched asylum seekers' ,
            '# localities envying another by more than 1 AS', 
            'Maximum envy'
            ]
        )

    return df