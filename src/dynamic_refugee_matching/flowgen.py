import numpy as np
import pandas as pd

def simulate_flow(no_asylum_seekers, no_municipalities, p_nond=0, p_over=0.5, autocorrelation=0):
    autocorrelation = 0.5 + autocorrelation/2
    matrix = np.zeros((no_asylum_seekers, no_municipalities), dtype=np.int16)
    matrix[:int(np.around(p_over*no_asylum_seekers))][:] =1

    base_vector = np.random.choice(2,no_asylum_seekers-int(np.around((p_over + p_nond)*no_asylum_seekers)))

    r_matrix = np.repeat(base_vector[np.newaxis].T,no_municipalities, axis=1)
    r_matrix[np.where(r_matrix==1)] = np.random.choice(
        2, 
        r_matrix[np.where(r_matrix==1)].shape,
        p=[1-autocorrelation, autocorrelation]
        )

    r_matrix[np.where(r_matrix==0)] = np.random.choice(
        2, r_matrix[np.where(r_matrix==0)].shape,
        p=[autocorrelation, 1-autocorrelation])
    matrix[int(np.around((p_over + p_nond)*no_asylum_seekers)):][:] = r_matrix[:][:]

    np.random.shuffle(matrix)
    return matrix


def scramble_matrix(matrix_acceptability, false_positives=0, false_negatives=0):
	"""
	The function creates an imperfect copy of a refugee acceptance matrix, 
	with a given % of false negatives and false positives (default: 2.5% each)  
	"""
	fp = (np.random.random(matrix_acceptability.shape)<false_positives*2/(1- (false_negatives)*2/100)/100)*1
	fn = (np.random.random(matrix_acceptability.shape)<false_negatives*2/100)*1
	# change false positives
	new_matrix = (matrix_acceptability+fp>0)*1
	# change false negatives
	new_matrix = (new_matrix - fn>0)*1
	return new_matrix

def add_error(demand_matrix, err=0):
    error = err*2/100
    n_refugees = np.size(demand_matrix, axis=0)
    to_be_switched = np.zeros(demand_matrix.shape, dtype=np.int8)
    rowstoswitch = np.random.random(n_refugees)<error
    to_be_switched[rowstoswitch] = np.random.randint(2, size=(np.sum(rowstoswitch), np.size(demand_matrix, axis=1)))
    new_matrix = demand_matrix + to_be_switched -2*to_be_switched*demand_matrix
    return new_matrix

def report_errors(true_matrix, false_matrix):
	matrix_eval = 2*true_matrix + false_matrix
	df_eval = pd.DataFrame(np.array(np.unique(matrix_eval, return_counts=True)).T,columns=['value','prop'])
	df_eval['prop'] = df_eval['prop']/(matrix_eval.shape[0]*matrix_eval.shape[1])
	#print(df_eval)
	print('False positives: ', float(df_eval[df_eval['value']==1]['prop']))
	print('False negatives: ', float(df_eval[df_eval['value']==2]['prop']))
