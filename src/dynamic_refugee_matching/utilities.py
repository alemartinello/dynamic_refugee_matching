import time
import os
import pickle
import datetime
import numpy as np

def f_writetime(time):
    hours = np.int8(np.floor(time/3600))
    minutes = np.int8(np.floor((time - hours*3600)/60))
    seconds = np.floor(time - hours*3600 - minutes*60)
    
    return str(hours) + " hour(s), " + str(minutes) + " minute(s), and " + str(seconds) + " seconds"

def print_time(file, n_simulations, n_refugees, elapsed_time):
    current_dir = os.path.dirname(__file__)
    f = open(os.path.join(current_dir, '../../', file),'a')
    f.write(
        'Date: ' + str(datetime.date.today()) + '\n' +
        'Parameters: ' + '\n' + 
        '  - # simulations: ' + str(n_simulations) + '\n'
        '  - # refugees   : ' + str(n_refugees) + '\n'
        'Total running time: ' + f_writetime(elapsed_time) + '\n\n'
    )
    f.close()

def save_results(results, file):
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, '../../results/data/', file), 'wb') as output_file:
        pickle.dump(results, output_file)

def load_results(file):
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, '../../results/data/', file), 'rb') as input_file:
        results = pickle.load(input_file)
    return results