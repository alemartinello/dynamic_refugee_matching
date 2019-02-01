import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler
import os
from jupyterthemes import jtplot

def dynamic_graphs(results, measure, legend=True, show=True, titles=True, saveas = None):
    # Clear matplotlib memory
    plt.close('all')
    cmap = cm.get_cmap('gist_heat')
    custom_cycler = (
        cycler(color=['black', cmap(0.8), cmap(0.7), cmap(0.5), cmap(0.4)]) +
        cycler(linestyle=['-', '-', '-.', '--', ':' ]) +
        cycler(lw=[2, 1, 1, 1, 1])
        )

    ## Code for graphs
    jtplot.reset()
    
    plt.close('all')
    fig, axarr = plt.subplots(1, 2, sharey=False,figsize=(16,4))

    for index, country in enumerate(results.keys()):
        # Set graph properties
        if measure == 'inefficiency':
            axarr[index].set_ylim(-5, 20)
        else:
            axarr[index].set_ylim(-5, 100)
        axarr[index].set_prop_cycle(custom_cycler)
        axarr[index].set_xlabel('# of assigned asylum seeker')
        if titles is True:
            axarr[index].set_title(country)
        # Plots
        for assignment in results[country].keys():
            axarr[index].plot(results[country][assignment][measure]*100, label=assignment)
        
        if legend is True and index==0:
            axarr[0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.2),
                        frameon=False, ncol=6, handlelength=4)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        #           frameon=False, ncol=3, handlelength=4)
    
    if show is True:
        plt.show()
    if saveas is not None:
        current_dir = os.path.dirname(__file__)
        fig.savefig(os.path.join(current_dir, '../../', saveas), bbox_inches = 'tight')
    
