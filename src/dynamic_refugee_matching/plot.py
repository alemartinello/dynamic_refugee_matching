import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
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
    
def temporal_quotas_graph(results, variable, show=True, saveas = None):    
    # Clear matplotlib memory
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax = sns.violinplot(x="splits", y=variable, hue="method", 
                        data=results, palette=['0.8', 'gray', 'black'], ax=ax)
    
    ax.set(xticklabels=['Year', 'Semester', 'Trimester', 'Month'],
           xlabel="",
           ylabel = "",
           #ylim = (0.04, 0.19)
          )
    ax.grid(True, axis='y', linestyle='--')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                      frameon=False, ncol=3)
    
    if show is True:
        plt.show()
    if saveas is not None:
        current_dir = os.path.dirname(__file__)
        fig.savefig(os.path.join(current_dir, '../../', saveas), bbox_inches = 'tight')

def temporal_quotas_saveallgraphs(results):
    
    for var in ['Envy by 0', 'Envy by 5', 'Inefficiency']:
        for err in results['error'].unique():
            temporal_quotas_graph(
                results[results['error']==err], 
                var, 
                show=False, 
                saveas = 'results/graphs/tempquotas_{}_{}.pdf'.format(var, err))

def conjecture_graphs(result_histograms, show=True, saveas = None):
    # Export figure
    plt.close('all')
    fig, axarr = plt.subplots(1, 2, sharey=False,figsize=(16,4))


    axarr[0].set_title('Demanded asylum seekers', y=1.02)
    axarr[0].set_ylim(-30,2)
    axarr[0].plot(result_histograms['demanded']['max'], color='gray', dashes=[6, 3], label='max')
    axarr[0].plot(result_histograms['demanded'][50], color='black', label='Median')
    axarr[0].plot(result_histograms['demanded']['min'], color='gray', dashes=[6, 3], label='min')
    axarr[0].set_xlabel('# of assigned asylum seeker')

    axarr[1].set_title('Non-demanded asylum seekers', y=1.02)
    axarr[1].set_ylim(-30,2)
    axarr[1].plot(result_histograms['nondemanded']['max'], color='gray', dashes=[6, 3], label='max')
    axarr[1].plot(result_histograms['nondemanded'][50], color='black', label='Median')
    axarr[1].plot(result_histograms['nondemanded']['min'], color='gray', dashes=[6, 3], label='min')
    axarr[1].set_xlabel('# of assigned asylum seeker')

    
    if show is True:
        plt.show()
    if saveas is not None:
        current_dir = os.path.dirname(__file__)
        fig.savefig(os.path.join(current_dir, '../../', saveas), bbox_inches = 'tight')

