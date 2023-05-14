'''

Linear Plotting of Properties

'''

import numpy
import matplotlib.pyplot as plt

def linear_plots(*prop, **kwargs):
    plt.close('all')
    figsize = (9,4*get_plot_num(*prop)) if not 'figsize' in kwargs else kwargs['figsize']
    sharex = True if not 'sharex' in kwargs else kwargs['sharex']
    xlabel = 't' if not 'xlabel' in kwargs else kwargs['xlabel']
    ylabel = '' if not 'ylabel' in kwargs else kwargs['ylabel']
    xlim = None if not 'xlim' in kwargs else kwargs['xlim']
    ylim = None if not 'ylim' in kwargs else kwargs['ylim']
    title = '' if not 'title' in kwargs else kwargs['title']
    name = "test" if not 'name' in kwargs else kwargs['name']
    sim = prop[0].sim

    fig, axs = plt.subplots(get_plot_num(*prop), 1, sharex=sharex, figsize=figsize)
    plt.suptitle(title)
    for ax in axs:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

    offset=0
    for index in range(len(prop)):
        linear_plot(axs, prop[index], index+offset, **kwargs)
        if prop[index] != 0:
            offset += numpy.prod(numpy.array(prop[index].shape))

    plt.tight_layout()
    plt.savefig('./' + name + '.png', bbox_inches='tight', dpi=200)


def linear_plot(axs, prop, index, **kwargs):
    sim = prop.sim
    
    yvals = prop.get()
    num_plots = 1 if prop.shape == 0 else numpy.prod(numpy.array(prop.shape))
    axs[index].set_title(prop.descriptor)
    for i in range(num_plots):
        axs[index+i].plot(numpy.arange(prop.sim.tsteps), yvals)

def get_plot_num(*prop):
    prop_num = 0
    for p in prop:
        if p.shape == 0:
            prop_num += 1
        else:
            prop_num += numpy.prod(numpy.array(p.shape))
    return prop_num

