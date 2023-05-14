'''

author : Linus Bjarne Dittmer

'''

import numpy
import matplotlib.pyplot as plt

def density_plots(wp, **kwargs):
    res = 100 if not 'res' in kwargs else kwargs['res']
    dist = 2.5 if not 'dist' in kwargs else kwargs['dist']

    fig, axs = plt.subplots(1, wp.sim.dim, figsize=(6,6*wp.sim.dim))
    c = None
    if wp.sim.dim == 1:
        axs = tuple([axs])
    for index in range(wp.sim.dim):
        c = density_plot(axs[index], wp, index, res, dist, 'amplitude', **kwargs)
    
    plt.colorbar(c)
    plt.tight_layout()
    plt.savefig('./test.png', bbox_inches='tight', dpi=200)

def plot_content(wp, xv, t, desc='amplitude', amp='real'):
    a = wp.evaluate(xv, t)
    if desc == 'amplitude' and amp == 'real':
        return a.real
    elif desc == 'amplitude' and amp == 'imag':
        return a.imag
    elif desc == 'density':
        return abs(a)**2
    else:
        return 0

def density_plot(ax, wp, index, res, dist, desc, **kwargs):
    density = numpy.zeros((wp.sim.tsteps, res))
    xcross = numpy.linspace(-dist, dist, num=res)
    for t in range(wp.sim.tsteps):
        for i, x in enumerate(xcross):
            xv = numpy.zeros(wp.sim.dim)
            xv[index] = x
            density[t,i] = plot_content(wp, xv, t, desc)

    ax.set_xlabel("$t$")
    ax.set_ylabel(f"$x_{index}$")
    cm = None if not 'cmap' in kwargs else kwargs['cmap']
    if cm is None:
        cm = 'afmhot' if desc == 'density' else 'seismic'
    c = ax.imshow(density, extent=(0, wp.sim.tsteps*wp.sim.tstep_val, -dist, dist), interpolation='bilinear', cmap=cm)
    return c

def get_ascii_string(d, width, height):
    ascii_template = ' .:-=+*#%@'
    i_indices = numpy.linspace(0, len(d)-1, num=height)
    j_indices = numpy.linspace(0, len(d[0])-1, num=width)
    ascii_string = ""
    for i in i_indices:
        line = "| "
        for j in j_indices:
            dval = int(d[int(i),int(j)] * (len(ascii_template)-1))
            line += ascii_template[dval]
        ascii_string += line + " |\n"
    return ascii_string


def density_plot_ascii(wp, path, index, desc, **kwargs):
    res = 100 if not 'res' in kwargs else kwargs['res']
    dist = 2.5 if not 'dist' in kwargs else kwargs['dist']
    amp = 'real' if not 'amp' in kwargs else kwargs['amp']
    width = 80 if not 'width' in kwargs else kwargs['width']
    height = int(width*0.5) if not 'height' in kwargs else kwargs['height']

    density = numpy.zeros((wp.sim.tsteps, res))
    xcross = numpy.linspace(-dist, dist, num=res)
    for t in range(wp.sim.tsteps):
        for i, x in enumerate(xcross):
            xv = numpy.zeros(wp.sim.dim)
            xv[index] = x
            density[t,i] = plot_content(wp, xv, t, desc)
    
    density_min = numpy.min(density)
    density_max = numpy.max(density)
    density_norm = (density - density_min) / (density_max - density_min)
    string = get_ascii_string(density_norm, width, height)
    with open(path, 'w') as f:
        f.write(string)


