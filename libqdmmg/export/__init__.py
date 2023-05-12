'''

author: Linus Bjarne Dittmer

This module handles export of data to comprehensible filetypes.

'''

import libqdmmg.export.linear_plot as lp
import libqdmmg.export.density_plot as dp

def linear_plot(axs, prop, index, **kwargs):
    lp.linear_plot(axs, prop, index, **kwargs)

def linear_plots(*prop, **kwargs):
    lp.linear_plots(*prop, **kwargs)

def density_plots(wp, **kwargs):
    dp.density_plots(wp, **kwargs)

def density_plot(ax, wp, index, res, dist, desc, **kwargs):
    dp.density_plot(ax, wp, index, res, dist, desc, **kwargs)
