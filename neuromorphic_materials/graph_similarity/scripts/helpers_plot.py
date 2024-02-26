import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd().rsplit('/', 1)[0]))

from matplotlib import pyplot as plt
import numpy as np
from neuromorphic_materials.graph_scripts.helpers.visual_utils import set_ticks_label, create_order_of_magnitude_strings, set_legend


def fitness_plot(ga, save_path=None, figname='fitness', figsize=(6, 4), figformat='.svg', show=False, max_fitness=True):

    if max_fitness:
        y_data = np.array(ga.saved_fitnesses).max(axis=1)
    else:
        y_data = np.array(ga.saved_fitnesses).mean(axis=1)
    x_data = range(ga.generations_completed)

    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    ax.plot(x_data, y_data, c='blue', linewidth=4)
    # ax.set_xlabel('Generation')
    # ax.set_ylabel('Fitness')
    # # Set font properties for the y-axis label
    font_properties = {'weight': 'bold', 'size': 'xx-large'}
    # y_label = ax.yaxis.get_label()
    # y_label.set_font_properties(font_properties)
    # x_label = ax.xaxis.get_label()
    # x_label.set_font_properties(font_properties)
    set_ticks_label(ax=ax, ax_type='y', data=y_data,
                    num=4, valfmt="{x:.2f}",
                    ax_label='Fitness',
                    )
    set_ticks_label(ax=ax, ax_type='x', data=x_data, ticks=x_data[::5],
                    num=4, valfmt="{x:.0f}",
                    ax_label='Generation',
                    )

    ax.tick_params(axis='both', which='major', width=2, length=7.5)
    # ax.tick_params(axis='both', which='minor', width=2, length=4)
    # set_legend(ax, title='', ncol=1, loc=0)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig('{:s}{:s}{:s}'.format(save_path, figname, figformat))
    if show:
        plt.show()
    else:
        plt.close()
