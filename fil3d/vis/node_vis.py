"""
node vis lib
"""

import matplotlib.pyplot as plt
import numpy as np

from fil3d.galfa import galfa_const
from fil3d.structs import util as struct_util
from fil3d.util import moments


def vis_node_mask_moments(mask_node, mask_name, save_fig=False, save_dir=None,
                          save_name=None, figsize=None, return_fig=False):
    """node mask contour with axis of least 2nd moment
    Arguments:
        mask_node {maskNode} -- node
        mask_name {str} -- name of node
    Keyword Arguments:
        save_fig {bool} -- (default: {False})
        save_dir {str} -- (default: {None})
        save_name {str} -- (default: {None})
        figsize {tuple of ints (x,y)} -- (default: {None})
        return_fig {bool} -- (default: {False})
    """

    if save_fig and save_name is None:
        save_name = mask_name
    if figsize is None or type(figsize) is not tuple or len(figsize) != 2:
        figsize = (10, 10)

    mask = mask_node.mask

    plot_corners = struct_util.get_node_plot_corners(mask_node)

    x_bar, y_bar, theta_1, theta_2, roundness = moments.get_node_mask_orientation_info(mask_node)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.imshow([[], []], extent=plot_corners, origin='lower', cmap='gray')
    ax.contour(mask, colors='black', extent=plot_corners)

    theta_1_scale = np.sqrt(mask_node.mask_size) / 2
    theta_2_scale = theta_1_scale * roundness

    ax.plot([x_bar - theta_1_scale * np.cos(theta_1), x_bar + theta_1_scale * np.cos(theta_1)],
            [y_bar - theta_1_scale * np.sin(theta_1), y_bar + theta_1_scale * np.sin(theta_1)], color='red')
    ax.plot([x_bar - theta_2_scale * np.cos(theta_2), x_bar + theta_2_scale * np.cos(theta_2)],
            [y_bar - theta_2_scale * np.sin(theta_2), y_bar + theta_2_scale * np.sin(theta_2)], color='green')

    fig.suptitle(mask_name)
    ax.set_title('Mask Axis of Min/Max Image Second Moments (Roundness: {0:.5f})'.format(roundness))

    if save_fig:
        fig.savefig(save_dir + save_name)

    if return_fig:
        return fig

    plt.clf()


def vis_mask_sky_dist(mask_node, mask_name, save_fig=False, save_dir=None, save_name=None, verbose=False):
    # WIP
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow([[], []], extent=[0, galfa_const.GALFA_X_STEPS, 0, galfa_const.GALFA_Y_STEPS], origin='lower')

    this_mask = mask_node.mask
    this_plot_corners = [mask_node.corner_BL[0], mask_node.corner_TR[0],
                         mask_node.corner_BL[1], mask_node.corner_TR[1]]
    ax.contour(this_mask, colors='r', extent=this_plot_corners)

    ax.set_title('{0} Sky Position'.format(mask_name))

    if verbose:
        fig.show()

    if save_fig:
        fig.savefig(save_dir + save_name)

    plt.clf()


def vis_mask_with_data(node, data, key=None, verbose=False, return_fig=False):
    if len(data.shape) == 3:  # v info will be taken into account
        data = data[node.v_slice_index[0]]
    elif len(data.shape) == 2:  # v info will not be taken into account
        data = data
    else:
        raise ValueError('data is neither 2d or 3d')

    corner_min = node.corner_min
    corner_max = node.corner_max
    mask = node.mask

    cut_data = data[corner_min[0]: corner_max[0], corner_min[1]: corner_max[1]]

    min_val, max_val = np.min(cut_data), np.max(cut_data)
    min_cut, max_cut = np.percentile(cut_data, 5), np.percentile(cut_data, 95)

    fig, ax = plt.subplots()

    ax.imshow(cut_data.clip(min_cut, max_cut), origin='lower', cmap='binary')
    ax.contour(mask, alpha=.5, colors='red', linewidths=.3)

    ax.set_title('{0} ({1})'.format(key, [corner_min, corner_max]))

    fig.tight_layout()

    if verbose:
        fig.show()

    if return_fig:
        return fig

    plt.clf()
