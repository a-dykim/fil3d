import numpy as np
import pickle 
from fil3d.util import moments
import random
import astropy.units as u
from fil3d.util import cube_util
from astropy.io import fits
import matplotlib.pyplot as plt
from fil_finder import FilFinder2D, Filament2D
from fil_finder.tests.testing_utils import generate_filament_model
from fil3d import MaskObjNode
from fil3d.structs import util as struct_util
from fil3d.util import moments
from fil3d.util import tree_dict_util 
from fil3d.structs import mask_obj_node_tree as maskTree

DEFAULT_DATA_DIR = '../../pickled_dicts/full_sky_gaussian_30_1.0/'
DEFAULT_SAVE_DIR = '../../pickled_dicts/fourth_batch/'


def ordering_corners(corner):
    """
    Just place ordering in the right place for extent plotting
    """
    aligned = [corner[0][1], corner[1][1], corner[0][0], corner[1][0]]
    return aligned

def construct_filaments(vslice, header, v_channel):
    """
    Find a filamentary structure at a given velocity slice.
    
    Returns:
    node of masked objects (for the purpose of checking overlaps)
    """
    header['BUNIT'] = 'K' # to be read by astropy.units
    scale_width = 0.1 * u.pc # empirical 
    filament = FilFinder2D(vslice, header=header, distance=100.*u.pc, beamwidth=10.*u.arcmin)
    filament.preprocess_image(flatten_percent=95)
    filament_masks, corners = filament.create_mask(smooth_size = scale_width/2, adapt_thresh=scale_width*2, 
                         size_thresh= 8*(scale_width*2)**2, border_masking=False, output_mask_objs=True)
    
    nmasks = len(filament_masks)
    
    node_dict = {}
    if filament_masks is None:
        pass
    else:
        for i in range(nmasks):
            try:
                mask_node = MaskObjNode(filament_masks[i], corners[i], v_channel)
                struct_util.add_node_to_dict(mask_node, node_dict)
            except AssertionError:
                pass

    return node_dict

def noderun_for_multichannel_modified(valid_slices, header, save_path=None):
    """
    Run for multiple velocity slices
    """
    all_nodes = {}
    nch = len(valid_slices) ## number of velocity slices
    for i in range(nch):
    #for i in range(nch, 0, -1): ## reverse test
        node_one = construct_filaments(valid_slices[i], header, i)
        all_nodes[i] = node_one
        #print (i)
    return all_nodes

def find_trees(nodes, overlap_thresh=.85, reverse_find=False):
    """
    Find trees (velocity coherent structure) from collection of nodes
    """
    trees = {} ## individual tree
    connected_trees = set()
    
    vindex = list(nodes.keys()) ## it is a list of velocity indices of a given node
    nchannel = len(vindex)
    
    for i in range(nchannel):
        #print ("vindex", vindex[i])
        current_node = nodes[vindex[i]]
        filaments = tree_dict_util.struct_util.sorted_struct_dict_keys_by_area(current_node, 'node')
        for j in filaments:
            mode = current_node[j]
            oldflag = tree_dict_util.match_and_add_node_onto_tree(mode, vindex[i], trees, overlap_thresh, continuous_tree_keys_set=connected_trees)
            if not oldflag: # if tree is empty raise
                new_tree = maskTree.newTreeFromNode(mode, verbose=False)
                struct_util.add_tree_to_dict(new_tree, trees)
                
        connected_trees = tree_dict_util.end_noncontinuous_trees(trees, vindex[i])
        #tree_dict_util.delete_short_dead_trees(trees, verbose=False)
    
    return trees

def filter_trees(trees, aspect='1_6', size=0, v_length=0):
    """
    Filter trees based on aspect ratio
    """
    filtered_trees = {}
    for k in trees:
        this_tree = trees[k]
        tree_roundness = moments.get_tree_mask_orientation_info(trees[k])[4]
        tree_v_length = this_tree.length
        tree_masked_size = this_tree.getTreeMaskedArea2D()
        if tree_roundness < moments.ROUNDNESS_AR_CONVERSION[aspect] and tree_v_length > v_length and tree_masked_size > size:
            filtered_trees[k] = this_tree
    return filtered_trees

def process_filaments(vslices, hdr, wkernel=15, overlap=0.85, usm_return=False):
    """
    Run all pipeline
    """
    usmed = np.zeros_like(vslices)
    for i in range(len(vslices)):
        usmed[i] = cube_util.umask(vslices[i], wkernel)
    
    nodes = noderun_for_multichannel_modified(usmed, hdr)
    pre_trees = find_trees(nodes, overlap_thresh=overlap)
    trees = filter_trees(pre_trees) # require length>2
    if usm_return is True:
        return trees, usmed
    else:
        return trees

hdu = fits.open(DEFAULT_DATA_DIR+"cube.fits")[0]

data = hdu.data
header = hdu.header
processed_data = process_filaments(data,header)
