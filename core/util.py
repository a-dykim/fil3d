from __future__ import annotations
import logging
import os
import numpy as np

from .mask_node import MaskObjNode

log = logging.getLogger(__name__)


# Logging
def get_logger(name="fil3d", env_var="FIL3D_LOGLEVEL", default="INFO"):
    """
    Return a configured logger (level via env var).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.getenv(env_var, default).upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


# Mask helpers

def ensure_bool_mask(a):
    """
    Cast to boolean without copy when possible.
    """
    return a.astype(bool, copy=False) if a.dtype != bool else a


def bbox_from_mask(mask):
    """
    Return [[y0,x0],[y1,x1]] (half-open). For empty masks returns [[0,0],[0,0]].
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [[0, 0], [0, 0]]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return [[y0, x0], [y1, x1]]


def crop_mask_and_corners(mask, corners):
    """
    Crop a full-size mask down to its bbox; return (cropped_mask, adjusted_corners).
    handy for building compact nodes from label fields.
    """
    (y0, x0), (y1, x1) = corners
    cropped = mask[y0:y1, x0:x1]
    return cropped, [[y0, x0], [y1, x1]]


def count_true(mask):
    """
    Count True pixels.
    """
    return int(np.count_nonzero(mask))


# Node / slice constructors

def nodes_from_label_slice(labels_2d, v_index):
    """
    Turn a 2D label map (integers; 0=background) into a list of MaskObjNode.

    Each distinct positive ID becomes a node whose mask is cropped to its bbox and
    whose corners reference the original image coordinates.
    """
    labels_2d = np.asarray(labels_2d)
    if labels_2d.ndim != 2:
        raise ValueError("labels_2d must be 2D")

    out = []
    ids = np.unique(labels_2d)
    ids = ids[ids > 0]

    for k in ids:
        m = labels_2d == k
        if not m.any():
            continue
        corners = bbox_from_mask(m)
        cropped, corners = crop_mask_and_corners(m, corners)
        if cropped.size == 0:
            continue
        out.append(MaskObjNode(cropped.astype(bool, copy=False), corners, v_slice_index=int(v_index)))
    return out


def slices_from_labeled_stack(labels_3d):
    """
    Convert a (nv, ny, nx) labeled stack to {v_index: [MaskObjNode,...]}.
    """
    labels_3d = np.asarray(labels_3d)
    if labels_3d.ndim != 3:
        raise ValueError("labels_3d must be (nv, ny, nx)")
    nv = labels_3d.shape[0]
    result = {}
    for v in range(nv):
        result[v] = nodes_from_label_slice(labels_3d[v], v_index=v)
    return result


# Dict key helpers (nodes / trees)

def _inc_suffix(key):
    base, _, suf = key.rpartition("_")
    try:
        return f"{base}_{int(suf) + 1}"
    except ValueError:
        return key + "_1"


def add_node_to_dict(node, dictionary):
    """
    Insert node into dict with area-based key '<masked_area>_n', avoiding collisions.
    """
    key = f"{node.masked_area_size}_0"
    while key in dictionary:
        key = _inc_suffix(key)
    dictionary[key] = node
    return key


def add_tree_to_dict(tree, dictionary):
    """
    Insert tree with key '<masked_area>_<vstart>_n', avoiding collisions.
    Works with MaskObjNodeTree implemented in core/mask_tree.py.
    """
    try:
        area = int(tree.masked_area_2d())
        vstart = int(tree.starting_velocity())
    except AttributeError:
        # Fallback for older/alternate API names
        area = int(tree.getTreeMaskedArea2D())
        vstart = int(tree.getTreeStartingVelocity())

    key = f"{area}_{vstart}_0"
    while key in dictionary:
        key = _inc_suffix(key)
    dictionary[key] = tree
    return key


def pop_tree_from_dict(tree_key, dictionary):
    """
    Remove and return a tree by key; raises KeyError if missing.
    """
    if tree_key in dictionary:
        log.debug("Removing tree %s from dictionary", tree_key)
        return dictionary.pop(tree_key)
    raise KeyError(f"{tree_key!r} not found")


def sorted_keys_by_area(dict_keys, key_type, descending=True):
    """
    Sort hashed keys by encoded masked area.

    key_type:
      - 'node': keys look like '<area>_<n>'
      - 'tree': keys look like '<area>_<vstart>_<n>'
    """
    if key_type.lower() == "node":
        mapped = []
        for k in dict_keys:
            try:
                area = int(k.split("_")[0])
            except Exception:
                area = -1
            mapped.append((k, area))
    elif key_type.lower() == "tree":
        mapped = []
        for k in dict_keys:
            try:
                area = int(k.split("_")[0])
            except Exception:
                area = -1
            mapped.append((k, area))
    else:
        return list(dict_keys)

    mapped.sort(key=lambda t: t[1], reverse=bool(descending))
    return [k for (k, _a) in mapped]



# Point queries

def is_point_on_node(node, point=(0, 0), strict=False):
    """
    Return True if (y, x) lies within node’s corners.
    """
    y, x = int(point[0]), int(point[1])
    y0, x0 = node.corner_min
    y1, x1 = node.corner_max
    inside = (y0 <= y < y1) and (x0 <= x < x1)
    if not inside:
        return False
    if not strict:
        return True
    return bool(node.mask[y - y0, x - x0])


def filter_nodes_by_point(nodes_dicts, point=(0, 0), strict=False):
    """
    Filter a dict-of-node-dicts, keeping only nodes that contain the point.
    """
    yx = (int(point[0]), int(point[1]))
    out = {}
    for group_key, nodes in nodes_dicts.items():
        out[group_key] = {
            k: n for (k, n) in nodes.items()
            if is_point_on_node(n, yx, strict=strict)
        }
    return out


def filter_trees_by_point(trees_dict, point=(0, 0), strict=False):
    """
    Filter a tree dict, keeping trees whose *root* mask contains the point.
    """
    yx = (int(point[0]), int(point[1]))
    out = {}
    for k, t in trees_dict.items():
        if is_point_on_node(t.root_node, yx, strict=strict):
            out[k] = t
    return out
