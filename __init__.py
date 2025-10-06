"""
fil3d
=====

A Python package for detecting and analyzing coherent 3D filamentary structures in 3D data cubes.

Subpackages
------------
- fil3d.core : Core data structures and algorithms for mask and tree processing.
- fil3d.cli  : Command-line interface for batch processing and automation.
- fil3d.io   : Optional I/O utilities for reading and writing 3D data.

Main Classes
------------
- MaskObjNode
- MaskObjNodeTree

Example
-------
>>> from fil3d import MaskObjNode, MaskObjNodeTree
>>> import numpy as np
>>> mask = np.random.rand(10, 10) > 0.8
>>> node = MaskObjNode(mask, [[0, 0], [10, 10]], v_slice_index=5)
>>> tree = MaskObjNodeTree(node)
>>> tree.add_on_new_channel(node)

fil3d is designed to be model-agnostic and survey-independent, supporting any 3D dataset 
(e.g., HI, CO, any emission data cubes) where spatial coherence across slices is relevant.
"""

from __future__ import annotations
import logging

# Public re-exports
from .core import (
    MaskObjNode,
    MaskObjNodeTree,
    get_logger,
    ensure_bool_mask,
    bbox_from_mask,
    nodes_from_label_slice,
    slices_from_labeled_stack,
    add_node_to_dict,
    add_tree_to_dict,
    filter_trees_by_point,
)

__version__ = "0.1.0"

__all__ = [
    "MaskObjNode",
    "MaskObjNodeTree",
    "get_logger",
    "ensure_bool_mask",
    "bbox_from_mask",
    "nodes_from_label_slice",
    "slices_from_labeled_stack",
    "add_node_to_dict",
    "add_tree_to_dict",
    "filter_trees_by_point",
    "__version__",
]


# Package-wide logging defaults

_log = logging.getLogger("fil3d")
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)



# Lazy subpackage import

def __getattr__(name):
    """
    Quick access to optional subpackages (e.g., `cli`, `io`).
    """
    if name == "cli":
        from . import cli
        return cli
    if name == "io":
        from . import io
        return io
    raise AttributeError(f"module 'fil3d' has no attribute '{name}'")
