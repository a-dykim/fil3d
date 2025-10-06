"""
fil3d.core
==========

Core data structures and utilities for 3D filament detection and analysis.

This subpackage provides:
    - MaskObjNode:  represents a 2D mask with its geometric metadata.
    - MaskObjNodeTree: hierarchical structure connecting masks across velocity slices.
    - Utility functions for mask manipulation, node construction, and dictionary management.
"""

from __future__ import annotations
import logging

# Public API
from .mask_node import MaskObjNode
from .mask_tree import MaskObjNodeTree
from .util import (
    get_logger,
    ensure_bool_mask,
    bbox_from_mask,
    nodes_from_label_slice,
    slices_from_labeled_stack,
    add_node_to_dict,
    add_tree_to_dict,
    filter_trees_by_point,
)

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
]


# Initialize default logger once
log = logging.getLogger("fil3d.core")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
