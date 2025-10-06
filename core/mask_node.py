from __future__ import annotations
import logging
import numpy as np
from fil3d import _const

log = logging.getLogger(__name__)


class MaskObjNode:
    """
    Container for a single 2D mask and its corners on one or more velocity slices.

    Notes
    -----
    - Corners follow NumPy indexing with (0, 0) at top-left:
      [[y0, x0], [y1, x1]] where the max corner is half-open (exclusive).
    - "mask size" (rectangle area) is derived from corners; "masked area size"
      counts True pixels inside the mask.
    """

    def __init__(self, mask_obj, corners, v_slice_index):
        """
        Parameters
        ----------
        mask_obj : np.ndarray
            2D bit/boolean mask.
        corners : list of list
            [[y0, x0], [y1, x1]] (half-open).
        v_slice_index : int or list of int
            Velocity-channel index or indices.
        """
        if mask_obj.ndim != 2:
            raise ValueError("mask_obj must be a 2D array")
        self.mask = mask_obj.astype(bool, copy=False)

        corners = self._normalize_corners(corners)
        corners = self._fix_border_corners(self.mask, corners)

        self.corners_original = corners
        self.corners = corners
        self.corner_min, self.corner_max = corners

        if not all(self.corner_max[i] > self.corner_min[i] for i in range(2)):
            raise ValueError(f"Invalid corners: {corners}")

        self.v_slice_index = (
            [int(v) for v in v_slice_index]
            if isinstance(v_slice_index, (list, tuple))
            else [int(v_slice_index)]
        )

        self.visited = False
        self.mask_size = self._check_area_size()
        self.masked_area_size = self._check_masked_area_size()

        expected = self.mask.shape[0] * self.mask.shape[1]
        if self.mask_size != expected:
            raise ValueError(
                f"Mask size/corner mismatch: {self.mask_size} != {expected}, "
                f"mask.shape={self.mask.shape}, corners={self.corners_original}"
            )

    def __eq__(self, other):
        return (
            isinstance(other, MaskObjNode)
            and np.array_equal(self.mask, other.mask)
            and self.corners_original == other.corners_original
            and self.v_slice_index == other.v_slice_index
            and self.visited == other.visited
        )

    def __repr__(self):
        return (
            f"MaskObjNode(v={self.v_slice_index}, corners={self.corners_original}, "
            f"area={self.mask_size}, masked={self.masked_area_size})"
        )


    def merge(self, other_node):
        """
        Merge ``other_node`` into this one via OR operation and corner union.
        """
        if self.v_slice_index[-1] != other_node.v_slice_index[0]:
            self.v_slice_index.append(other_node.v_slice_index[0])

        if not self.overlaps_with(other_node):
            log.warning("Corners don't overlap: self=%r, other=%r", self.corners, other_node.corners)

        new_corners = self._match_corners(other_node)
        expanded_self = self._expand_mask(new_corners)
        expanded_other = other_node._expand_mask(new_corners)

        combined = np.bitwise_or(expanded_self, expanded_other)
        self.mask = combined
        self.corners = self.corners_original = new_corners
        self.corner_min, self.corner_max = new_corners
        self.mask_size = self._check_area_size(new_corners)
        self.masked_area_size = np.count_nonzero(combined)
        return True

    def overlaps_with(self, other_node, overlap_thresh=0.0):
        """
        Return True if overlap fraction ≥ `overlap_thresh`.
        """
        if not self.overlaps_with_corners(other_node):
            return False

        new_corners = self._match_corners(other_node)
        a = self._expand_mask(new_corners)
        b = other_node._expand_mask(new_corners)
        inter = np.count_nonzero(np.bitwise_and(a, b))

        return (
            inter / self.masked_area_size >= overlap_thresh
            or inter / other_node.masked_area_size >= overlap_thresh
        )

    def combine(self, other_node, merge_type='AND'):
        """
        Combine with another mask using 'AND' or 'OR' operation.
        """
        mt = merge_type.upper()
        if mt not in ('AND', 'OR'):
            raise ValueError(f"merge_type must be 'AND' or 'OR', got {merge_type!r}")

        new_corners = self._match_corners(other_node)
        a = self._expand_mask(new_corners)
        b = other_node._expand_mask(new_corners)

        return np.bitwise_and(a, b) if mt == 'AND' else np.bitwise_or(a, b)


    # Geometry and corner utilities

    def overlaps_with_corners(self, other_node):
        """
        Check whether bounding rectangles overlap.
        """
        y_overlap = not (other_node.corner_max[0] <= self.corner_min[0] or other_node.corner_min[0] >= self.corner_max[0])
        x_overlap = not (other_node.corner_max[1] <= self.corner_min[1] or other_node.corner_min[1] >= self.corner_max[1])
        return y_overlap and x_overlap

    def _match_corners(self, other_node):
        """
        Return minimal enclosing rectangle for self and other.
        """
        y0 = min(self.corner_min[0], other_node.corner_min[0])
        x0 = min(self.corner_min[1], other_node.corner_min[1])
        y1 = max(self.corner_max[0], other_node.corner_max[0])
        x1 = max(self.corner_max[1], other_node.corner_max[1])
        return [[y0, x0], [y1, x1]]

    def _expand_mask(self, new_corners):
        """
        Pad mask to match new_corners.
        """
        old = self.corners_original
        pad_y = (old[0][0] - new_corners[0][0], new_corners[1][0] - old[1][0])
        pad_x = (old[0][1] - new_corners[0][1], new_corners[1][1] - old[1][1])
        return np.pad(self.mask, (pad_y, pad_x), mode='constant', constant_values=False)

    def _check_area_size(self, corners=None):
        c = self.corners if corners is None else corners
        return (c[1][0] - c[0][0]) * (c[1][1] - c[0][1])

    def _check_masked_area_size(self, mask=None):
        m = self.mask if mask is None else mask
        return np.count_nonzero(m)

    def dimensions(self):
        """
        Return width, height.
        """
        height = self.corner_max[0] - self.corner_min[0]
        width = self.corner_max[1] - self.corner_min[1]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid dimensions from corners: {self.corners_original}")
        return width, height

    def aspect_ratio(self):
        """
        Return aspect ratio ≥ 1.
        """
        w, h = self.dimensions()
        ar = max(w, h) / min(w, h)
        return ar if np.isfinite(ar) else np.inf


    # Dict helpers

    @staticmethod
    def add_to_dict(node, dictionary):
        """
        Add node into dict using unique masked_area_size key.
        """
        key = f"{node.masked_area_size}_0"
        while key in dictionary:
            key = MaskObjNode._increment_key(key)
        dictionary[key] = node

    @staticmethod
    def _increment_key(key):
        """
        Increment key suffix for collision resolution.
        """
        base, _, suffix = key.rpartition('_')
        try:
            return f"{base}_{int(suffix) + 1}"
        except ValueError:
            return key + "_1"


    # Corner normalization

    @staticmethod
    def _normalize_corners(corners):
        if not (corners and len(corners) == 2):
            raise ValueError("corners must be [[y0,x0],[y1,x1]]")
        y0, x0 = corners[0]
        y1, x1 = corners[1]
        return [[min(y0, y1), min(x0, x1)], [max(y0, y1), max(x0, x1)]]

    @staticmethod
    def _fix_border_corners(mask, corners):
        """
        Adjust corners to match mask shape; retain GALFA edge conventions.
        """
        m_mask, n_mask = mask.shape
        c = [list(corners[0]), list(corners[1])]

        if (c[1][0] - c[0][0] != m_mask) or (c[1][1] - c[0][1] != n_mask):
            if c[0][0] == 0:
                c[0][0] = -1
            if c[0][1] == 0:
                c[0][1] = -1
            if c[1][0] == _const.NAXIS_Y - 1:
                c[1][0] = _const.NAXIS_Y
            if c[1][1] == _const.NAXIS_X - 1:
                c[1][1] = _const.NAXIS_X
        return c
