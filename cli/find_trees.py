from __future__ import annotations
import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
from astropy.io import fits

from fil3d.core.mask_node import MaskObjNode
from fil3d.core.util import add_node_to_dict
from fil3d.core.tree import find_all_trees_from_slices

log = logging.getLogger(__name__)


# Filament extraction per slice

def construct_filaments(vslice, header, v_channel):
    """
    Build MaskObjNode objects for a single velocity slice using FilFinder2D.

    Parameters
    ----------
    vslice : np.ndarray (2D)
        Intensity image for one velocity channel.
    header : dict-like
        FITS header or mapping with needed WCS/beam keys for FilFinder2D.
    v_channel : int
        Velocity index for this slice.

    Returns
    -------
    dict
        { key -> MaskObjNode } (keys are area-based hashed strings).
    """
    try:
        from fil_finder import FilFinder2D
        import astropy.units as u
    except Exception as e:
        raise RuntimeError(
            "FilFinder2D (fil_finder) and astropy.units are required for construct_filaments()."
        ) from e

    # FilFinder expects units on header; ensure a sane brightness unit
    hdr = dict(header) if header is not None else {}
    hdr["BUNIT"] = hdr.get("BUNIT", "K")

    # Empirical default scale values (adjustable by callers if needed)
    distance = 100.0 * u.pc
    beamwidth = 10.0 * u.arcmin
    scale_width = 0.1 * u.pc

    ff = FilFinder2D(vslice, header=hdr, distance=distance, beamwidth=beamwidth)
    ff.preprocess_image(flatten_percent=95)

    # Ask FilFinder for mask objects and corners in image pixel coordinates
    # corners: [[y0,x0],[y1,x1]] (half-open)
    masks, corners = ff.create_mask(
        smooth_size=scale_width / 2.0,
        adapt_thresh=scale_width * 2.0,
        size_thresh=8.0 * (float(scale_width.to_value(u.pc)) * 2.0) ** 2,
        border_masking=False,
        output_mask_objs=True,
    )

    node_dict = {}
    if masks is None or len(masks) == 0:
        return node_dict

    for m, c in zip(masks, corners):
        node = MaskObjNode(m, c, v_slice_index=int(v_channel))
        add_node_to_dict(node, node_dict)
    return node_dict


def noderun_for_multichannel(valid_slices, header, vchannels, save_path=None):
    """
    Run FilFinder-based node construction for multiple velocity slices.

    Parameters
    ----------
    valid_slices : list/array of 2D arrays
        A sequence of 2D images, one per velocity channel.
    header : dict-like
        FITS header (passed to FilFinder2D).
    vchannels : list/array of int
        Velocity channel indices corresponding to valid_slices.
    save_path : str or None
        If provided, pickle the per-slice nodes mapping to this path.

    Returns
    -------
    dict
        { v_index -> { node_key -> MaskObjNode } }
    """
    all_nodes = {}
    for img, v in zip(valid_slices, vchannels):
        nodes = construct_filaments(img, header, int(v))
        all_nodes[int(v)] = nodes
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(all_nodes, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved nodes to %s", save_path)
    return all_nodes



# Linking into 3D structures

def _to_slices_map(nodes_by_v):
    """
    Convert {v: {key: MaskObjNode}} -> {v: [MaskObjNode, ...]} for the linker.
    """
    out = {}
    for v, d in nodes_by_v.items():
        out[int(v)] = list(d.values())
    return out


def find_trees(nodes_by_v, overlap_thresh=0.85, reverse_find=False):
    """
    Find coherent 3D structures (trees) from per-slice nodes.

    Parameters
    ----------
    nodes_by_v : dict
        { v_index -> { key -> MaskObjNode } }
    overlap_thresh : float
        Minimum overlap score to retain a link. With the new linker, the most
        similar behavior to your original "either-mask" criterion is using the
        'min' metric: |A∩B| / min(|A|, |B|). Default 0.85.
    reverse_find : bool
        Unused (kept for backward API compatibility).

    Returns
    -------
    dict
        JSON-files from `find_all_trees_from_slices`:
        {
          "nodes": [...],
          "edges": [{"u": "v:i", "v": "vp:j", "w": score}, ...],
          "components": [{"nodes": [...], "stats": {...}}, ...]
        }
    """
    slices = _to_slices_map(nodes_by_v)
    # Matching strategy: mutual best by default; allow_skip=0 to mimic contiguous search
    result = find_all_trees_from_slices(
        slices=slices,
        overlap_thresh=float(overlap_thresh),
        metric="min",           # closer to your original "either-mask fraction" check
        allow_skip=0,
        matching="mutual",
        as_dict=True,
    )
    return result


def run_and_save_trees(nodes_by_v, save_path, overlap_thresh=0.85, reverse_find=False):
    """
    Runscript for finding trees and pickling results (backward-compatible name).

    Parameters
    ----------
    nodes_by_v : dict
        { v_index -> { key -> MaskObjNode } }
    save_path : str
        Output path (.p or .pkl recommended).
    overlap_thresh : float
        Overlap threshold passed to finder (default: 0.85).
    reverse_find : bool
        Unused (kept for compatibility).

    Returns
    -------
    dict
        The JSON-files that was saved.
    """
    result = find_trees(nodes_by_v, overlap_thresh=overlap_thresh, reverse_find=reverse_find)
    with open(save_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("%s TREE SAVED (thr=%.2f)", os.path.basename(save_path), overlap_thresh)
    return result



# Optional CLI entry point

def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Detect 2D filaments per slice (FilFinder2D) and link across "
                    "velocity channels into coherent 3D structures."
    )
    p.add_argument("--fits", "-i", required=True, help="Path to input FITS cube (nv, ny, nx).")
    p.add_argument("--save-nodes", help="Optional path to pickle the {v: {key: node}} map.")
    p.add_argument("--save-trees", "-o", required=True, help="Path to pickle the trees result dict.")
    p.add_argument("--v-start", type=int, default=0, help="Start velocity index (inclusive).")
    p.add_argument("--v-end", type=int, default=None, help="End velocity index (exclusive).")
    p.add_argument("--thr", type=float, default=0.85, help="Overlap threshold (default: 0.85).")
    p.add_argument("--log-level", default=os.getenv("FIL3D_LOGLEVEL", "INFO"),
                   help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")

    # Load FITS cube
    data, header = fits.getdata(args.fits, header=True)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D FITS cube (nv, ny, nx); got {data.shape}")

    nv = data.shape[0]
    v0 = max(0, int(args.v_start))
    v1 = int(args.v_end) if args.v_end is not None else nv
    if not (0 <= v0 < v1 <= nv):
        raise ValueError(f"Invalid velocity range: [{v0}, {v1}) within [0, {nv})")

    valid_slices = [data[i] for i in range(v0, v1)]
    vchannels = list(range(v0, v1))

    # Per-slice detection
    nodes_by_v = noderun_for_multichannel(valid_slices, header, vchannels, save_path=args.save_nodes)

    # Link & save trees
    run_and_save_trees(nodes_by_v, save_path=args.save_trees, overlap_thresh=args.thr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
