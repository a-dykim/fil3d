import logging
import copy
import numpy as np

log = logging.getLogger(__name__)


class MaskObjNodeTree:
    """
    A sequence of `MaskObjNode` objects linked across velocity channels.

    The tree maintains:
      - a deep-copied `root_node` that accumulates the overall OR-mask “shadow”
      - an ordered `node_list` of per-channel nodes
      - velocity bookkeeping (`root_v_slice`, `length`)
    """

    def __init__(self, node_obj):
        """
        Parameters
        ----------
        node_obj : MaskObjNode
            Starting node of the tree.
        """
        self.root_node = copy.deepcopy(node_obj)
        self.root_v_slice = int(node_obj.v_slice_index[0])

        self.node_list = [copy.deepcopy(node_obj)]
        self.length = 1
        self.has_ended = False


    # Mutators

    def add_on_new_channel(self, new_node):
        """
        Add a node assumed to be on the *next* velocity channel.

        Steps:
          1) Merge into `root_node` (keeps the global shadow).
          2) Append a copy to `node_list`.
          3) Increment `length`.

        Returns
        -------
        New tree length.
        """
        log.debug("Adding node on new velocity channel; new_node corners=%r", new_node.corners)
        log.debug("Old root corners=%r", self.root_node.corners)

        self.root_node.merge(new_node)
        self.node_list.append(copy.deepcopy(new_node))
        self.length += 1

        log.debug("New root corners=%r", self.root_node.corners)
        return self.length

    def add_on_same_channel(self, new_node):
        """
        Add a node that belongs to the *same* velocity channel as the last node.

        The last node in `node_list` represents the aggregate of all nodes at that channel.
        We therefore:
          1) Merge into `root_node` (global shadow),
          2) Merge into the last node (per-channel aggregate remains one element).

        Returns
        -------
        Tree length.
        """
        log.debug("Adding node on same velocity channel; new_node corners=%r", new_node.corners)
        log.debug("Old root corners=%r", self.root_node.corners)

        self.root_node.merge(new_node)
        self.last_node().merge(new_node)

        log.debug("New root corners=%r", self.root_node.corners)
        return self.length

    def remove_last_node(self):
        """
        Remove the last node and update the accumulated root accordingly.

        Notes
        -----
        - This is an *approximate* revert: since `root_node` has been merged across all
          nodes by OR, removing the last node cannot un-OR its contribution without
          recomputing from scratch. So we recompute `root_node` by re-merging from the
          first node.
        """
        if self.length == 0:
            raise RuntimeError("Cannot remove from an empty tree")
        if self.length == 1:
            # Drop to empty “shell” with no nodes
            self.node_list.pop()
            self.length = 0
            self.has_ended = True
            # root_node no longer meaningful; keep as-is to avoid None checks
            return 0

        # Remove last and recompute root from scratch
        self.node_list.pop()
        self.length -= 1

        base = copy.deepcopy(self.node_list[0])
        for k in range(1, self.length):
            base.merge(self.node_list[k])
        self.root_node = base
        return self.length

    def visit_all_nodes(self):
        """
        Mark all nodes (including root) as visited.
        """
        self.root_node.visited = True
        for n in self.node_list:
            n.visited = True
        return True


    # Accessors

    def node(self, idx):
        """
        Return node at index `idx`.
        """
        return self.node_list[idx]

    def last_node(self):
        """
        Return the last node.
        """
        return self.node_list[self.length - 1]

    def mask(self):
        """
        Return the overall OR-mask (“shadow”) of the tree (2D np.array[bool]).
        """
        return self.root_node.mask

    def mask_size_2d(self):
        """
        Rectangle area (from corners) of the tree’s accumulated mask.
        """
        return self.root_node.mask_size

    def masked_area_2d(self):
        """
        Count of True pixels in the tree’s accumulated mask.
        """
        return self.root_node.masked_area_size

    def velocity_range(self):
        """
        Return 1D np.arange covering all velocity channels spanned by the tree.
        """
        return np.arange(self.root_v_slice, self.root_v_slice + self.length)

    def starting_velocity(self):
        """
        Return the starting velocity channel index.
        """
        return self.root_v_slice

    def aspect_ratio(self):
        """
        Return the rectangle aspect ratio (≥ 1) of the accumulated mask.
        """
        return self.root_node.aspect_ratio()



# Helpers

def new_tree_from_node(node, mark_as_visited=True):
    """
    Construct a new tree from a single node.
    """
    log.info("Constructing new tree from node")
    if mark_as_visited:
        node.visited = True
    return MaskObjNodeTree(node)


def back_merge_trees(base_tree, other_tree):
    """
    Merge two trees *backwards* aligned on their most recent velocity channel.
    The resulting tree contains, at each step, the union of nodes present in the aligned slices.
    """
    # Ensure base_tree is the longer or equal-length one
    if base_tree.length < other_tree.length:
        base_tree, other_tree = other_tree, base_tree

    v_diff = base_tree.length - other_tree.length
    merged = new_tree_from_node(copy.deepcopy(base_tree.node(0)), mark_as_visited=False)

    for i in range(base_tree.length):
        if i != 0:
            merged.add_on_new_channel(copy.deepcopy(base_tree.node(i)))

        j = i - v_diff
        if j >= 0:
            merged.add_on_same_channel(copy.deepcopy(other_tree.node(j)))

    return merged
