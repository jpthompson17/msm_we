"""This module contains functions for decomposing a weighted ensemble
trajectory into trajectory segments of a specified length.

"""
import networkx as nx
import scipy.sparse as sparse

from westpa.core.data_manager import WESTDataManager
from westpa.core.h5io import WESTPAH5File
from typing import Union


def get_history_graph(we_h5file: Union[WESTPAH5File, str]) -> nx.DiGraph:
    """Return the directed graph whose edges connect trajectory segments
    to their immediate parents.

    Parameters
    ----------
    we_h5file : WESTPAH5File | str
        The WESTPA data file to read.

    Returns
    -------
    nx.DiGraph
        The history graph of the weighted ensemble trajectory stored in
        `we_h5file`.

    """
    data_manager = WESTDataManager()
    if isinstance(we_h5file, str):
        data_manager.we_h5filename = we_h5file
    else:
        data_manager.we_h5file = we_h5file
    data_manager.open_backing()

    history_graph = nx.DiGraph()
    parent_segments = []
    for n_iter in range(1, data_manager.current_iteration):
        segments = data_manager.get_segments(n_iter)
        initial_states = data_manager.get_initial_states(n_iter)
        initial_states = {state.state_id: state for state in initial_states}
        for segment in segments:
            if segment.parent_id >= 0:  # parent is a Segment
                parent = parent_segments[segment.parent_id]
            else:  # parent is an InitialState
                state_id = -(segment.parent_id + 1)
                parent = initial_states[state_id]
            history_graph.add_edge(segment, parent)
        parent_segments = segments

    data_manager.close_backing()

    return history_graph


def child_to_parent_mapping(history_graph, degree=1):
    adjacency_matrix = nx.adjacency_matrix(history_graph) ** degree
    nodes = list(history_graph.nodes)
    mapping = {}
    for i, j, _ in zip(*sparse.find(adjacency_matrix)):
        mapping[nodes[i]] = nodes[j]
    return mapping
