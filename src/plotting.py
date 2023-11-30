import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

root_dir = Path(__file__).resolve().parent.parent.parent

struct_type_colours = {
    "HDA": "#dddf00",
    "LDA": "#386641",
    "Liquid": "#80b918",
    "MDA": "#03045e",
}


def colour_gradient(things, c1, c2):
    cmap = LinearSegmentedColormap.from_list("mycmap", [c1, c2])
    gradient = cmap(np.linspace(0, 1, len(things)))
    return zip(things, gradient)


def prepare_shearing_data(mda_dataset: Dict, descriptor_idx: int) -> List[np.ndarray]:
    """Process the MDA data for plotting.
    Split the data into 10 individual structures for the initial shears,
    9 groups of 10 individual structures for the final shears,
    and the ice Ih and final MDA structures.

    Parameters
    ----------
    mda_dataset : Dict
        The MDA dataset.
        As prepared by the `steinhardt.data.process_mda_structures` function.
    descriptor_idx : int
        The index of the descriptor to plot.
        The index corresponds to the position of the descriptor in the 30-dimensional descriptor vector.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays containing the data to plo:
            - Element 0: is the ice Ih structure (2880 atoms),
            - Elements 1-10: are the initial shears (2880 atoms each).
            - Elemements 11-19: are the final shears. Each element contains 10 individual structures (i.e. 28880 atoms)
            - Element 20: is the final MDA structure. Contains either 2880 atoms (a single run) or 14400 atoms (all 5 runs).


    """
    data = []
    # get the ice Ih structure
    data.append(mda_dataset["ice_Ih"][:, descriptor_idx])

    # separate the initial shears into 10 individual structures
    for i in range(10):
        data.append(
            mda_dataset["initial_shears"][2880 * i : 2880 * (i + 1), descriptor_idx]
        )

    # separate the final shears into 9 groups of 10 individual structures
    for i in range(9):
        data.append(
            mda_dataset["final_shears"][28800 * i : 28800 * (i + 1), descriptor_idx]
        )

    # get the final mda structure
    data.append(mda_dataset["mda"][:, descriptor_idx])

    return data
