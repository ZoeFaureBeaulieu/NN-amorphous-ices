import numpy as np
from typing import List, Tuple, Dict
from ase import Atoms
from itertools import product
from scipy.special import sph_harm
from ase.neighborlist import NeighborList
from ase import Atoms
from ase.io import write
from pathlib import Path
import wigners

root_dir = Path(__file__).resolve().parent.parent.parent


def get_n_nearest_neighbours(
    structure: Atoms, cutoff_radius: float, numb_neighbours: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the distance vectors to the n nearest neighbours of each atom in a structure.
    Also returns the indices of said neighbours.

    Parameters
    ----------
    structure : ase.Atoms
        The structure to get the neighbours for.
    cutoff_radius : float
        Cut-off radius to use when generating the neighbour list.
    numb_neighbours : int
        Number of nearest neighbours to get.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the distance vectors and indices of the neighbours.
    """
    # to handle cases where an atom has a neighbour that is
    # outside the (0, 0, 0) unit cell (i.e. in a neighbouring unit cell)
    # get the distance vectors
    # see https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.PrimitiveNeighborList.get_neighbors

    nl = NeighborList(
        cutoffs=[cutoff_radius] * len(structure),
        self_interaction=False,
        bothways=True,
    )
    nl.update(structure)

    neighbour_indices = np.zeros((len(structure), numb_neighbours), dtype=int)
    neighbour_vectors = np.zeros((len(structure), numb_neighbours, 3))

    for atom_index in range(len(structure)):
        indices, offsets = nl.get_neighbors(atom_index)
        neighbour_locations = []

        # get the neighbour locations
        # and the offsets (i.e which repeat of the unit cell the neighbour is in)
        for i, offset in zip(indices, offsets):
            neighbour_locations.append(
                structure[i].position + offset @ structure.get_cell()
            )

        dist_vectors = np.array(neighbour_locations) - structure[atom_index].position

        assert (
            0 < np.linalg.norm(dist_vectors, axis=-1)
        ).all(), "One of the distances is 0! Check the neighbour list."

        dist_to_neighbour = np.linalg.norm(dist_vectors, axis=1)
        order = np.argsort(dist_to_neighbour)

        ordered_distances = dist_to_neighbour[order]

        # sort the vectors and indices by distance
        # from smallest to largest
        sorted_vectors = dist_vectors[order]
        sorted_indices = indices[order]

        assert sorted_indices.max() < len(structure)
        assert (
            len(sorted_vectors) >= numb_neighbours
        ), "Not enough neighbours! Try increasing the cutoff radius."

        neighbour_vectors[atom_index] = sorted_vectors[:numb_neighbours]
        neighbour_indices[atom_index] = sorted_indices[:numb_neighbours]

    return neighbour_vectors, neighbour_indices


def find_wigner_combinations(m_values):
    """Generates all the combinations of m_values which sum to 0.

    Parameters
    ----------
    m_values : List
        The m values to find combinations for

    Returns
    -------
    _type_
        _description_
    """
    combinations_list = []
    for combination in product(m_values, repeat=3):
        if sum(combination) == 0:
            combinations_list.append(combination)
    return combinations_list


def get_indices_for_tuple(tuple_values, mapping):
    return tuple(mapping[m] for m in tuple_values)


def spherical_harmonics(vectors: np.ndarray, l: int) -> np.ndarray:
    """Convert 3D cartesian coordinates into spherical harmonics.

    Parameters
    ----------
    vectors : np.ndarray
        The vectors to calculate the spherical harmonics for. Must have shape (..., 3).
        This can be a single vector per atom (numb_neigh_per_atom,3)
        or an array of vectors per structure (numb_atoms,numb_neigh_per_atom,3).
    l : int
        The l value to use.

    Returns
    -------
    np.ndarray
        The spherical harmonics for each vector in vectors.
        This will have shape (..., 2l+1).
    """

    # Check the inputs
    assert vectors.shape[-1] == 3, "vectors must have shape (..., 3)"
    assert len(vectors.shape) > 1, "vectors must have at least 2 dimensions"
    assert l >= 0, "l must be non-negative"

    # Convert the last dimension of vectors (i.e cartesian coords)
    # to spherical coordinates - i.e. angles theta and phi
    theta = np.arctan2(vectors[..., 1], vectors[..., 0])
    phi = np.arccos(vectors[..., 2] / np.linalg.norm(vectors, axis=-1))

    # get the m values for a given l
    # there will be 2l+1 m values
    m_values = np.arange(-l, l + 1)
    sph_harmonics = np.array([sph_harm(m, l, theta, phi) for m in m_values])

    # reshape from (2l+1, ...) to (..., 2l+1)
    return np.moveaxis(sph_harmonics, 0, -1)


def make_invariant(q_lm: np.ndarray, l: int) -> np.ndarray:
    """Calculate the invariant q_l from the q_lm values.
    This is done by summing over the 2l+1 m values.

    Parameters
    ----------
    q_lm : np.ndarray
        Array of shape (len(struct), 2l+1) containing the q_lm values for each atom in the structure.
    l : int
        The l value to use.

    Returns
    -------
    np.ndarray
        Array of shape (len(struct),) containing the invariant q_l values for each atom in the structure.
    """
    assert len(q_lm.shape) == 2
    assert q_lm.shape[-1] == 2 * l + 1

    q_lm_squared = (q_lm * np.conj(q_lm)).real
    return np.sqrt(4 * np.pi / (2 * l + 1)) * np.sqrt(q_lm_squared.sum(axis=1))


def get_q_lm(neighbour_vectors, neighbour_indices, l, numb_neighbours):
    len_struct = len(neighbour_indices)

    sph_harmonics = spherical_harmonics(neighbour_vectors, l)

    assert sph_harmonics.shape == (
        len_struct,
        numb_neighbours,
        2 * l + 1,
    )

    # sum over the neighbours to get the q_lm for each atom
    q_lm = (1 / numb_neighbours) * np.sum(sph_harmonics, axis=1)

    assert q_lm.shape == (len_struct, 2 * l + 1)

    av_q_lm = (q_lm[neighbour_indices].sum(axis=1) + q_lm) / (numb_neighbours + 1)

    assert av_q_lm.shape == (len_struct, 2 * l + 1)

    return q_lm, av_q_lm


def get_steinhardt_q_l_parameters(
    l_values,
    numb_neighbours,
    neighbour_vectors,
    neighbour_indices,
):
    len_struct = len(neighbour_vectors)

    all_q_l = np.zeros((len_struct, len(l_values)))
    all_av_q_l = np.zeros((len_struct, len(l_values)))

    for q_l_index, l in enumerate(l_values):
        q_lm, av_q_lm = get_q_lm(
            neighbour_vectors, neighbour_indices, l, numb_neighbours
        )

        assert q_lm.shape == (len_struct, 2 * l + 1)
        assert av_q_lm.shape == (len_struct, 2 * l + 1)

        q_l = make_invariant(q_lm, l)
        av_q_l = make_invariant(av_q_lm, l)

        assert q_l.shape == (len_struct,)
        assert av_q_l.shape == (len_struct,)

        all_q_l[:, q_l_index] = q_l
        all_av_q_l[:, q_l_index] = av_q_l

    assert all_q_l.shape == (len_struct, len(l_values))
    assert all_av_q_l.shape == (len_struct, len(l_values))

    return all_q_l, all_av_q_l


def get_steinhardt_w_l_parameters(
    l_values, numb_neighbours, neighbour_vectors, neighbour_indices
):
    len_struct = len(neighbour_vectors)

    all_w_l = np.zeros((len_struct, len(l_values)))
    all_av_w_l = np.zeros((len_struct, len(l_values)))

    for w_l_index, l in enumerate(l_values):
        q_lm, av_q_lm = get_q_lm(
            neighbour_vectors, neighbour_indices, l, numb_neighbours
        )

        m_values = np.arange(-l, l + 1)
        m_to_index_mapping = {m: i for i, m in enumerate(m_values)}
        wigner_m_combs = find_wigner_combinations(m_values)
        wigner_m_indices = [
            get_indices_for_tuple(combo, m_to_index_mapping) for combo in wigner_m_combs
        ]

        w_l_numerator = np.zeros(len_struct)
        w_l_denominator = np.zeros(len_struct)
        av_w_l_numerator = np.zeros(len_struct)
        av_w_l_denominator = np.zeros(len_struct)

        for m_triples, index_triples in zip(wigner_m_combs, wigner_m_indices):
            m1, m2, m3 = m_triples
            i1, i2, i3 = index_triples
            wigner_coeff = wigners.wigner_3j(l, l, l, m1, m2, m3)

            w_l_product = q_lm[:, i1] * q_lm[:, i2] * q_lm[:, i3] * wigner_coeff
            av_w_l_product = (
                av_q_lm[:, i1] * av_q_lm[:, i2] * av_q_lm[:, i3] * wigner_coeff
            )

            w_l_numerator += w_l_product.real
            av_w_l_numerator += av_w_l_product.real

        w_l_denominator = ((q_lm * np.conj(q_lm)).real.sum(axis=1)) ** (3 / 2)
        av_w_l_denominator = ((av_q_lm * np.conj(av_q_lm)).real.sum(axis=1)) ** (3 / 2)

        w_l = w_l_numerator / w_l_denominator
        av_w_l = av_w_l_numerator / av_w_l_denominator

        assert w_l.shape == (len_struct,)
        assert av_w_l.shape == (len_struct,)

        all_w_l[:, w_l_index] = w_l
        all_av_w_l[:, w_l_index] = av_w_l

    assert all_w_l.shape == (len_struct, len(l_values))
    assert all_av_w_l.shape == (len_struct, len(l_values))

    return all_w_l, all_av_w_l


def get_steinhardt_params(
    struct: Atoms,
    cutoff_radius: float,
    numb_neighbours: int,
    q_l_values: List[int] = None,
    w_l_values: List[int] = None,
):
    if q_l_values is None and w_l_values is None:
        raise ValueError("At least one of q_l_values or w_l_values must be specified")

    neighbour_vectors, neighbour_indices = get_n_nearest_neighbours(
        struct, cutoff_radius, numb_neighbours
    )

    if q_l_values is not None:
        all_q_l, all_av_q_l = get_steinhardt_q_l_parameters(
            q_l_values, numb_neighbours, neighbour_vectors, neighbour_indices
        )
        q_descriptor = np.hstack((all_q_l, all_av_q_l))
    else:
        q_descriptor = np.zeros((len(struct), 0))

    if w_l_values is not None:
        all_w_l, all_av_w_l = get_steinhardt_w_l_parameters(
            w_l_values, numb_neighbours, neighbour_vectors, neighbour_indices
        )
        w_descriptor = np.hstack((all_w_l, all_av_w_l))

    else:
        w_descriptor = np.zeros((len(struct), 0))

    complete_descriptor = np.hstack((q_descriptor, w_descriptor))

    return complete_descriptor


def get_desc_idx_and_names(descriptors) -> Tuple[List[int], List[str]]:
    """Get the indices and names of the descriptors. Usually used for plotting.
    Can take either a list of indices or a list of names as strings.

    Parameters
    ----------
    descriptors : List[int] or List[str]
        The descriptors to get the indices and names for.

    Returns
    -------
    Tuple[List[int], List[str]]
        The indices and names of the descriptors.
    """
    # check whether the list of descriptors is a list of indices or strings
    if isinstance(descriptors[0], str):
        # if it's a list of strings, convert to a list of indices
        # index corresponds to position of the descriptor in the 30-dimensional array
        desc_idx = [desc_to_index_mapping[desc] for desc in descriptors]
        feature_names = descriptors
    elif isinstance(descriptors[0], int):
        # if it's a list of indices, convert to a list of strings
        # the strings give you the name of the descriptor
        # useful for plotting to label the axes/colour code
        desc_idx = descriptors
        idx_to_desc_mapping = {
            value: key for key, value in desc_to_index_mapping.items()
        }
        feature_names = [idx_to_desc_mapping[idx] for idx in desc_idx]

    return desc_idx, feature_names


desc_to_index_mapping = {
    "q3": 0,
    "q4": 1,
    "q5": 2,
    "q6": 3,
    "q7": 4,
    "q8": 5,
    "q9": 6,
    "q10": 7,
    "q11": 8,
    "q12": 9,
    "av_q3": 10,
    "av_q4": 11,
    "av_q5": 12,
    "av_q6": 13,
    "av_q7": 14,
    "av_q8": 15,
    "av_q9": 16,
    "av_q10": 17,
    "av_q11": 18,
    "av_q12": 19,
    "w4": 20,
    "w6": 21,
    "w8": 22,
    "w10": 23,
    "w12": 24,
    "av_w4": 25,
    "av_w6": 26,
    "av_w8": 27,
    "av_w10": 28,
    "av_w12": 29,
}

desc_to_plot_label_mapping = {
    "q3": r"$q_3$",
    "q4": r"$q_4$",
    "q5": r"$q_5$",
    "q6": r"$q_6$",
    "q7": r"$q_7$",
    "q8": r"$q_8$",
    "q9": r"$q_9$",
    "q10": r"$q_{10}$",
    "q11": r"$q_{11}$",
    "q12": r"$q_{12}$",
    "av_q3": r"$\bar{q}_3$",
    "av_q4": r"$\bar{q}_4$",
    "av_q5": r"$\bar{q}_5$",
    "av_q6": r"$\bar{q}_6$",
    "av_q7": r"$\bar{q}_7$",
    "av_q8": r"$\bar{q}_8$",
    "av_q9": r"$\bar{q}_9$",
    "av_q10": r"$\bar{q}_{10}$",
    "av_q11": r"$\bar{q}_{11}$",
    "av_q12": r"$\bar{q}_{12}$",
    "w4": r"$w_4$",
    "w6": r"$w_6$",
    "w8": r"$w_8$",
    "w10": r"$w_{10}$",
    "w12": r"$w_{12}$",
    "av_w4": r"$\bar{w}_4$",
    "av_w6": r"$\bar{w}_6$",
    "av_w8": r"$\bar{w}_8$",
    "av_w10": r"$\bar{w}_{10}$",
    "av_w12": r"$\bar{w}_{12}$",
}
