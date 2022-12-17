import itertools
from collections import defaultdict

import numpy as np
import torch

ATOMIC_PAIRS = [x for x in itertools.combinations_with_replacement([1, 6, 7, 8], 2)]


def torch_cdist(x, y=None):
    x = torch.from_numpy(x)
    if y is None:
        return torch.cdist(x, x)
    else:
        y = torch.from_numpy(y)
        return torch.cdist(x, y)


def distances_from_coordinates(coordinates, atomic_numbers):
    def group_distances(distance_matrix, atomic_numbers):
        distances = {pair: [] for pair in ATOMIC_PAIRS}
        for i, j in zip(*np.triu_indices_from(distance_matrix, k=1)):
            atom_atom_distance = distance_matrix[i, j]
            atomic_number_pair = tuple(sorted((atomic_numbers[i], atomic_numbers[j])))
            distances[atomic_number_pair].append(atom_atom_distance)
        return distances

    if coordinates.ndim == 2:
        coordinates = np.expand_dims(coordinates, axis=0)

    pairwise_distances = torch_cdist(coordinates).numpy()

    if isinstance(atomic_numbers, np.ndarray) and len(atomic_numbers.shape) == 1:
        atomic_numbers = np.array([atomic_numbers] * len(coordinates))
    elif isinstance(atomic_numbers, list) and not isinstance(atomic_numbers[0], list):
        atomic_numbers = [atomic_numbers]

    all_distances = []
    for pairwise_distance, _atomic_numbers in zip(pairwise_distances, atomic_numbers):
        all_distances.append(group_distances(pairwise_distance, _atomic_numbers))

    if len(all_distances) == 1:
        return all_distances[0]
    else:
        return all_distances


def pdist_mol(coordinates, atomic_numbers):
    zs = itertools.combinations_with_replacement(atomic_numbers, 2)
    zs = list(tuple(sorted(z)) for z in zs)
    if coordinates.ndim == 2:
        ds = torch.nn.functional.pdist(torch.as_tensor(coordinates)).numpy()
    elif coordinates.ndim == 3:
        ds = np.vstack(
            [torch.nn.functional.pdist(co).numpy() for co in torch.as_tensor(coordinates)]
        )
    else:
        raise ValueError()

    res = defaultdict(list)
    for z, d in zip(zs, ds if ds.ndim == 1 else ds.T):
        z = tuple(sorted(z))
        res[z].append(d)

    res = {z: np.asarray(res[z]) if ds.ndim == 1 else np.concatenate(res[z]) for z in zs}

    return res

