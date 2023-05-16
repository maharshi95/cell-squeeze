# %%

import os
import numpy as np
from sklearn.cluster import SpectralBiclustering
from base import BiClusterMatrixPermuter
from scipy.sparse import load_npz
from matplotlib import pyplot as plt
from itertools import product
from bit_vector import IntVector
from rich import print as rprint
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, save_npz, load_npz
from utils import flatten_array_dict, unflatten_array_dict
from tqdm import tqdm
from rich import print as rprint

# %%
permuter = BiClusterMatrixPermuter(2, 2)

# %% Load matrix for experimentation
data_path = (
    "../prepared/1k_mouse_kidney_CNIK_3pv3_raw_feature_bc_matrix/5000_5000_0.npz"
)
np.random.seed(0)
mat = np.array(load_npz(data_path).todense())
# %%


def make_int_vector_goop(mat: np.ndarray, bitwidth: int = 32):
    shape = mat.shape
    int_vec = IntVector(mat.size, bitwidth)
    for i, elem in enumerate(mat.reshape(-1)):
        int_vec[i] = elem
    return {
        "data": int_vec,
        "shape": shape,
    }


def make_array_from_int_vector_goop(goop: dict[str, tuple[int] | IntVector]):
    """De-serializes an IntVector goop dict into a numpy array"""
    shape = goop["shape"]
    data = goop["data"]
    array = data.to_numpy()
    N = np.prod(shape)
    array = array[-N:]
    return array.reshape(shape)


def make_sparse_goop(mat: np.ndarray, method: str = "csr"):
    if method == "csr":
        sparse_mat = csr_matrix(mat)
    else:
        raise NotImplementedError
    return {
        "d": make_int_vector_goop(sparse_mat.data)["data"],
        "i1": make_int_vector_goop(sparse_mat.indices)["data"],
        "i2": make_int_vector_goop(sparse_mat.indptr)["data"],
        "n": sparse_mat.shape[0],
        "w": sparse_mat.nnz,
    }


# %%


def make_goop_dict(mat: np.ndarray):
    # mat shape: [N, M]
    non_zero_rows = np.where(mat.sum(axis=1) > 0)[0]
    non_zero_cols = np.where(mat.sum(axis=0) > 0)[0]
    non_zero_mat = mat[non_zero_rows, :][:, non_zero_cols]

    pdata = permuter(non_zero_mat)

    print("Finished permuting")

    # size: [N,]
    row_labels = permuter.row_labels_

    # size: [M,]
    col_labels = permuter.col_labels_

    n_row_clusters = len(np.unique(row_labels))
    n_col_clusters = len(np.unique(col_labels))

    goop_dict = {}
    cluster_mins = []
    sparse_goops = []

    clusters = list(product(range(n_row_clusters), range(n_col_clusters)))

    for l_row, l_col in tqdm(clusters):
        cluster_rows = np.where(permuter.row_labels_ == l_row)[0]
        cluster_cols = np.where(permuter.col_labels_ == l_col)[0]
        cluster_mat = mat[cluster_rows, :][:, cluster_cols]
        cluster_min = cluster_mat.min()
        cluster_mins.append(cluster_min)

        cluster_deltas = cluster_mat - cluster_min
        sparse_goop = make_sparse_goop(cluster_deltas)
        sparse_goops.append(sparse_goop)

    cluster_mins_goop = make_int_vector_goop(np.array(cluster_mins))
    goop_dict["m"] = cluster_mins_goop["data"]
    goop_dict["r"] = make_int_vector_goop(np.array(row_labels))["data"]
    goop_dict["c"] = make_int_vector_goop(np.array(col_labels))["data"]
    goop_dict["sh"] = (len(row_labels), len(col_labels))
    goop_dict["s"] = sparse_goops
    return goop_dict


def ungoop_data(goop_dict: dict[str, tuple[int] | IntVector]):
    shape = goop_dict["sh"]
    sparse_goops = goop_dict["s"]
    mat = np.zeros(shape)

    cluster_mins = make_array_from_int_vector_goop(
        {
            "data": IntVector.from_bitword_array(goop_dict["m"], 32),
            "shape": (len(goop_dict["m"]), 1),
        }
    )

    row_labels = make_array_from_int_vector_goop(
        {
            "data": IntVector.from_bitword_array(goop_dict["r"], 32),
            "shape": (sparse_goops[0]["n"], 1),
        }
    )
    col_labels = make_array_from_int_vector_goop(
        {
            "data": IntVector.from_bitword_array(goop_dict["c"], 32),
            "shape": (sparse_goops[0]["w"], 1),
        }
    )

    n_row_clusters = len(np.unique(row_labels))
    n_col_clusters = len(np.unique(col_labels))
    clusters = list(product(range(n_row_clusters), range(n_col_clusters)))
    for l_row, l_col in tqdm(clusters):
        c_idx = l_row * len(np.unique(col_labels)) + l_col
        cluster_rows = np.where(row_labels == l_row)[0]
        cluster_cols = np.where(col_labels == l_col)[0]
        cluster_mat = mat[cluster_rows, :][:, cluster_cols]
        cluster_min = cluster_mins[c_idx]
        cluster_deltas = make_array_from_int_vector_goop(
            {
                "data": IntVector.from_bitword_array(sparse_goops[c_idx]["d"], 32),
                "shape": (sparse_goops[c_idx]["n"], sparse_goops[c_idx]["w"]),
            }
        )
        cluster_mat = cluster_deltas + cluster_min
        mat[cluster_rows, :][:, cluster_cols] = cluster_mat
    return mat


# %%
goop_dict = make_goop_dict(mat)
goop_dict_flat = flatten_array_dict(goop_dict)
rprint(goop_dict_flat.keys())

# %%
np.savez_compressed("/tmp/goop.npz", **goop_dict_flat)
goop_size = os.path.getsize("/tmp/goop.npz")
print("Goop size:", goop_size)

gooped_bits = {**np.load("/tmp/goop.npz")}
gooped_bits_unflat = unflatten_array_dict(gooped_bits)
mat_recon = ungoop_data(gooped_bits_unflat)
print(np.allclose(mat, mat_recon))

np.savez_compressed("/tmp/mat.npz", csr_matrix(mat))
mat_size = os.path.getsize("/tmp/mat.npz")
print("Mat size:", mat_size)

# %%
plt.spy(mat)
# %%

from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix

p = reverse_cuthill_mckee(csr_matrix(mat))
# %%
plt.spy(mat[p][:, p])
# %%


# %%


# Open the npz file


# %%
np.random.seed(42)
A = np.random.randn(10, 10)
A = (A > 0.8).astype(int) * np.random.randint(0, 10, size=(10, 10))
print(A)
# %%

C = csr_matrix(A)
print(C.indptr)
print(C.indices)
print(C.data)
# %%
# Reconstruct matrix from data, indices, indptr
C2 = csr_matrix((C.data, C.indices, C.indptr))
C2.todense()
# %%
np.all(C2.todense() == A)

# %%
from scipy.sparse import bsr_array

# Count number of non-zero elements in A
C.nnz


# %%
