# %%

from bz2 import compress
import os
import numpy as np
from sklearn.cluster import SpectralBiclustering
from cell_squeeze.base import BiClusterMatrixPermuter
from scipy.sparse import load_npz
from matplotlib import pyplot as plt
from itertools import product
from cell_squeeze.bit_vector import IntVector
from rich import print as rprint
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, save_npz, load_npz
from cell_squeeze.utils import flatten_array_dict, unflatten_array_dict
from tqdm import tqdm
from rich import print as rprint
from typing import Any

# %%
# permuter = BiClusterMatrixPermuter(2, 2)

# # %% Load matrix for experimentation
# data_path = (
#     "../prepared/1k_mouse_kidney_CNIK_3pv3_raw_feature_bc_matrix/5000_5000_0.npz"
# )
# np.random.seed(0)
# mat = np.array(load_npz(data_path).todense())
# %%


def array2intvec(mat: np.ndarray, bitwidth: int = 32) -> tuple[IntVector, tuple[int]]:
    int_vec = IntVector(mat.size, bitwidth)
    for i, elem in enumerate(mat.reshape(-1)):
        int_vec[i] = elem
    return int_vec, mat.shape


def intvec2nparray(intvec: IntVector, shape: tuple[int]) -> np.ndarray:
    """De-serializes an IntVector goop dict into a numpy array"""
    array = intvec.to_numpy()
    N = np.prod(shape)
    return array[-N:].reshape(shape)


def mat_to_sparse_goop(mat: np.ndarray, method: str = "csr"):
    if method == "csr":
        sparse_mat = csr_matrix(mat)
    else:
        raise NotImplementedError
    return {
        "d": array2intvec(sparse_mat.data)[0],
        "i1": array2intvec(sparse_mat.indices)[0],
        "i2": array2intvec(sparse_mat.indptr)[0],
        "n": sparse_mat.shape[0],
        "w": sparse_mat.nnz,
    }


def sparse_goop_to_mat(sparse_goop: dict[str, Any], method: str = "csr"):
    data = intvec2nparray(sparse_goop["d"], (sparse_goop["w"],))
    indices = intvec2nparray(sparse_goop["i1"], (sparse_goop["w"],))
    indptr = intvec2nparray(sparse_goop["i2"], (sparse_goop["n"] + 1,))
    if method == "csr":
        return csr_matrix((data, indices, indptr)).todense()
    else:
        raise NotImplementedError


# np.random.seed(0)
# R = np.random.rand(31, 23)
# R = (R > 0.8).astype(int) * np.random.randint(0, 10, size=(31, 23))
# Rs = csr_matrix(R)
# print(Rs.data.shape, Rs.indices.shape, Rs.indptr.shape)
# iv = array2intvec(R)[0]
# R2 = intvec2nparray(iv, R.shape)
# print(np.allclose(R, R2))
# g = mat_to_sparse_goop(R)
# R2 = sparse_goop_to_mat(g)
# np.allclose(R, R2)

# # %%


def make_goop_dict(mat: np.ndarray, permuter: BiClusterMatrixPermuter):
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
        sparse_goop = mat_to_sparse_goop(cluster_deltas)
        sparse_goops.append(sparse_goop)

    goop_dict["m"] = array2intvec(np.array(cluster_mins))[0]
    goop_dict["r"] = array2intvec(np.array(row_labels))[0]
    goop_dict["c"] = array2intvec(np.array(col_labels))[0]
    goop_dict["sh"] = (len(row_labels), len(col_labels))
    goop_dict["s"] = sparse_goops
    return goop_dict


def ungoop_data(goop_dict: dict[str, tuple[int] | IntVector]):
    n_rows, n_cols = goop_dict["sh"]
    sparse_goops = goop_dict["s"]
    mat = np.zeros(shape=(n_rows, n_cols))

    row_labels = intvec2nparray(
        IntVector.from_bitword_array(goop_dict["r"], 32),
        (n_rows,),
    )

    col_labels = intvec2nparray(
        IntVector.from_bitword_array(goop_dict["c"], 32),
        (n_cols,),
    )

    n_row_clusters = len(np.unique(row_labels))
    n_col_clusters = len(np.unique(col_labels))
    n_clusters = n_row_clusters * n_col_clusters

    cluster_mins = intvec2nparray(
        IntVector.from_bitword_array(goop_dict["m"], 32),
        (n_clusters,),
    )

    clusters = list(product(range(n_row_clusters), range(n_col_clusters)))
    for l_row, l_col in tqdm(clusters):
        c_idx = l_row * len(np.unique(col_labels)) + l_col
        cluster_rows = np.where(row_labels == l_row)[0]
        cluster_cols = np.where(col_labels == l_col)[0]
        cluster_mat = mat[cluster_rows, :][:, cluster_cols]
        cluster_min = cluster_mins[c_idx]
        intvec_goop = {
            k: IntVector.from_bitword_array(v, 32) if k in {"d", "i1", "i2"} else v
            for k, v in sparse_goops[c_idx].items()
        }
        print(intvec_goop)
        cluster_deltas = sparse_goop_to_mat(intvec_goop, method="csr")
        cluster_mat = cluster_deltas + cluster_min
        mat[cluster_rows, :][:, cluster_cols] = cluster_mat
    return mat


if __name__ == "__main__":
    
    goop_dict = make_goop_dict(mat, permuter)
    goop_dict_flat = flatten_array_dict(goop_dict)
    rprint(goop_dict_flat.keys())

    # %%
    np.savez_compressed("/tmp/goop.npz", **goop_dict_flat, compressed=True)
    goop_size = os.path.getsize("/tmp/goop.npz")
    print("Goop size:", goop_size)

    gooped_bits = {**np.load("/tmp/goop.npz")}
    gooped_bits_unflat = unflatten_array_dict(gooped_bits)
    print(gooped_bits_unflat)
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
