# %%

import time
import numpy as np
from itertools import product
from scipy.sparse import csr_matrix
from tqdm import tqdm
from typing import Any, Sequence
from loguru import logger
from cell_squeeze.base import BiClusterMatrixPermuter
from cell_squeeze.utils import flatten_array_dict, unflatten_array_dict
from cell_squeeze import bit_vector


IntVec = bit_vector.IntVector


def dedense(dense_mat, rows, cols, n_rows, n_cols):
    submat_temp = np.zeros((len(rows) + 1, len(cols) + 1))
    submat_temp[1:, 1:] = dense_mat

    rows_ = np.zeros(n_rows, dtype=np.int32)
    rows_[rows] = 1 + np.arange(len(rows))

    cols_ = np.zeros(n_cols, dtype=np.int32)
    cols_[cols] = 1 + np.arange(len(cols))

    return submat_temp[rows_, :][:, cols_]


def bit_length(num: int | np.integer):
    """Minimum number of bits required to represent a number."""
    if isinstance(num, np.integer):
        num = int(num)
    return num.bit_length() + 1


def vec_bit_length(arr: Sequence[int]):
    """Minimum number of bits required to represent a number."""
    if hasattr(arr, "size"):
        size = arr.size
    else:
        size = len(arr)
    if size == 0:
        return 0
    return bit_length(np.max(arr))


def array2intvec(arr: np.ndarray, bitwidth: int, word_size: int):
    if arr.size == 0:
        return np.array([], dtype=np.uint16), arr.shape
    int_vec = IntVec(arr.size, bitwidth)
    fmt = "0{}b".format(bitwidth)
    bit_strings = [format(a, fmt) for a in arr.reshape(-1)]
    B = "".join(bit_strings)
    int_vec.bitvec = bit_vector.BitVector(bitstring=B)
    return int_vec.bitword_array(word_size), arr.shape


def array2intvec_slow(
    arr: np.ndarray, bitwidth: int, word_size: int
) -> tuple[np.ndarray, tuple[int]]:
    int_vec = IntVec(arr.size, bitwidth)

    for i, elem in enumerate(arr.reshape(-1)):
        try:
            int_vec[i] = elem
        except ValueError as e:
            logger.error("Bitwidth:{} Elem:{}".format(bitwidth, elem))
            raise e
    return int_vec.bitword_array(word_size), arr.shape


def intvec2nparray(
    intvec: np.ndarray,
    shape: Sequence[int],
    bitwidth: int,
    dtype=np.int32,
) -> np.ndarray:
    """De-serializes an IntVec goop dict into a numpy array"""
    n_elem = np.prod(shape)
    intvec = IntVec.from_bitword_array(intvec, bitwidth, n_elem)
    arr = intvec.to_numpy(dtype=dtype)
    N = np.prod(shape)
    return arr[-N:].reshape(shape)


def mat_to_sparse_goop(mat: np.ndarray, method: str = "csr", word_size: int = 32):
    if method == "csr":
        sparse_mat = csr_matrix(mat)
    else:
        raise NotImplementedError
    data_nbits = vec_bit_length(sparse_mat.data)
    indices_nbits = vec_bit_length(sparse_mat.indices)
    indptr_nbits = vec_bit_length(sparse_mat.indptr)
    return {
        "d": array2intvec(sparse_mat.data, data_nbits, word_size)[0],
        "i1": array2intvec(sparse_mat.indices, indices_nbits, word_size)[0],
        "i2": array2intvec(sparse_mat.indptr, indptr_nbits, word_size)[0],
        "n": sparse_mat.shape[0],
        "w": sparse_mat.nnz,
        "bd": data_nbits,
        "b1": indices_nbits,
        "b2": indptr_nbits,
    }


def sparse_goop_to_mat(
    sparse_goop: dict[str, Any], method: str = "csr", keep_sparse: bool = False
):
    bw_data = sparse_goop["bd"]
    bw_ind = sparse_goop["b1"]
    bw_indptr = sparse_goop["b2"]
    nnz = sparse_goop["w"]
    n_rows = sparse_goop["n"]
    data = intvec2nparray(
        sparse_goop["d"],
        shape=(nnz,),
        bitwidth=bw_data,
    )
    indices = intvec2nparray(
        sparse_goop["i1"],
        shape=(nnz,),
        bitwidth=bw_ind,
    )
    indptr = intvec2nparray(
        sparse_goop["i2"],
        shape=(n_rows + 1,),
        bitwidth=bw_indptr,
    )
    if method == "csr":
        mat = csr_matrix((data, indices, indptr))
        return mat if keep_sparse else mat.toarray()
    else:
        raise NotImplementedError


# %%


def make_goop_dict(
    mat: np.ndarray, permuter: BiClusterMatrixPermuter, word_size: int = 32
):
    # mat shape: [N, M]

    n_bits_theoretical = 0

    non_zero_rows = np.where(mat.sum(axis=1) > 0)[0]
    non_zero_cols = np.where(mat.sum(axis=0) > 0)[0]
    non_zero_row_cols = np.concatenate([non_zero_rows, non_zero_cols])

    non_zero_mat = mat[non_zero_rows, :][:, non_zero_cols]

    permuter(non_zero_mat)

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
        sparse_goop = mat_to_sparse_goop(
            cluster_deltas, method="csr", word_size=word_size
        )
        sparse_goops.append(sparse_goop)
        n_bits_theoretical += vec_bit_length(cluster_deltas) * cluster_mat.size

    bw_cluster_mins = vec_bit_length(cluster_mins)
    bw_row_labels = vec_bit_length(row_labels)
    bw_col_labels = vec_bit_length(col_labels)
    bw_nzrc = vec_bit_length(non_zero_row_cols)
    bw_array = np.array(
        [bw_cluster_mins, bw_row_labels, bw_col_labels, bw_nzrc], dtype=np.int8
    )

    n_bits_theoretical += bw_cluster_mins * len(cluster_mins)
    n_bits_theoretical += bw_row_labels * len(row_labels)
    n_bits_theoretical += bw_col_labels * len(col_labels)
    n_bits_theoretical += bw_nzrc * len(non_zero_row_cols)
    n_bits_theoretical += 2 * 32  # shape

    cluster_mins_intvec, _ = array2intvec(
        np.array(cluster_mins), bw_cluster_mins, word_size
    )

    row_labels_intvec, _ = array2intvec(row_labels, bw_row_labels, word_size)
    col_labels_intvec, _ = array2intvec(col_labels, bw_col_labels, word_size)
    non_zero_matix_shape = np.array(non_zero_mat.shape, dtype=np.int32)

    non_zero_rc_intvec = array2intvec(non_zero_row_cols, bw_nzrc, word_size)[0]
    goop_dict = {
        "m": cluster_mins_intvec,
        "r": row_labels_intvec,
        "c": col_labels_intvec,
        "nz": non_zero_rc_intvec,
        "sh": non_zero_matix_shape,
        "s": sparse_goops,
        "b": bw_array,
        "M": np.int32(mat.shape[0]),
        "N": np.int32(mat.shape[1]),
    }
    flat_goopdict = flatten_array_dict(goop_dict)
    return flat_goopdict, row_labels, col_labels, n_bits_theoretical


def ungoop_data(flat_goop_dict: dict[str, tuple[int] | np.ndarray]):
    goop_dict = unflatten_array_dict(flat_goop_dict)
    bw_cluster_width, bw_row_labels, bw_col_labels, bw_nzrc = goop_dict["b"]
    sparse_goops = goop_dict["s"]
    n_non_zero_rows, n_non_zero_cols = goop_dict["sh"]

    # Nonrow rows and cols
    non_zero_rc_intvec = goop_dict["nz"]
    non_zero_row_cols = intvec2nparray(
        non_zero_rc_intvec, shape=(n_non_zero_rows + n_non_zero_cols,), bitwidth=bw_nzrc
    )
    non_zero_rows = non_zero_row_cols[:n_non_zero_rows]
    non_zero_cols = non_zero_row_cols[n_non_zero_rows:]

    row_labels = intvec2nparray(
        goop_dict["r"], shape=(n_non_zero_rows,), bitwidth=bw_row_labels
    )
    col_labels = intvec2nparray(
        goop_dict["c"], shape=(n_non_zero_cols,), bitwidth=bw_col_labels
    )

    n_row_clusters = len(np.unique(row_labels))
    n_col_clusters = len(np.unique(col_labels))
    n_clusters = n_row_clusters * n_col_clusters

    cluster_mins = intvec2nparray(
        goop_dict["m"], shape=(n_clusters,), bitwidth=bw_cluster_width
    )

    clusters = list(product(range(n_row_clusters), range(n_col_clusters)))

    mat_nz = np.zeros((n_non_zero_rows, n_non_zero_cols))
    for l_row, l_col in tqdm(clusters):
        c_idx = l_row * len(np.unique(col_labels)) + l_col
        cluster_rows = np.where(row_labels == l_row)[0]
        cluster_cols = np.where(col_labels == l_col)[0]
        cluster_mat = mat_nz[cluster_rows, :][:, cluster_cols]
        cluster_min = cluster_mins[c_idx]
        cluster_deltas = sparse_goop_to_mat(sparse_goops[c_idx], method="csr")
        cluster_mat = cluster_deltas + cluster_min
        mat_nz[cluster_rows, :][:, cluster_cols] = cluster_mat

    N, M = goop_dict["M"], goop_dict["N"]
    mat = dedense(mat_nz, non_zero_rows, non_zero_cols, N, M)
    return mat


# %%
if __name__ == "__main__":
    # Testing array2intvec and intvec2nparray

    WORD_SIZE = 32

    logger.info("testing speeds for array2intvec and array2intvec_fast")
    test_shape = (500, 500)
    np.random.seed(0)
    mask = np.random.rand(*test_shape) > 0.7
    mat_s = np.random.randint(0, 16, size=test_shape) * mask
    tic = time.time()
    iv = array2intvec_slow(mat_s, 4, WORD_SIZE)[0]
    tac = time.time()
    logger.info("array2intvec time: {:.3f}", tac - tic)

    tic = time.time()
    iv = array2intvec(mat_s, 4, WORD_SIZE)[0]
    tac = time.time()
    logger.info("array2intvec_fast time: {:.3f}", tac - tic)

    test_shape = (32, 29)
    np.random.seed(0)
    logger.info("Testing array2intvec and intvec2nparray")
    for it in range(10):
        mask = np.random.rand(*test_shape) > 0.7
        mat_s = np.random.randint(0, 10, size=test_shape) * mask

        iv = array2intvec_slow(mat_s, 4, WORD_SIZE)[0]
        iv_fast = array2intvec(mat_s, 4, WORD_SIZE)[0]
        assert np.all(iv == iv_fast), "Fast and slow methods don't match"
        mat_r = intvec2nparray(iv, test_shape, bitwidth=4)
        if np.all(mat_s == mat_r):
            logger.info(f"TC {it + 1} [PASS]")
        else:
            logger.error(f"TC {it + 1} [FAIL]")

    logger.info("Testing mat_to_sparse_goop and sparse_goop_to_mat")
    # Testing mat_to_sparse_goop and sparse_goop_to_mat
    for it in range(10):
        mask = np.random.rand(*test_shape) > 0.7
        mat_s = np.random.randint(0, 10, size=test_shape) * mask

        g = mat_to_sparse_goop(mat_s, method="csr", word_size=WORD_SIZE)
        mat_r = sparse_goop_to_mat(g, method="csr", keep_sparse=False)

        if np.all(mat_s == mat_r):
            logger.info(f"TC {it + 1} [PASS]")
        else:
            logger.error(f"TC {it + 1} [FAIL]")

    logger.info("Testing make_goop_dict and ungoop_data")
    test_shape = (10000, 10000)

    break_mat = np.ones((4, 4), dtype=np.int32)
    permuter = BiClusterMatrixPermuter(5, 5)
    break_mat[0, 0] = 0

    mask = np.random.rand(*test_shape) > 0.7
    mat_s = np.random.randint(0, 16, size=test_shape) * mask

    test_mats = [mat_s]
    # Testing make_goop_dict and ungoop_data
    for it, mat_s in enumerate(test_mats):
        # mask = np.random.rand(*test_shape) > 0.7
        # mat_s = np.random.randint(0, 124334, size=test_shape) * mask

        gdict, row_labels, col_labels, nbits_theoretical = make_goop_dict(
            mat_s, permuter, WORD_SIZE
        )
        # mat_r = ungoop_data(gdict)

        # if np.all(mat_s == mat_r):
        #     logger.info(f"TC {it + 1} [PASS]")
        # else:
        #     logger.error(f"TC {it + 1} [FAIL]")

# %%
# exit()

# # %%
# goop_dict = make_goop_dict(mat)
# rprint(goop_dict_flat.keys())

# # %%
# np.savez_compressed("/tmp/goop.npz", **goop_dict_flat, compressed=True)
# goop_size = os.path.getsize("/tmp/goop.npz")
# print("Goop size:", goop_size)

# gooped_bits = {**np.load("/tmp/goop.npz")}
# gooped_bits_unflat = unflatten_array_dict(gooped_bits)
# print(gooped_bits_unflat)
# mat_recon = ungoop_data(gooped_bits_unflat)
# print(np.allclose(mat, mat_recon))

# np.savez_compressed("/tmp/mat.npz", csr_matrix(mat))
# mat_size = os.path.getsize("/tmp/mat.npz")
# print("Mat size:", mat_size)

# # %%
# plt.spy(mat)
# # %%

# from scipy.sparse.csgraph import reverse_cuthill_mckee
# from scipy.sparse import csr_matrix

# p = reverse_cuthill_mckee(csr_matrix(mat))
# # %%
# plt.spy(mat[p][:, p])
# # %%


# # %%


# # Open the npz file


# # %%
# np.random.seed(42)
# A = np.random.randn(10, 10)
# A = (A > 0.8).astype(int) * np.random.randint(0, 10, size=(10, 10))
# print(A)
# # %%

# C = csr_matrix(A)
# print(C.indptr)
# print(C.indices)
# print(C.data)
# # %%
# # Reconstruct matrix from data, indices, indptr
# C2 = csr_matrix((C.data, C.indices, C.indptr))
# C2.todense()
# # %%
# np.all(C2.todense() == A)

# # %%
# from scipy.sparse import bsr_array

# # Count number of non-zero elements in A
# C.nnz


# # %%
# a = np.int64(12)
# print(a)
# a = 12
# a.bit_length()
# # number of max bits in a
# # %%

# mat = np.zeros((10, 10))
# submat = np.random.randint(0, 10, size=(3, 3))
# rows = np.array([0, 4, 9])
# cols = np.array([1, 5, 7])


# dense_mat = np.array([[3, 2, 1], [9, 8, 9]])
# rows = np.array([4, 9])
# cols = np.array([1, 5, 7])
# dede = dedense(dense_mat, rows, cols, 10, 10)
# dede[rows, :][:, cols]


# # %%
