# %%
from einops import pack
import numpy as np
from sklearn.cluster import SpectralBiclustering


class MatrixPermuter:
    def __init__(self) -> None:
        pass

    def __call__(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class BiClusterMatrixPermuter(MatrixPermuter):
    def __init__(self, n_row_clusters: int, n_col_clusters) -> None:
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters

        self.model = SpectralBiclustering(
            n_clusters=(n_row_clusters, n_col_clusters), random_state=0
        )

    def __call__(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.fit(matrix)
        self.row_perm_ = np.argsort(self.model.row_labels_)
        self.col_perm_ = np.argsort(self.model.column_labels_)
        permuted_data = matrix[self.row_perm_][:, self.col_perm_]
        return permuted_data


# %%
permuter = BiClusterMatrixPermuter(200, 10)
# Create a random sparse matrix
np.random.seed(0)
mat = np.random.randn(10000, 10000)
mask = np.random.randint(0, 10, size=(10000, 10000)) == 0
mat = mat * mask
# %%

pdata = permuter(mat)
# %%
from matplotlib import pyplot as plt

plt.spy(pdata)
# %%
plt.spy(mat)
# %%

from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix

p = reverse_cuthill_mckee(csr_matrix(mat))
# %%
plt.spy(mat[p][:, p])
# %%
import struct

unpack_formats = {
    8: ">B",
    16: ">H",
    32: ">I",
    64: ">Q",
}

dtypes = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64,
}


def long_int_to_int_vector(num, bit_length=64):
    # Determine the number of 64-bit integers needed to represent the long int
    assert bit_length % 8 == 0, "bit_length must be a multiple of 8"
    word_size = bit_length // 8
    print("Word size", word_size)
    num_words = (num.bit_length() + bit_length - 1) // bit_length

    # Pack the long int into binary format using big-endian byte order
    total_bytes = num_words * word_size
    packed = num.to_bytes(total_bytes, byteorder="big")
    print(len(packed))
    # padding = b"\x00" * (num_words * word_size - len(packed))
    # packed = padding + packed
    # Unpack the binary data into a list of int64 integers
    word_list = []
    unpack_format = unpack_formats[bit_length]
    dtype = dtypes[bit_length]

    for i in range(num_words):
        start = i * word_size
        end = start + word_size
        segment = packed[start:end]
        print(start, end, segment, word_list)
        print("Segment length", len(segment))
        print("packed legnth", len(packed))
        word = struct.unpack(unpack_format, segment)[0]
        word_list.append(word)

    return np.array(word_list, dtype=dtype)

def compress_array(array, bit_length=64):
    assert bit_length % 8 == 0, "bit_length must be a multiple of 8"
    segments = []
    format = unpack_formats[bit_length]
    for word in vector:
        segments.append(struct.pack(format, word))
    packed = b"".join(segments)
    return int.from_bytes(packed, byteorder="big"

def int_vector_to_long_int(vector, bit_length=64):
    assert bit_length % 8 == 0, "bit_length must be a multiple of 8"
    segments = []
    format = unpack_formats[bit_length]
    for word in vector:
        segments.append(struct.pack(format, word))
    packed = b"".join(segments)
    return int.from_bytes(packed, byteorder="big")


# %%
from BitVector import BitVector

b = BitVector(intVal=33234354**2 + 343541**5)
print(b.length())
print(str(b), int(b))
vec = long_int_to_int_vector(int(b), 32)
b2 = int_vector_to_long_int(vec, 32)
b.intValue() == b2
# %%
