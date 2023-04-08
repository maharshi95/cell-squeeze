import os
from scipy.sparse import csr_matrix, csc_matrix, save_npz, load_npz
from sklearn.cluster import SpectralBiclustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Serializable:
    def serialize(self, file_path: str, compressed: bool = False):
        raise NotImplementedError

    @classmethod
    def deserialize(cls, file_path: str):
        raise NotImplementedError


class GeneMatrixSerializer:
    def __init__(self, gene_matrix: np.ndarray):
        self.gene_matrix = gene_matrix

    def to_array_dict(self):
        raise NotImplementedError

    def serialize(self, file_path: str, compressed: bool = False):
        array_dict = self.to_array_dict()
        if compressed:
            np.savez_compressed(file_path, **array_dict)
        np.savez(file_path, **array_dict)

    def deserialize(self, file_path: str):
        np.load_npz


class CRCMatrixSerializer(GeneMatrixSerializer):
    def _serialize(self):
        return csr_matrix(self.gene_matrix.matrix).toarray()


# create a random sparse matrix using scipy.sparse
mat = csr_matrix(np.random.randint(0, 2, size=(100, 100)))
# save matrix to file
filepath = "matrix.npz"

save_npz(filepath, mat)
loaded_mat = load_npz(filepath)
print(loaded_mat)

# load xls file using pandas

# %%

df = pd.read_csv("../data/gse61533_htseq.csv", index_col=0)
bool_mat = df.to_numpy().astype(bool).astype(int)
non_zero_rows = np.where(bool_mat.sum(axis=1) > 0)[0]
bool_mat = bool_mat[non_zero_rows, :]


model = SpectralBiclustering(n_clusters=(200, 10), random_state=0)
model.fit(bool_mat)


# %%
def get_label_permutation(n: int, labels: np.ndarray) -> np.ndarray:
    n_labels = np.max(labels) + 1
    indices = np.arange(n)
    selectors = labels == np.arange(n_labels)[:, None]
    return np.stack([indices] * n_labels)[selectors]


row_perm = get_label_permutation(bool_mat.shape[0], model.row_labels_)
col_perm = get_label_permutation(bool_mat.shape[1], model.column_labels_)

new_mat = bool_mat[row_perm, :][:, col_perm]
plt.imshow(new_mat, aspect="auto")
plt.show()
plt.imshow(bool_mat, aspect="auto")
# %%
