import os
import random
from typing import Any
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, save_npz, load_npz
from scipy.io import mmwrite, mmread
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
    def __init__(self, gene_matrix):
        self.gene_matrix = gene_matrix

    def serialize(self, file_path: str):
        np.savez(file_path, self.gene_matrix)
        return os.stat(file_path).st_size

    def csrSerialize(self, file_path: str):
        np.savez_compressed(file_path, self.gene_matrix.tocsr())
        return os.stat(file_path).st_size

    def cscSerialize(self, file_path: str):
        np.savez_compressed(file_path, self.gene_matrix.tocsc())
        return os.stat(file_path).st_size

    def cooSerialize(self, file_path: str):
        np.savez_compressed(file_path, self.gene_matrix)
        return os.stat(file_path).st_size

    def mtxSerialize(self, file_path: str):
        mmwrite(file_path, self.gene_matrix)
        return os.stat(file_path).st_size

    def deserialize(self, file_path: str):
        np.load_npz

    def sample(self):
        m = self.gene_matrix
        return (
            m.getcol(random.randint(0, m.shape[1]))
            .getrow(random.randint(0, m.shape[0]))
            .todense()
            .var()
        )

    def sample_n(self, n):
        samples = [self.sample() for _ in range(n)]
        return np.mean(samples), np.std(samples)


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

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        print("fitting...")
        self.model.fit(matrix)
        print("fitting done")
        self.row_labels_ = np.array(self.model.row_labels_)
        self.col_labels_ = np.array(self.model.column_labels_)
        row_perm_ = np.argsort(self.model.row_labels_)
        col_perm_ = np.argsort(self.model.column_labels_)
        permuted_data = matrix[row_perm_][:, col_perm_]
        return permuted_data


def main():
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


if __name__ == "__main__":
    main()
