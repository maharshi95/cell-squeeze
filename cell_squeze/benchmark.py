# %%
import csv, gzip, os, glob
import pandas as pd
import scipy.io
import scipy
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt

from argparse import ArgumentParser
from base import *
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score


import numpy as np
from scipy.sparse import random


def sparsity_rate(sparse_matrix):
    #nnz = sparse_matrix.nnz # number of nonzero elements in the matrix
    nnz = np.count_nonzero(sparse_matrix)
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    return nnz / total_elements

def generate_sparse_matrix(n, m, p, mu, sigma):
    """
    Generate a sparse matrix with sparsity rate p and random values using a distribution parameterized by mu and sigma.

    Args:
    n x m matrix (int):  rows x columns
    p (float): sparsity rate
    mu (float): distribution mean
    sigma (float): distribution standard deviation
    """

    nnz = int(np.ceil(n * m * p)) # number of nonzero elements in the matrix
    row_indices = np.random.choice(n, nnz) # row indices of the nonzero elements
    col_indices = np.random.choice(m, nnz) # column indices of the nonzero elements

    values = np.random.normal(mu, sigma, nnz)
    values = np.abs(values).astype(int) # convert to positive integers
    sparse_matrix = scipy.sparse.coo_matrix((values, (row_indices, col_indices)), shape=(n, m))

    return sparse_matrix


def setup_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, default="../data/")
    return parser

def make_nonzero_matrix(M):
    non_zero_rows = np.where(np.sum(M, axis=1) != 0)[0]
    M = M[non_zero_rows, :]
    non_zero_cols = np.where(np.sum(M, axis=0) != 0)[0]
    M = M[:, non_zero_cols]
    return M

# %%
# def main():
    args = setup_argparse().parse_args()
    datapath = args.datapath

    matrix_dirs = glob.glob(datapath+"*/raw_feature_bc_matrix")

#     names = []
#     dense_elems = []
#     non_zero_elems = []
#     uncomp_size = []
#     csc_size = []
#     csr_size = []
#     mtx_size = []

#     temp_path_npz= "temp.npz"
#     temp_path_mtx = "temp.mtx"


    #my_data = np.genfromtxt('../data/gse61533_htseq.csv', delimiter=',', skip_header=1, usecols=list(range(1, 97)))[:100, :]
    # orig_data = pd.read_csv('../data/gse61533_htseq.csv', index_col='ID').values
    # my_data = make_nonzero_matrix(orig_data)[:100, :]
    # my_data = make_nonzero_matrix(my_data)
    # # print(my_data.shape) 

    # plt.matshow(my_data, cmap=plt.cm.Blues)
    # plt.savefig("toy_heatmap.png")
    
    # model = SpectralBiclustering(n_clusters=(5, 5), random_state=0)
    # model.fit(my_data)
    # print(model.column_labels_)
    # fit_data = my_data[np.argsort(model.row_labels_)]
    # fit_data = fit_data[:, np.argsort(model.column_labels_)]
    # plt.matshow(fit_data, cmap=plt.cm.Blues)
    # plt.savefig("toy_heatmap_fit.png")
    # assert False


    for path in matrix_dirs:

        print(path)
        mat = GeneMatrixSerializer(scipy.io.mmread(os.path.join(path, "matrix.mtx.gz")))
        n = mat.gene_matrix.get_shape()[0]
        m = mat.gene_matrix.get_shape()[1]

    #     nonzero_indices = scipy.sparse.find(mat.gene_matrix)
    #     num_nonzero = len(nonzero_indices[0])
    #     sp = sparsity_rate(mat.gene_matrix)

    #     rand_indices = np.random.choice(num_nonzero, int(0.1*num_nonzero), replace=False)
    #     sampled_values = nonzero_indices[2][rand_indices]
    #     print(np.mean(sampled_values), np.std(sampled_values), sp)

    #     toy = generate_sparse_matrix(1000, 1000, 0.4, np.mean(sampled_values), np.std(sampled_values))


    #     model = SpectralBiclustering(n_clusters=(10, 5), random_state=0)
    #     model.fit(toy)

    #     plt.matshow(toy.toarray(), cmap=plt.cm.Blues)
    #     plt.savefig("toy_heatmap.png")

    #     fit_data = toy.toarray()[np.argsort(model.row_labels_)]
    #     fit_data = fit_data[:, np.argsort(model.column_labels_)]
    #     plt.matshow(fit_data, cmap=plt.cm.Blues)
    #     plt.savefig("toy_heatmap_fit.png")



        # names.append(path[len(datapath):])


        # dense_elems.append(mat.gene_matrix.get_shape()[0]*mat.gene_matrix.get_shape()[1])
        # non_zero_elems.append(mat.gene_matrix.count_nonzero())

        # mtx_size.append(mat.mtxSerialize(temp_path_mtx))
        # csc_size.append(mat.cscSerialize(temp_path_npz))
        # csr_size.append(mat.csrSerialize(temp_path_npz))

        #print(mat.sample_n(1000))



    # os.remove(temp_path_npz)
    # os.remove(temp_path_mtx)

    # results = pd.DataFrame.from_dict({"name":names, "num_dense_elems":dense_elems, "num_sparse_elems":non_zero_elems,  "csc":csc_size, "csr":csr_size, "mtx":mtx_size})
    # results.to_csv("results.csv")


# if __name__ == "__main__":
#     main()
# %%

orig_data = pd.read_csv('../data/gse61533_htseq.csv', index_col='ID').values

my_data = make_nonzero_matrix(orig_data)[:1000, :]
my_data = make_nonzero_matrix(my_data)

# %%
print(sparsity_rate(my_data)   )
my_data = np.where(my_data > 0, 1, 0)
print(sparsity_rate(my_data))

# %% 

# print(my_data.shape) 
n = my_data.shape[0]
m = my_data.shape[1]

print((n + m) / (n*m))

print(np.count_nonzero(my_data)/(n*m))

plt.spy(my_data)
plt.show()

# %%
N_ROW_CLUSTERS = 5
N_COL_CLUSTERS = 5

model = SpectralBiclustering(n_clusters=(N_ROW_CLUSTERS, N_COL_CLUSTERS), random_state=0)
model.fit(my_data)
row_perm = np.argsort(model.row_labels_)
col_perm = np.argsort(model.column_labels_)
fit_data = my_data[row_perm]
fit_data = fit_data[:, col_perm]
plt.spy(fit_data)

# %%  
# plt.(
#     np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
#     cmap=plt.cm.Blues,
# )
# plt.title("Checkerboard structure of rearranged data")

# %% 





# %%
row_labels = model.row_labels_
col_labels = model.column_labels_

row_cluster_spans = []
col_cluster_spans = []

labels_R = row_labels[row_perm]
labels_C = col_labels[col_perm]

print("Sparsity rate of the original matrix: ", sparsity_rate(my_data))
print("Range of values of the original matrix: ", np.max(my_data)-np.min(my_data) + 1)

for i_row in range(N_ROW_CLUSTERS):
    for j_col in range(N_COL_CLUSTERS):
        row_spans = np.where(labels_R == i_row)[0]
        col_spans = np.where(labels_C == j_col)[0]
        row_cluster_spans.append(row_spans)
        col_cluster_spans.append(col_spans)
        mat_slice = my_data[row_spans, :][:, col_spans]
        print(f"Sparsity rate: [{i_row},{j_col}]: {sparsity_rate(mat_slice):.3f}\tShape: {mat_slice.shape}")
        print(f"Range of values: [{i_row},{j_col}]: {np.max(mat_slice)-np.min(mat_slice) + 1}")



# %%

path = "../data/1k_hgmm_3p_LT_raw_feature_bc_matrix.tar.gz"


mat = GeneMatrixSerializer(scipy.io.mmread(path))

