# %%
import csv, gzip, os, glob
import pandas as pd
import scipy.io
import scipy
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser
from base import *
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.metrics import consensus_score
import random as default_random

import numpy as np
from scipy.sparse import random

# %% 
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

#orig_data = pd.read_csv('../data/gse61533_htseq.csv', index_col='ID').values

brain_tumor_path = Path("/Users/styx97/Projects/cell-squeeze/data/Brain_Tumor_3p_raw_feature_bc_matrix/raw_feature_bc_matrix")
matrix_path = brain_tumor_path / "matrix.mtx.gz"

mat = GeneMatrixSerializer(scipy.io.mmread(matrix_path)).gene_matrix.tocsc()

print(mat.get_shape())

# %% 
mat.get_shape()
subsample_shape = 10000

row_start_range = list(range(1, mat.get_shape()[0] - subsample_shape))
col_start_range = list(range(1, mat.get_shape()[1] - subsample_shape)) 

# %%
def create_biclusters(matrix, n_row_clusters, n_col_clusters):
        model = SpectralBiclustering(n_clusters=(n_row_clusters, n_col_clusters), random_state=0)
        #model = SpectralCoclustering(n_clusters=5, random_state=0)
        model.fit(matrix)
        row_perm = np.argsort(model.row_labels_)
        col_perm = np.argsort(model.column_labels_)
        fit_data = matrix[row_perm]
        fit_data = fit_data[:, col_perm]
        return fit_data, row_perm, col_perm, model

def bits_needed(range_: int):
        return np.ceil(np.log2(range_+1))

def mat_range(mat):
    return np.max(mat)-np.min(mat) + 1


# %%
num_runs = 10
run_results = {}


for sample_run in tqdm(range(num_runs)): 
    print("run: ", sample_run)
    temp_results = {}
    row_start, col_start = (default_random.choice(row_start_range), default_random.choice(col_start_range))
    submat = mat[row_start:row_start + subsample_shape, col_start: col_start + subsample_shape].A
    #orig_data = pd.read_csv('../data/gse61533_htseq.csv', index_col='ID').values
    #my_data = make_nonzero_matrix(orig_data)[:500, :]
    my_data = make_nonzero_matrix(submat)
    plt.spy(my_data, markersize=0.3, aspect='auto')
    plt.title("Original Data")
    plt.show()

    #print(sparsity_rate(my_data))
    bin_data = np.where(my_data > 0, 1, 0)
    #print(sparsity_rate(my_data))
    temp_results["sparsity_rate_orig"] = sparsity_rate(my_data)

    # print(my_data.shape) 
    n = my_data.shape[0]
    m = my_data.shape[1]

    #print((n + m) / (n*m))

    #print(np.count_nonzero(my_data)/(n*m))

    #plt.spy(my_data)
    #plt.show()

    # plt.(
    #     np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
    #     cmap=plt.cm.Blues,
    # )
    # plt.title("Checkerboard structure of rearranged data")
    row_indices = np.argsort(np.sum(bin_data, axis=1))
    col_indices = np.argsort(np.sum(bin_data, axis=0))

    # permute rows and columns using row_indices and col_indices
    permuted_data = my_data[row_indices, :][:, col_indices]
    plt.spy(permuted_data, markersize=0.4, aspect='auto')
    plt.title("Permuted data")
    plt.show()

    N_ROW_CLUSTERS = 5
    N_COL_CLUSTERS = 5 

    fit_data, row_perm, col_perm, model = create_biclusters(permuted_data, N_ROW_CLUSTERS, N_COL_CLUSTERS)
    #plt.spy(fit_data)

    plt.spy(fit_data, markersize=0.3, aspect='auto')
    plt.title("Biclustered data")
    plt.show()

    row_labels = model.row_labels_
    col_labels = model.column_labels_

    row_cluster_spans = []
    col_cluster_spans = []

    labels_R = row_labels[row_perm]
    labels_C = col_labels[col_perm]

    print("Sparsity rate of the original matrix: ", sparsity_rate(fit_data))
    print("Range of values of the original matrix: ", np.max(fit_data)-np.min(fit_data) + 1)
    temp_results["sparsity_rate_whole"] = sparsity_rate(fit_data)
    temp_results["original_range"] = np.max(fit_data)-np.min(fit_data) + 1

    sparsity_rates = []
    cluster_sizes = [] # submatrix size
    cluster_ranges = [] # max - min + 1 for that submatrix
    cluster_nonzeros = [] # number of nonzero elements in that submatrix
    cluster_shapes = []

    for i_row in range(N_ROW_CLUSTERS):
        for j_col in range(N_COL_CLUSTERS):
            row_spans = np.where(labels_R == i_row)[0]
            col_spans = np.where(labels_C == j_col)[0]
            row_cluster_spans.append(row_spans)
            col_cluster_spans.append(col_spans)
            mat_slice = fit_data[row_spans, :][:, col_spans]
            sparsity_rates.append(sparsity_rate(mat_slice))
            cluster_sizes.append(mat_slice.shape[0]*mat_slice.shape[1])
            cluster_shapes.append(mat_slice.shape)
            cluster_ranges.append(np.max(mat_slice)-np.min(mat_slice) + 1)
            cluster_nonzeros.append(np.count_nonzero(mat_slice))
            # print(f"Sparsity rate: [{i_row},{j_col}]: {sparsity_rate(mat_slice):.3f}\tShape: {mat_slice.shape}")
            # print(f"Range of values: [{i_row},{j_col}]: {np.max(mat_slice)-np.min(mat_slice) + 1}")

    #plt.hist(sparsity_rates, bins=30)


    #plt.scatter(cluster_sizes, sparsity_rates)
    #plt.xlabel("Cluster size")
    #plt.ylabel("Sparsity rate")
    #plt.show()



    assert(len(sparsity_rates) == len(cluster_sizes))
    
    total_nz_counts = np.sum(bin_data)
    N_COORDS = 2
    index_bits = bits_needed(max(bin_data.shape))
    mm_baseline_size = total_nz_counts * bits_needed(mat_range(fit_data))
    mm_baseline_size += total_nz_counts * (N_COORDS * bits_needed(index_bits))

    print("Baseline size: ", mm_baseline_size)

    temp_results["mm_baseline_size"] = mm_baseline_size
    bic_mm_size = 0
    for cluster_nz_count, cluster_range, cluster_shape in zip(cluster_nonzeros, cluster_ranges, cluster_shapes):
        bic_mm_size += cluster_nz_count * bits_needed(cluster_range)
        index_bits = bits_needed(max(cluster_shape))
        bic_mm_size += cluster_nz_count * (N_COORDS * bits_needed(index_bits))
    bic_mm_size += N_ROW_CLUSTERS * bits_needed(N_ROW_CLUSTERS)
    bic_mm_size += N_COL_CLUSTERS * bits_needed(N_COL_CLUSTERS)
    
    temp_results["bic_mm_size"] = bic_mm_size
    print("Biclustering+MM size: ", bic_mm_size)
    
    plt.hist(cluster_ranges, bins=30, ec='black')
    plt.show()

    run_results[sample_run] = temp_results
    print("/n/n")

# %%

df = pd.DataFrame.from_dict(run_results, orient="index")

# %%
df['difference'] = df['mm_baseline_size'] - df['bic_mm_size']
df['difference_percent'] = df['difference'] / df['mm_baseline_size'] * 100
print(df['difference_percent'].mean(), df['difference_percent'].std())


# %%

