import csv, gzip, os, glob
import pandas as pd
import scipy.io
from scipy.sparse import csc_matrix
from argparse import ArgumentParser
from base import *


def setup_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, default="../data/")
    return parser


def main():
    args = setup_argparse().parse_args()
    datapath = args.datapath

    matrix_dirs = glob.glob(datapath+"*/raw_feature_bc_matrix")

    names = []
    dense_elems = []
    non_zero_elems = []
    uncomp_size = []
    csc_size = []
    csr_size = []
    mtx_size = []

    temp_path_npz= "temp.npz"
    temp_path_mtx = "temp.mtx"

    for path in matrix_dirs:

        mat = GeneMatrixSerializer(scipy.io.mmread(os.path.join(path, "matrix.mtx.gz")))


        names.append(path[len(datapath):])


        dense_elems.append(mat.gene_matrix.get_shape()[0]*mat.gene_matrix.get_shape()[1])
        non_zero_elems.append(mat.gene_matrix.count_nonzero())

        mtx_size.append(mat.mtxSerialize(temp_path_mtx))

        csc_size.append(mat.cscSerialize(temp_path_npz))

        csr_size.append(mat.csrSerialize(temp_path_npz))



    os.remove(temp_path_npz)
    os.remove(temp_path_mtx)

    results = pd.DataFrame.from_dict({"name":names, "num_dense_elems":dense_elems, "num_sparse_elems":non_zero_elems,  "csc":csc_size, "csr":csr_size, "mtx":mtx_size})
    results.to_csv("results.csv")


if __name__ == "__main__":
    main()