from argparse import ArgumentParser
import  os, glob, scipy
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm
import random as default_random
from itertools import product
import numpy as np


'''
Example usage:

python prepare_data.py -t synthetic -r 1000 -c 1000 -n 5 -mu 3 -si .5 -s .1 .2 .3
    creates 5*3 synthetic matricies of size 1000x1000. 
    5 each with sparisities .1, .2, .3 
    Each nonzero value comes from normal distribtuionwith mu=3 and sigma=.5

python prepare_data.py -t original -r 1000 -c 1000 -n 5 -p ../data/1k_hgmm_3p_LT_raw_feature_bc_matrix/
    creates 5 submatricies of size 1000x1000.
    submatrix comes from random continuious section of 1k_hgmm_3p_LT_raw_feature_bc_matrix

'''


def setup_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-t", "--type", type=str, required=True, choices=["synthetic", "original"])
    parser.add_argument("-p", "--orig-path", type=str,  default="../data/1k_hgmm_3p_LT_raw_feature_bc_matrix/")
    parser.add_argument("-o", "--out-path", type=str,  default="../prepared/")
    parser.add_argument("-r", type=int, required=True)
    parser.add_argument("-c", type=int, required=True)
    parser.add_argument("-n", "--num-files",type=int, required=True)
    parser.add_argument("-mu", type=float, required=False)
    parser.add_argument("-si", type=float, required=False)
    parser.add_argument("-s", "--sparsity",type=float, required=False, nargs='+')
    return parser


def save(mat, num, args, sparsity=None):
    base_path = args.out_path
    if args.type=="original":
        base_path = os.path.join(base_path,args.orig_path.split("/")[-2])
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        path = os.path.join(base_path,"_".join([str(x) for x in [args.r,args.c,num]])+".npz")
    else:
        base_path = os.path.join(base_path,args.type)
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        path = os.path.join(base_path,"_".join([str(x) for x in [args.r,args.c,args.mu,args.si,sparsity,num]])+".npz")
    
    
    np.savez_compressed(path, data=mat.data, indices=mat.indices,
             indptr=mat.indptr, shape=mat.shape)

def load(path):
    npz = np.load(path)
    return csc_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])

def remove_zero_rc(mat):
    return mat[mat.getnnz(1)>0][:,mat.getnnz(0)>0]


def original(args):
    matrix_dirs = glob.glob(args.orig_path+"*raw_feature_bc_matrix")
    assert(len(matrix_dirs)==1)
    in_path = matrix_dirs[0]

    mat = scipy.io.mmread(os.path.join(in_path, "matrix.mtx.gz")).tocsc()

    shape = mat.get_shape()
    sample_shape = (args.r, args.c)

    row_start_range = list(range(1, shape[0] - sample_shape[0]))
    col_start_range = list(range(1, shape[1] - sample_shape[1]))

    for run in tqdm(range(args.num_files)): 
        row_start, col_start = (default_random.choice(row_start_range), default_random.choice(col_start_range))
        submat = mat[row_start:row_start + sample_shape[0], col_start: col_start + sample_shape[1]]

        save(submat, run, args)


def generate_normal_vals(r,c,mu,sig,sparsity):
    num_nonzero = int(sparsity*r*c)
    return np.random.normal(mu, sig, num_nonzero)

def randomly_insert_vals(r,c,vals, pts):
    indicies = np.random.choice(len(pts), len(vals))
    rows,cols = zip(*pts[indicies])
    return csc_matrix((vals, (rows, cols)), shape=(r, c))

def synthetic(args):
    pts = np.array(list(product(list(range(args.r)), list(range(args.c)))))
    
    for run in tqdm(range(args.num_files)):
        for sparsity in args.sparsity:
            vals = generate_normal_vals(args.r, args.c, args.mu, args.si,sparsity)
            mat = randomly_insert_vals(args.r, args.c, vals, pts)

            save(mat, run, args, sparsity)


def main():
    args = setup_argparse().parse_args()

    if args.type == "original":
        original(args)
    else:
        synthetic(args)

    

if __name__ =="__main__":
    main()