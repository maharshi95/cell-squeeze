"""
Script that performs a range of experiments on sqeezing 
a count matrix into a sparse goop.

"""
import argparse 
import os 
from pathlib import Path
import numpy as np 
from scipy.sparse import load_npz, csr_matrix, csc_matrix, coo_matrix, bsr_matrix
#from cell_squeeze.benchmark import BiClusterMatrixPermuter, make_goop_dict, flatten_array_dict
from cell_squeeze.base import BiClusterMatrixPermuter
from cell_squeeze.goop import make_goop_dict 

from loguru import logger
import json
from typing import Any, Optional


def add_arguments(parser: Optional[argparse.ArgumentParser] = None,) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser()

    parser.add_argument(
        "--matrix_path",
        type=str,
        help="Path to the count matrix.",
        required=True,
    ) 

    parser.add_argument(
        '--n_row_clusters', 
        type=int,
        help="Number of row clusters to use.",
        required=True,
    )

    parser.add_argument(
        '--word_size',
        type=int,
        help="Word size to use to build goop dict.",
        required=False,
        default=32,
    )

    parser.add_argument(
        '--n_col_clusters',
        type=int,
        help="Number of column clusters to use.",
        required=True,
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help="Path to the output directory.",
        required=True,
    )

    # add a debug flag. if debug is true then we dont save the goop dict
    # debug is False by default
    parser.add_argument(
        '--debug',
        action='store_true',
        help="If true, then we don't save the goop dict.",
        required=False,
    )

    return parser 

def find_default_bits(mat):
    # find number of bits needed to store in matrix market format 
    # theoretically 
    max_bitsize = 64
    num_nonzero_elems = np.count_nonzero(mat)
    N_COORDS = 2 

    # number of bits needed to store the coordinates 
    coord_bits = num_nonzero_elems * N_COORDS * max_bitsize
    # number of bits needed to store the values
    value_bits = max_bitsize * num_nonzero_elems
    
    return value_bits + coord_bits
    

def compute_sizes(mat, n_row_clusters, n_col_clusters):
    """Compute the sizes of the row and column clusters."""
    permuter = BiClusterMatrixPermuter(n_row_clusters, n_col_clusters)
    
    logger.info("Biclustering and computing goop dict.")
    goop_dict, row_labels, col_labels, n_bits_theoretical = make_goop_dict(mat, permuter)

    default_bits = find_default_bits(mat)
    logger.info(f"Default bits: {default_bits}")

    # compute size of goop 
    np.savez_compressed("/tmp/goop_c.npz", **goop_dict)
    goop_size = os.path.getsize("/tmp/goop_c.npz")
    np.savez("/tmp/goop_uc.npz", **goop_dict, compress=False)
    goop_size_uncompressed = os.path.getsize("/tmp/goop_uc.npz")


    logger.info("Computing sizes of other formats of sparse matrices.")
    # compute size of csr matrix 
    np.savez_compressed("/tmp/mat_csr.npz", csr_matrix(mat))
    csr_size = os.path.getsize("/tmp/mat_csr.npz")

    np.savez_compressed("/tmp/mat_csr.npz", csr_matrix(mat), compressed=False)
    csr_size_uncompressed = os.path.getsize("/tmp/mat_csr.npz")

    # compute the size of csc matrix 
    np.savez_compressed("/tmp/mat_csc.npz", csc_matrix(mat)) 
    csc_size = os.path.getsize("/tmp/mat_csc.npz")

    np.savez_compressed("/tmp/mat_csc.npz", csc_matrix(mat), compressed=False)
    csc_size_uncompressed = os.path.getsize("/tmp/mat_csc.npz")

    # compute size of bsr matrix 
    np.savez_compressed("/tmp/mat_bsr.npz", bsr_matrix(mat)) 
    bsr_size = os.path.getsize("/tmp/mat_bsr.npz")

    np.savez_compressed("/tmp/mat_bsr.npz", bsr_matrix(mat), compressed=False)
    bsr_size_uncompressed = os.path.getsize("/tmp/mat_bsr.npz")


    # compute size of coo matrix
    np.savez_compressed("/tmp/mat_coo.npz", coo_matrix(mat))
    coo_size = os.path.getsize("/tmp/mat_coo.npz")

    np.savez_compressed("/tmp/mat_coo.npz", coo_matrix(mat), compressed=False)
    coo_size_uncompressed = os.path.getsize("/tmp/mat_coo.npz")

    # compute size of the original matrix
    # np.savez_compressed("/tmp/mat.npz", mat)
    # mat_size = os.path.getsize("/tmp/mat.npz")

    # return a dict of the sizes
    return {
        "goop_size": goop_size,
        "goop_size_uncompressed": goop_size_uncompressed,
        "csr_size": csr_size,
        "csr_size_uncompressed": csr_size_uncompressed,
        "csc_size": csc_size,
        "csc_size_uncompressed": csc_size_uncompressed,
        "coo_size": coo_size,
        "coo_size_uncompressed": coo_size_uncompressed,
        "bsr_size": bsr_size,
        "bsr_size_uncompressed": bsr_size_uncompressed,
        "theoretical_size": n_bits_theoretical / 8,
        "default_bits": default_bits / 8
    }
    

def create_output_path(n_row_clusters, 
                       n_col_clusters,  
                       matrix_path, 
                       output_dir): 
    
    """Create the output path for the experiment."""
    matrix_name = f"{Path(matrix_path).parent.stem}_{Path(matrix_path).stem}"
    output_path = output_dir / f"{matrix_name}_rc{n_row_clusters}_cc{n_col_clusters}_.json"
    
    return output_path



def main(args):
    output_dir = Path(args.output_dir)
    num_row_clusters = args.n_row_clusters
    num_col_clusters = args.n_col_clusters


    # Load the matrix
    logger.info("Loading the matrix.")
    #mat = np.array(load_npz(args.matrix_path).todense())
    mat = np.load(args.matrix_path)
    # compute the sizes of the matrices
    logger.info("Computing the sizes of the matrices.")
    sizes = compute_sizes(mat, num_row_clusters, num_col_clusters)
    logger.info(f" ..Done.")
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the output path
    output_path = create_output_path(args.n_row_clusters, 
                                     args.n_col_clusters, 
                                     args.matrix_path, 
                                        output_dir)

    if not args.debug: 
        # Save the sizes to the output path
        logger.info("Saving the sizes to the output path.")
        with open(output_path, "w") as f: 
            json.dump(sizes, f, indent=4)


if __name__ == "__main__": 
    parser = add_arguments()
    args = parser.parse_args()
    main(args)

    """
    single run script --- 

    python scripts/compute_sizes.py \
    --n_row_clusters 2 --n_col_clusters 2 --debug True \
    --matrix_path "../prepared/1k_hgmm_3p_LT_raw_feature_bc_matrix/1000_1000_0.npz" \
    --output_dir="/tmp/"
    


    slaunch scripts/hyperparam_sweep.py \
    --exp_name="csqz_test" --sweep n_row_clusters n_col_clusters \
    --slurm_profile=scavenger \
    --slurm_time="5:00:00" --slurm_mem=32G \
    --output_dir="/fs/clip-scratch/cqz/" \
    --n_row_clusters  2 5 --n_col_clusters  2 5 \
    --matrix_path outputs/whole.npz

    """
