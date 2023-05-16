"""
Script that performs a range of experiments on sqeezing 
a count matrix into a sparse goop.

"""
import argparse 
import os 
from pathlib import Path
import numpy as np 
from scipy.sparse import load_npz, csr_matrix, csc_matrix, coo_matrix, bsr_matrix
from cell_squeeze.benchmark import BiClusterMatrixPermuter, make_goop_dict, flatten_array_dict
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

    return parser 


def compute_sizes(mat, n_row_clusters, n_col_clusters): 
    """Compute the sizes of the row and column clusters."""
    permuter = BiClusterMatrixPermuter(n_row_clusters, n_col_clusters)
    
    
    logger.info("Biclustering and computing goop dict.")
    goop_dict = make_goop_dict(mat, permuter)
    goop_dict_flat = flatten_array_dict(goop_dict)

    # compute size of goop 
    np.savez_compressed("/tmp/goop.npz", **goop_dict_flat, compressed=True)
    goop_size = os.path.getsize("/tmp/goop.npz")
    

    logger.info("Computing sizes of other formats of sparse matrices.")
    # compute size of csr matrix 
    np.savez_compressed("/tmp/mat_csr.npz", csr_matrix(mat))
    csr_size = os.path.getsize("/tmp/mat_csr.npz")

    # compute the size of csc matrix 
    np.savez_compressed("/tmp/mat_csc.npz", csc_matrix(mat)) 
    csc_size = os.path.getsize("/tmp/mat_csc.npz")
    
    # compute size of bsr matrix 
    np.savez_compressed("/tmp/mat_bsr.npz", bsr_matrix(mat)) 
    bsr_size = os.path.getsize("/tmp/mat_bsr.npz")

    # compute size of coo matrix
    np.savez_compressed("/tmp/mat_coo.npz", coo_matrix(mat))
    coo_size = os.path.getsize("/tmp/mat_coo.npz")

    # compute size of the original matrix
    # np.savez_compressed("/tmp/mat.npz", mat)
    # mat_size = os.path.getsize("/tmp/mat.npz")

    # return a dict of the sizes
    return {
        "goop_size": goop_size,
        "csr_size": csr_size,
        "csc_size": csc_size,
        "coo_size": coo_size,
        "bsr_size": bsr_size
    }
    


def create_output_path(n_row_clusters, 
                       n_col_clusters,  
                       matrix_path, 
                       output_dir): 
    
    """Create the output path for the experiment."""
    matrix_name = f"{Path(matrix_path).parent.stem}_{Path(matrix_path).stem}"
    output_path = output_dir / f"{matrix_name}_rc{n_row_clusters}_cc{n_col_clusters}.json"
    
    return output_path



def main(args):
    output_dir = Path(args.output_dir)
    num_row_clusters = args.n_row_clusters
    num_col_clusters = args.n_col_clusters


    # Load the matrix
    logger.info("Loading the matrix.")
    mat = np.array(load_npz(args.matrix_path).todense())

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

    # Save the sizes to the output path
    logger.info("Saving the sizes to the output path.")
    with open(output_path, "w") as f: 
        json.dump(sizes, f, indent=4)


if __name__ == "__main__": 
    parser = add_arguments()
    args = parser.parse_args()
    main(args)

    """
    slaunch scripts/hyperparam_sweep.py \
    --exp_name="csqz_test" --sweep matrix_path \
    --slurm_profile=scavenger \
    --slurm_time="10:00:00" --slurm_mem=24G \
    --output_dir="compression_results" \
    --n_row_clusters 2 --n_col_clusters 2 \
    --matrix_path "../prepared/1k_hgmm_3p_LT_raw_feature_bc_matrix/10000_10000_0.npz" "../prepared/1k_mouse_kidney_CNIK_3pv3_raw_feature_bc_matrix/10000_10000_2.npz"  
    
    """