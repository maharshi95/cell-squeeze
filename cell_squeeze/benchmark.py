# %%
import numpy as np
from sklearn.cluster import SpectralBiclustering
from base import BiClusterMatrixPermuter

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


# %%
data_path = "prepared/1k_hgmm_3p_LT_raw_feature_bc_matrix/1000_1000_0.npz"

# Open the npz file
data_mat = np.load(data_path)
# %%
