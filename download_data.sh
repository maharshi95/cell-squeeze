mkdir -p data/

wget https://cf.10xgenomics.com/samples/cell-exp/7.0.0/1k_mouse_kidney_CNIK_3pv3/1k_mouse_kidney_CNIK_3pv3_raw_feature_bc_matrix.tar.gz -P data/
mkdir -p data/1k_mouse_kidney_CNIK_3pv3_raw_feature_bc_matrix

tar -C data/1k_mouse_kidney_CNIK_3pv3_raw_feature_bc_matrix -xzf data/1k_mouse_kidney_CNIK_3pv3_raw_feature_bc_matrix.tar.gz 

wget https://cf.10xgenomics.com/samples/cell-exp/6.0.0/1k_hgmm_3p_LT/1k_hgmm_3p_LT_raw_feature_bc_matrix.tar.gz -P data/
mkdir -p data/1k_hgmm_3p_LT_raw_feature_bc_matrix

tar -C data/1k_hgmm_3p_LT_raw_feature_bc_matrix -xzf data/1k_hgmm_3p_LT_raw_feature_bc_matrix.tar.gz 

wget https://cf.10xgenomics.com/samples/cell-exp/6.0.0/Brain_Tumor_3p/Brain_Tumor_3p_raw_feature_bc_matrix.tar.gz -P data/
mkdir -p data/Brain_Tumor_3p_raw_feature_bc_matrix

tar -C data/Brain_Tumor_3p_raw_feature_bc_matrix -xzf data/Brain_Tumor_3p_raw_feature_bc_matrix.tar.gz 