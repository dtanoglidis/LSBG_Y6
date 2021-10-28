# Code that reads Y3 LSBGs and negatives 
# And returns feature matrix for classification

import numpy as np
import scipy
from astropy.io import fits
import pandas as pd


# Load catalogs as dataframes
LSBG_Y3_df = pd.read_csv('random_LSBGs_all.csv') # LSBGs
Artifacts_2_df = pd.read_csv('random_negative_all_2.csv') #"Hard" artifacts

# Each one has shape (20000x24); concatenate them

Full_df = pd.concat([LSBG_Y3_df,Artifacts_2_df])


# read features - only those that we need 
coadd_id = Full_df['coadd_id'].values
ra = Full_df['ra'].values
dec = Full_df['dec'].values

# Define ellipticity 
ellipticity = 1 - Full_df['B_IMAGE'].values/Full_df['A_IMAGE'].values

# Define colors
col_g_i = Full_df['mag_auto_g'].values - Full_df['mag_auto_i'].values
col_g_r = Full_df['mag_auto_g'].values - Full_df['mag_auto_r'].values
col_r_i = Full_df['mag_auto_r'].values - Full_df['mag_auto_i'].values

# ================================================================
# ================================================================
# Define feature matrix
# Initialize 
X_feat_train = np.zeros([len(ra),19])


# Populate the matrix
X_feat_train[:,0] = ellipticity # Ellipticity
# Colors
X_feat_train[:,1] = col_g_i
X_feat_train[:,2] = col_g_r
X_feat_train[:,3] = col_r_i  
# Magnitudes
X_feat_train[:,4] = Full_df['mag_auto_g'].values
X_feat_train[:,5] = Full_df['mag_auto_r'].values
X_feat_train[:,6] = Full_df['mag_auto_i'].values
# Flux radii
X_feat_train[:,7] = Full_df['flux_radius_g'].values
X_feat_train[:,8] = Full_df['flux_radius_r'].values
X_feat_train[:,9] = Full_df['flux_radius_i'].values
# Peak (max) surface brightness
X_feat_train[:,10] = Full_df['mu_max_model_g'].values
X_feat_train[:,11] = Full_df['mu_max_model_r'].values
X_feat_train[:,12] = Full_df['mu_max_model_i'].values
# Effective surface brightness 
X_feat_train[:,13] =  Full_df['mu_eff_model_g'].values
X_feat_train[:,14] =  Full_df['mu_eff_model_r'].values
X_feat_train[:,15] =  Full_df['mu_eff_model_i'].values
# Mean surface brightness 
X_feat_train[:,16] = Full_df['mu_mean_model_g'].values
X_feat_train[:,17] = Full_df['mu_mean_model_r'].values
X_feat_train[:,18] = Full_df['mu_mean_model_i'].values

# ===================================================
# ===================================================
y_lsbg = np.ones(20000)
y_art = np.zeros(20000)

y_label = np.concatenate((y_lsbg,y_art))



# Define functions that return the feature matrix, the labels
# and the coordinates of the Y3 training set
def Feature_Y3():
	return y_label, X_feat_train

def coords_Y3():
    return ra, dec

















