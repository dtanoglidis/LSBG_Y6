# Code that reads and returns a feature matrix for the SMUDGES 
# UDGs (all of them are true LSBGs, so there is no need)

import numpy as np
import scipy
from astropy.io import fits


# Import SMUDGES catalog
SMUDGES = fits.open('y6_gold_2_0_smudges_skim.fits')


# Load the different properties
coadd_id = SMUDGES[1].data['COADD_OBJECT_ID']
ra = SMUDGES[1].data['RA']
dec = SMUDGES[1].data['DEC']


A_IMAGE = SMUDGES[1].data['A_IMAGE']
B_IMAGE = SMUDGES[1].data['B_IMAGE']
MAG_AUTO_G = SMUDGES[1].data['MAG_AUTO_G']
FLUX_RADIUS_G = 0.263*SMUDGES[1].data['FLUX_RADIUS_G']
MU_EFF_MODEL_G = SMUDGES[1].data['MU_EFF_MODEL_G']
MU_MAX_G = SMUDGES[1].data['MU_MAX_G']
MU_MAX_MODEL_G = SMUDGES[1].data['MU_MAX_MODEL_G']
MU_MEAN_MODEL_G = SMUDGES[1].data['MU_MEAN_MODEL_G']
MAG_AUTO_R = SMUDGES[1].data['MAG_AUTO_R']
FLUX_RADIUS_R = 0.263*SMUDGES[1].data['FLUX_RADIUS_R']
MU_EFF_MODEL_R = SMUDGES[1].data['MU_EFF_MODEL_R']
MU_MAX_R = SMUDGES[1].data['MU_MAX_R']
MU_MAX_MODEL_R = SMUDGES[1].data['MU_MAX_MODEL_R']
MU_MEAN_MODEL_R = SMUDGES[1].data['MU_MEAN_MODEL_R']
MAG_AUTO_I = SMUDGES[1].data['MAG_AUTO_I']
FLUX_RADIUS_I = 0.263*SMUDGES[1].data['FLUX_RADIUS_I']
MU_EFF_MODEL_I = SMUDGES[1].data['MU_EFF_MODEL_I']
MU_MAX_I = SMUDGES[1].data['MU_MAX_I']
MU_MAX_MODEL_I = SMUDGES[1].data['MU_MAX_MODEL_I']
MU_MEAN_MODEL_I = SMUDGES[1].data['MU_MEAN_MODEL_I']

# ====================================================
# Define derivative quantities

# Ellipticity
ellipticity = 1.0 - A_IMAGE/B_IMAGE

# Define the colors
col_g_i = MAG_AUTO_G - MAG_AUTO_I
col_g_r = MAG_AUTO_G - MAG_AUTO_R
col_r_i = MAG_AUTO_R - MAG_AUTO_I

# ===================================================
# ===================================================
# ===================================================
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
X_feat_train[:,4] = MAG_AUTO_G
X_feat_train[:,5] = MAG_AUTO_R
X_feat_train[:,6] = MAG_AUTO_I
# Flux radii
X_feat_train[:,7] = FLUX_RADIUS_G
X_feat_train[:,8] = FLUX_RADIUS_R
X_feat_train[:,9] = FLUX_RADIUS_I
# Peak (max) surface brightness
X_feat_train[:,10] = MU_MAX_MODEL_G
X_feat_train[:,11] = MU_MAX_MODEL_R
X_feat_train[:,12] = MU_MAX_MODEL_I
# Effective surface brightness - this is actually the "MU_EFF_MODEL" 
X_feat_train[:,13] = MU_EFF_MODEL_G
X_feat_train[:,14] = MU_EFF_MODEL_R
X_feat_train[:,15] = MU_EFF_MODEL_I
# Mean surface brightness - this is actually the "MU_MEAN_MODEL"
X_feat_train[:,16] = MU_MEAN_MODEL_G
X_feat_train[:,17] = MU_MEAN_MODEL_R
X_feat_train[:,18] = MU_MEAN_MODEL_I


# Define functions that return the feature matrix
# and the coordinates of the SMUDGES 
def Feature_SMUDGES():
	return X_feat_train

def coords_SMUDGES():
    return ra, dec















