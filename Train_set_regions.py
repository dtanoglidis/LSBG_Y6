# Code that summarizes and returns the training set for UDG galaxy finding
# Returns the matrix of features and an array with inputs 1=UDG, 0=non-UDG

import numpy as np
import scipy
from astropy.io import fits
#from astropy import units as u
#from astropy.coordinates import SkyCoord
# =======================================================================# 
#========================================================================#
# Open the regions

REG_1 = fits.open('Region_1.fits')
REG_2 = fits.open('Region_2.fits')
REG_3 = fits.open('Region_3.fits')
REG_4 = fits.open('Region_4.fits')
REG_5 = fits.open('Region_5.fits')
REG_6 = fits.open('Region_6.fits')
REG_7 = fits.open('Region_7.fits')


# ======================================================================
# ======================================================================

def feat_matrix_return(i):
	""" Retrurns the matrix of features from region i"""


	if (i==1):
		REG = REG_1
	elif (i==2):
		REG = REG_2
	elif (i==3):
		REG = REG_3
	elif (i==4):
		REG = REG_4
	elif (i==5):
		REG = REG_5
	elif (i==6):
		REG = REG_6
	else:
		REG = REG_7
   

	# ==================================================================
	# First import the data 

	# ==================================================================
	# spread_model_i and its error
	spread_model_i = REG[1].data['spread_model_i']
	speraderr_model_i = REG[1].data['spreaderr_model_i']

	# RA and DEC 
	RA = REG[1].data['ra']
	DEC = REG[1].data['dec']

	# Image A/ Image B
	IMAGE_A = REG[1].data['a_image']
	IMAGE_B = REG[1].data['b_image']

	# Magnitudes 
	MAG_AUTO_G = REG[1].data['mag_auto_g']
	MAG_AUTO_R = REG[1].data['mag_auto_r']
	MAG_AUTO_I = REG[1].data['mag_auto_i']

	# Flux radii - convert from pixels to arcseconds
	FLUX_RADIUS_G = 0.263*REG[1].data['flux_radius_g']
	FLUX_RADIUS_R = 0.263*REG[1].data['flux_radius_r']
	FLUX_RADIUS_I = 0.263*REG[1].data['flux_radius_i'] 
	# ================================================================================
	# SURFACE BRIGHTNESSES

	##Effective model surface brightness (SB) above background [mag/sq. arcmin]. 
	#SB at the isophote which includes half of the flux from the model, above background
	MU_EFF_G = REG[1].data['mu_eff_model_g']
	MU_EFF_R = REG[1].data['mu_eff_model_r']
	MU_EFF_I = REG[1].data['mu_eff_model_i']

	#Peak surface brightness above background [mag/asec^2]
	MU_MAX_G = REG[1].data['mu_max_g']
	MU_MAX_R = REG[1].data['mu_max_r']
	MU_MAX_I = REG[1].data['mu_max_i']

	#Peak surface brightness above background  - using Model[mag/asec^2]
	MU_MAX_MODEL_G = REG[1].data['mu_max_model_g']
	MU_MAX_MODEL_R = REG[1].data['mu_max_model_r']
	MU_MAX_MODEL_I = REG[1].data['mu_max_model_i']

	#Mean surface brightness using the whole area inside the isophote used for MU_EFF_MODEL
	MU_MEAN_G = REG[1].data['mu_mean_model_g']
	MU_MEAN_R = REG[1].data['mu_mean_model_r']
	MU_MEAN_I = REG[1].data['mu_mean_model_i']


	# Define the ellipticity now
	ellipticity = 1.0 - IMAGE_B/IMAGE_A


	# =============================================================================
	# =============================================================================
	# Define the cuts here 
	# =============================================================================

	# Star-galaxy separation 
	star_gal_cut = ((spread_model_i+(5.0/3.0)*speraderr_model_i)>0.007)

	# Surface brightness cut
	mu_mean_cut = (MU_MEAN_G > 24.3)&(MU_MEAN_G < 28.0)

	# Color cuts 
	col_1 = ((MAG_AUTO_G - MAG_AUTO_I)>(-0.1))
	col_2 = ((MAG_AUTO_G - MAG_AUTO_I)<(1.4))
	col_3 = ((MAG_AUTO_G - MAG_AUTO_R) > 0.7*(MAG_AUTO_G - MAG_AUTO_I) - 0.4)
	col_4 = ((MAG_AUTO_G - MAG_AUTO_R) < 0.7*(MAG_AUTO_G - MAG_AUTO_I) + 0.4)
	color_cuts = col_1&col_2&col_3&col_4

	# Radius cut
	radius_cut = ((FLUX_RADIUS_G > 2.5)&(FLUX_RADIUS_G < 20.0))

	# Ellipticity cut
	ell_cut = (ellipticity < 0.7)

	# Summarize the final cut

	Total_cut = star_gal_cut&mu_mean_cut&color_cuts&radius_cut&ell_cut

	# ================================================================================
	# ================================================================================

	# RA and DEC
	RA = RA[Total_cut]
	DEC = DEC[Total_cut]

	# Image A/ Image B
	IMAGE_A = IMAGE_A[Total_cut]
	IMAGE_B = IMAGE_B[Total_cut]
	
	#Magnitudes 
	MAG_AUTO_G = MAG_AUTO_G[Total_cut]
	MAG_AUTO_R = MAG_AUTO_R[Total_cut]
	MAG_AUTO_I = MAG_AUTO_I[Total_cut]

	#Flux radii 
	FLUX_RADIUS_G = FLUX_RADIUS_G[Total_cut]
	FLUX_RADIUS_R = FLUX_RADIUS_R[Total_cut]
	FLUX_RADIUS_I = FLUX_RADIUS_I[Total_cut]

	# ================================================================================
	# ================================================================================
	# SURFACE BRIGHTNESSES

	##Effective model surface brightness (SB) above background [mag/sq. arcmin]. 
	#SB at the isophote which includes half of the flux from the model, above background
	MU_EFF_G = MU_EFF_G[Total_cut]
	MU_EFF_R = MU_EFF_R[Total_cut]
	MU_EFF_I = MU_EFF_I[Total_cut]

	#Peak surface brightness above background [mag/asec^2]
	MU_MAX_G = MU_MAX_G[Total_cut]
	MU_MAX_R = MU_MAX_R[Total_cut]
	MU_MAX_I = MU_MAX_I[Total_cut]

	#Peak surface brightness above background  - using Model[mag/asec^2]
	MU_MAX_MODEL_G = MU_MAX_MODEL_G[Total_cut]
	MU_MAX_MODEL_R = MU_MAX_MODEL_R[Total_cut]
	MU_MAX_MODEL_I = MU_MAX_MODEL_I[Total_cut]

	#Mean surface brightness using the whole area inside the isophote used for MU_EFF_MODEL
	MU_MEAN_G = MU_MEAN_G[Total_cut]
	MU_MEAN_R = MU_MEAN_R[Total_cut]
	MU_MEAN_I = MU_MEAN_I[Total_cut]

	# Ellipticity
	ellipticity = ellipticity[Total_cut]

	# ===========================================================================================
	# Define the colors
	col_g_i = MAG_AUTO_G - MAG_AUTO_I
	col_g_r = MAG_AUTO_G - MAG_AUTO_R
	col_r_i = MAG_AUTO_R - MAG_AUTO_I

	# =================================================================================
	# =================================================================================
	# =================================================================================
	# =================================================================================
	# Define the train data matrix of features 

	X_feat_train = np.zeros([len(RA),19])

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
	X_feat_train[:,13] = MU_EFF_G
	X_feat_train[:,14] = MU_EFF_R
	X_feat_train[:,15] = MU_EFF_I
	# Mean surface brightness - this is actually the "MU_MEAN_MODEL"
	X_feat_train[:,16] = MU_MEAN_G
	X_feat_train[:,17] = MU_MEAN_R
	X_feat_train[:,18] = MU_MEAN_I

	leng = len(RA)
    
	return RA, DEC, leng, X_feat_train

# ===================================================================================================
# ===================================================================================================
RA_1, DEC_1, len_1, X_feat_1 = feat_matrix_return(1)
RA_2, DEC_2, len_2, X_feat_2 = feat_matrix_return(2)
RA_3, DEC_3, len_3, X_feat_3 = feat_matrix_return(3)
RA_4, DEC_4, len_4, X_feat_4 = feat_matrix_return(4)
RA_5, DEC_5, len_5, X_feat_5 = feat_matrix_return(5)
RA_6, DEC_6, len_6, X_feat_6 = feat_matrix_return(6)
RA_7, DEC_7, len_7, X_feat_7 = feat_matrix_return(7)
# ===================================================================================================
# ===================================================================================================

# Define the target classes now. 
# 0 = non - LSB objects
# 1 = UDGs, 2 = Spirals, 3 = other LSB objects


# ===================================================================================================
# First region
y_1 = np.zeros(len_1)

ones_1 = [79,125,220,303,306,583,603,847,982,993,1028]
twos_1 = [219,244,378,1098]
threes_1 = [49,171,227,268,449,468,487,492,510,540,619,637,639,641,705,712,738,812,886,945,959,1114]

y_1[ones_1] = 1
y_1[twos_1] = 2
y_1[threes_1] = 3
# ===================================================================================================
# Second region
y_2 = np.zeros(len_2)

ones_2 = [1,12,19,25,29,43,44,45,46,54,68,69,80,98,108,125,141,151,170,189,200,222,233,247,250,263,264,291,303,
318,328,330,345,348,352,354,355,363,377,379,403,423,425,435,446,466,476,486,513,514,522,526,530,547,549,556,561,
575,577,611,626,635,656,659,666,675,677,680,685,690,692,714,725,726,739,756,765,768,783,794,824,850,869,882,899,
900,920,946,949,950,954,967,968,972,994,1012,1014,1023,1034,1038,1044,1071,1080,1098,1111,1115,1128,1130,1154,1163,
1170,1176,1188,1189,1191,1197,1201,1207,1220,1230,1231,1235,1243,1255,1271,1280,1286,1302,1312,1316,1322,1329,1330,
1344,1377,1393,1404,1406,1414,1419,1429,1447,1463,1477,1498,1543,1546,1550,1554,1588,1592,1616,1620,1639,1646,1659,
1665,1666,1672,1673,1681,1684,1696,1706,1717,1722,1735,1737,1749,1757,1762,1777,1781,1783,1796,1799,1817,1827,1832,
1833,1863,1869,1878,1887]
twos_2 = [380,408,816,840,1412]
threes_2 = [13,22,50,86,123,229,232,300,315,340,398,436,551,610,622,650,690,694,802,814,819,826,831,888,895,896,
903,904,951,959,961,976,988,1023,1027,1063,1165,1274,1275,1282,1327,1343,1375,1420,1549,1595,1658,1690,1697,1741,1753,1846]

y_2[ones_2] = 1
y_2[twos_2] = 2
y_2[threes_2] = 3

# ===================================================================================================
# Third region

y_3 = np.zeros(len_3)

ones_3 = [10,170,229,431,509,584,748,758,786]
twos_3 = [44,117,211,230,341,408,529,751,757]
threes_3 = [8,40,134,136,193,287,300,317,323,356,399,423,455,460,477,548,559,565,597,609,625,707,750,754,756,767,781,788,792]

y_3[ones_3] = 1
y_3[twos_3] = 2
y_3[threes_3] = 3

# ===================================================================================================
# Fourth region

y_4 = np.zeros(len_4)

ones_4 = [0,36,70,76,88,90,143,152,173,176,180,186,188,210,221,229,239,248,257,261,268,292,329,342,343,348,352,
359,383,385,402,411,412,455,514,519,528,544,597,604,606,613,614,620,634,636,657,675,677,678,723,742,744,752,758,
760,765,773,784,787,790,819,846,850,853,881,886,894,899,919,949,956,959,969,1005,1023,1034,1049,1051,1056,1071,
1082,1083,1087,1093,1095,1098,1100,1005,1106,1109,1113,1115,1119,1133,1142,1149,1160,1169,1172,1179,1180,1210,
1218,1218,1222,1233,1237,1241,1246,1247,1253,1261,1277,1284,1346,1354,1363,1381]
twos_4 = [39,53,121,208,313,328,336,338,550,563,629,803,839,1193,1229,1323]
threes_4 = [8,24,29,57,74,113,118,119,131,139,147,158,166,170,190,193,200,205,219,227,244,264,270,283,288,295,306,318,347,355,
382,400,406,410,413,417,424,430,431,448,446,477,499,512,517,520,523,538,553,557,559,586,625,645,651,668,676,698,704,721,
735,766,768,770,788,793,810,820,822,824,895,896,902,904,928,988,1000,1009,1018,1033,1042,1047,1074,1078,1092,1094,1127,1144,
1165,1176,1184,1198,1213,1231,1235,1249,1258,1265,1271,1290,1291,1292,1293,1298,1314,1329,1352,1355,1357,1382]

y_4[ones_4] = 1
y_4[twos_4] = 2
y_4[threes_4] = 3

# ===================================================================================================
# Fifth region


y_5 = np.zeros(len_5)

ones_5 = [6,7,65,78,86,102,191,194,223,254,281,282,434,541,633,696,767]
twos_5 = [56,385,577,604]
threes_5 = [2,91,201,273,432,451,533,540,549,616,617,677,681,705,737,759]

y_5[ones_5] = 1
y_5[twos_5] = 2
y_5[threes_5] = 3

# ===================================================================================================
# Sixth region

y_6 = np.zeros(len_6)

ones_6 = [105,222,252,306,329,519,539,590,594,639,665,691,830,890,900,925,1000]
twos_6 = [547,571,827,1053]
threes_6 = [33,52,63,103,153,162,246,258,274,284,315,358,372,384,391,476,499,556,635,650,700,812,866,904,932,933,937,1018,1042,1063,1073]

y_6[ones_6] = 1
y_6[twos_6] = 2
y_6[threes_6] = 3

# ===================================================================================================
# Seventh region

y_7 = np.zeros(len_7)

ones_7 = [23,55,60,124,145,190,314,344,548,555,586,603,605,646]
twos_7 = [180,306,599]
threes_7 = [28,53,115,154,208,231,333,391,412,476,519,541,615]

y_7[ones_7] = 1
y_7[twos_7] = 2
y_7[threes_7] = 3

# ===================================================================================================
# ===================================================================================================
y_tot = np.concatenate((y_1,y_2,y_3,y_4,y_5,y_6,y_7))
X_feat_tot = np.concatenate((X_feat_1,X_feat_2,X_feat_3,X_feat_4,X_feat_5,X_feat_6,X_feat_7), axis=0)


RA_tot = np.concatenate((RA_1,RA_2,RA_3,RA_4,RA_5,RA_6,RA_7))
DEC_tot = np.concatenate((DEC_1,DEC_2,DEC_3,DEC_4,DEC_5,DEC_6,DEC_7))
def supervised():
	return y_tot, X_feat_tot

def coords():
    return RA_tot, DEC_tot








