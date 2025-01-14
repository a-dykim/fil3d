"""
GALFA constants

steps and ranges and a fake header dict

LL2017
"""

GALFA_Y_STEPS = 2432
GALFA_X_STEPS = 21600

GALFA_SELECT_V_SLICES_RANGE = [955, 1092]

GALFA_SLICE_COMMON_NAME = 'GALFA_HI_W_S'

GALFA_W_SLICE_SEPARATION = 0.73612284

MOCK_GALFA_HDR = {
    'CRPIX1': 10800.5,
    'CDELT1': -0.0166667,
    'CRVAL1': 180.000,
    'CRPIX2': 256.500,
    'CDELT2': 0.0166667,
    'CRVAL2': 2.35000
}

GALFA_BACKUP_DATA_DIR = '/Volumes/LarryExternal1/Research_2017/GALFA_slices_backup/umask_gaussian_30/'
