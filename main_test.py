import numpy as np

from registration_utils import read_point_cloud_from_stl_file, align
from ssim_utils import pc_ssim

NUM_POINTS = 10000


pcA = read_point_cloud_from_stl_file(
    'train/tr_01.stl',
    num_points= NUM_POINTS,
    outlier_removal= False,
)

pcB = read_point_cloud_from_stl_file(
    'train/tr_02.stl',
    num_points= NUM_POINTS,
    outlier_removal= False,
)

pcB = align(pcB, pcA)

pcA.estimate_normals()
pcB.estimate_normals()

score = pc_ssim(
    pcA, 
    pcB,
    neighborhood_size=12,
    feature='curvature',
    ref=0,
    estimators=['std', 'var', 'mean_ad', 'median_ad', 'coef_var', 'qcd'],
    pooling_methods=['mean', 'mse', 'rms'],
    const=np.finfo(float).eps
)



