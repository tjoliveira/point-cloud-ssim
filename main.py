import argparse
import numpy as np

from registration_utils import read_point_cloud_from_ply_file, align
from ssim_utils import pc_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--pcA', help='Filepath to stl file for point cloud A.')
parser.add_argument('--pcB', help='Filepath to stl file for point cloud B.')
parser.add_argument(
    '--neighborhood_size', 
    type= int, 
    default=12, 
    help='Number nearest neighbors.'
)
parser.add_argument(
    '--num_points', 
    type= int, 
    default=10000, 
    help='Number of points for the point clouds.'
)
parser.add_argument(
    '--feature',  
    default='geometry',
    help='Type of feature to use in the computation of the feature map.'
)
parser.add_argument(
    '--ref', 
    type=int, 
    default=0,
    help='Symmetric (0) or asymmetric structural similarity. 1 to use pcA as reference. 2 to use pcB as reference.'
)
parser.add_argument(
    '--estimators', 
    nargs="+", 
    default=['std', 'var', 'mean_ad', 'median_ad', 'coef_var', 'qcd'],
    help ='Estimators used to extract features'
 )
parser.add_argument(
    '--pooling_methods', 
    nargs="+", 
    default=['mean', 'mse', 'rms'],
    help ='Pooling methods used to aggregate the feature maps'
 )

args = parser.parse_args()

print('Number of points: {}'.format(args.num_points))
print('Neighborhood size: {}'.format(args.neighborhood_size))
print('Feature: {}'.format(args.feature))
print('Reference: {}'.format(args.ref))
print('Estimators: {}'.format(args.estimators))
print('Pooling methods: {}'.format(args.pooling_methods))

pcA = read_point_cloud_from_ply_file(
    args.pcA,
    num_points= args.num_points,
    outlier_removal= False,
)

pcB = read_point_cloud_from_ply_file(
    args.pcB,
    num_points= args.num_points,
    outlier_removal= False,
)

pcA.estimate_normals()
pcB.estimate_normals()

pcB = align(pcB, pcA)



score = pc_ssim(
    pcA,
    pcB,
    neighborhood_size= args.neighborhood_size,
    feature=args.feature,
    ref=args.ref,
    estimators=args.estimators,
    pooling_methods=args.pooling_methods,
    const=np.finfo(float).eps
)

print('Structural similarity scores (ESTIMATORSxPOOLING):')
print(score)
