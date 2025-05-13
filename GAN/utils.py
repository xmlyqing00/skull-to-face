import torch
import numpy as np
from trimesh import transformations, primitives, util
from matplotlib.pyplot import get_cmap
from DECA.decalib.utils.config import get_cfg_defaults, update_cfg


def chamfer_dist(x, y, nx=None, ny=None, normal_filtering=False):
    """
    to reduce memory consumption, we first compute the matching that produces the minimal distance with no grad
    """

    def tensor_sqrt(data, epsilon=0.00001):
        return (data + torch.ones_like(data)*epsilon).sqrt()

    with torch.no_grad():
        dist_nograd = torch.norm(x[None,...] - y[:,None,...], dim=-1)

        dist2, idx2 = torch.min(dist_nograd, dim=1)  # 78
        idx2_ = torch.arange(0, y.shape[0]).to(idx2.device)

        if normal_filtering and nx is not None and ny is not None:
            cos_sim_nn1 = torch.sum((nx[idx1_, :]*ny[idx1,:]), dim=-1)
            filter_mask1 = (cos_sim_nn1 > 0.7)
            idx1 = idx1[filter_mask1]
            idx1_ = idx1_[filter_mask1]

            cos_sim_nn2 = torch.sum((nx[idx2, :]*ny[idx2_,:]), dim=-1)
            filter_mask2 = (cos_sim_nn2 > 0.7)
            idx2_ = idx2_[filter_mask2]
            idx2 = idx2[filter_mask2]

    dist2 = torch.norm(x[idx2,:] - y[idx2_,:], dim=-1)
    dist2 = dist2.mean()
    
    return dist2, idx2, idx2_


def build_spheres(pt: np.array, radius, color):
    
    if len(pt.shape) > 1:
        meshes = []
        for p in pt:
            meshes.append(build_spheres(p, radius, color))
        return util.concatenate(meshes)

    sphere = primitives.Sphere(radius=radius, center=pt)
    sphere.visual.vertex_colors = color

    return sphere


def build_cylinders(pt0: np.array, pt1: np.array, radius, color):
    
    if len(pt0.shape) > 1:
        meshes = []
        for p0, p1 in zip(pt0, pt1):
            meshes.append(build_cylinders(p0, p1, radius, color))
        return util.concatenate(meshes)

    h = np.linalg.norm(pt0 - pt1)
    stick = primitives.Cylinder(radius=radius, height=h, sections=6)
    stick.visual.vertex_colors = color

    normal = pt0 - pt1
    normal = normal / np.linalg.norm(normal)
    rot_axis = np.cross(stick.direction, normal)
    rot_angle = np.arccos(np.dot(stick.direction, normal))
    rot_mat = transformations.rotation_matrix(rot_angle, rot_axis, (0, 0, 0))
    trans_mat1 = transformations.translation_matrix((0, 0, h / 2))
    trans_mat2 = transformations.translation_matrix(pt1)
    transform_mat = trans_mat2 @ rot_mat @ trans_mat1
    stick.apply_transform(transform_mat)
    
    return stick


def interpolate(values, color_map=None, normalize='minmax', normalize_scale=1, dtype=np.uint8):
    """
    Given a 1D list of values, return interpolated colors
    for the range.

    Parameters
    ---------------
    values : (n, ) float
      Values to be interpolated over
    color_map : None, or str
      Key to a colormap contained in:
      matplotlib.pyplot.colormaps()
      e.g: 'viridis'

    Returns
    -------------
    interpolated : (n, 4) dtype
      Interpolated RGBA colors
    """

    # get a color interpolation function

    cmap = get_cmap(color_map)

    # make input always float
    values = np.asanyarray(values, dtype=np.float64).ravel()

    # scale values to 0.0 - 1.0 and get colors
    if normalize == 'minmax':
        values = (values - values.min()) / values.ptp()
        colors = cmap(values)
    elif normalize == '-1to1':
        values = np.clip(values * normalize_scale, -1, 1) / 2 + 0.5
        colors = cmap(values)
    elif normalize == '0to1':
        values = np.clip(values * normalize_scale, 0, 1)
        colors = cmap(values)
    else:
        raise NotImplementedError

    return colors


def normalize_codedict(codedict, keep_exp=False, keep_light=True, keep_pose=False):
    '''Normalize the settings of key: cam, pose, exp, light
    '''
    t = codedict['cam']
    # codedict['cam'] = t.new_tensor([[10.0, 0.0, 0.0]]).repeat(t.shape[0], 1)
    if not keep_pose:    
        codedict['pose'] = t.new_zeros(codedict['pose'].shape)
    if not keep_light:
        codedict['light']= t.new_tensor([[
            [ 3.5237,  3.5040,  3.4894],
            [ 0.2638,  0.2439,  0.2265],
            [ 0.0776,  0.0813,  0.0833],
            [-0.4833, -0.5552, -0.5953],
            [-0.1478, -0.1476, -0.1461],
            [-0.1409, -0.1509, -0.1604],
            [ 0.2000,  0.2031,  0.2025],
            [ 1.2140,  1.2144,  1.2098],
            [ 0.1632,  0.1330,  0.1217]
        ]]).repeat(t.shape[0], 1, 1)
    
    if not keep_exp:
        codedict['exp'] = t.new_zeros(codedict['exp'].shape)
        
    return codedict


def parse_args(parser, return_args=False):
    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    
    cfg.mode = args.mode
    cfg.force_reconstruct = args.force_reconstruct
    
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    if return_args: 
        return cfg, args 
    else: 
        return cfg
