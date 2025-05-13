import cv2
import numpy as np
import trimesh
import torch
import pickle
from pathlib import Path
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from pysdf import SDF

from GAN.utils import normalize_codedict, build_spheres
from DECA.decalib.datasets import detectors
from DECA.decalib.utils import util


def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center


def load_single_image(image_path, crop_flag=True):

    face_detector = detectors.FAN()
    scale = 1.25
    crop_size = 224
    resolution_inp = crop_size
    
    image_name = image_path.stem
    image = np.array(imread(image_path))
    h, w, _ = image.shape

    if crop_flag:
        bbox, bbox_type = face_detector.run(image)
        if len(bbox) < 4:
            print('no face detected! run original image')
            left = 0; right = h-1; top=0; bottom=w-1
        else:
            left = bbox[0]; right=bbox[2]
            top = bbox[1]; bottom=bbox[3]
    else:
        scale = 1
        bbox_type = 'kpt68'
        left = 0; right = h-1; top=0; bottom=w-1
    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
    size = int(old_size * scale)
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

    DST_PTS = np.array([[0,0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    
    image = image/255.

    dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
    dst_image = dst_image.transpose(2,0,1)
    return {
        'image': torch.tensor(dst_image).float(),
        'image_name': image_name,
        'tform': torch.tensor(tform.params).float(),
        'original_image': torch.tensor(image.transpose(2,0,1)).float(),
    }


def build_3dmm_data(deca, deca_dir_path: str, face_img_path: str, dense_face_fids_path: str, device: str, 
                    neutral_pose: bool = True, neutral_face: bool = False):
    img_dict = load_single_image(face_img_path, crop_flag=True)
    img = img_dict['image'].unsqueeze(0).to(device)

    codedict = deca.encode(img)
    # opdict, visdict = deca.decode(codedict)

    tform = img_dict['tform'][None, ...]
    tform = torch.inverse(tform).transpose(1,2).to(device)
    original_image = img_dict['original_image'][None, ...].to(device)
    orig_opdict, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform, neutral_pose=neutral_pose, neutral_face=neutral_face)    
    orig_visdict['inputs'] = original_image

    deca.save_obj(str(deca_dir_path / 'face.obj'), orig_opdict)

    cv2.imwrite(str(deca_dir_path / 'vis_original_size.jpg'), deca.visualize(orig_visdict))

    for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
        if vis_name not in orig_visdict.keys():
            continue
        cv2.imwrite(str(deca_dir_path / f'{vis_name}_orig.jpg'), util.tensor2image(orig_visdict[vis_name][0]))

    dense_mesh_path = str(deca_dir_path / 'face_neutralpose_detail.obj')
    dense_mesh = trimesh.load(
        dense_mesh_path, 
        process=False, 
        maintain_order=True, 
    )
    dense_face_fids = np.load(dense_face_fids_path)
    dense_mesh.update_faces(dense_face_fids)
    dense_mesh.export(dense_mesh_path.replace('.obj', '_face.ply'))

    # Save data
    torch.save(codedict, deca_dir_path / 'codedict.pt')

    # Extract landmark
    dense_mesh_orig = trimesh.load(
        str(deca_dir_path / 'face_detail.obj'), 
        process=False, 
        maintain_order=True, 
    )
    dense_mesh_orig.update_faces(dense_face_fids)

    f = SDF(dense_mesh_orig.vertices, dense_mesh_orig.faces)
    landmarks3d_world = orig_opdict['landmarks3d_world'][0].detach().cpu().numpy()
    nearest_vids = f.nn(landmarks3d_world)
    nearest_v = dense_mesh.vertices[nearest_vids]
    np.savetxt(
        deca_dir_path / 'kpts_dense_face.txt', 
        np.concatenate([nearest_vids[:, np.newaxis], nearest_v], axis=-1),
    )
    landmarks3d_world_mesh = build_spheres(nearest_v, 0.002, (117, 250, 97))
    landmarks3d_world_mesh.export(deca_dir_path / 'landmarks3d_world.ply')

    face_mesh_orig = trimesh.load(
        str(deca_dir_path / 'face.obj'), 
        process=False, 
        maintain_order=True, 
    )
    f = SDF(face_mesh_orig.vertices, face_mesh_orig.faces)
    nearest_vids = f.nn(dense_mesh_orig.vertices[nearest_vids])
    nearest_v = face_mesh_orig.vertices[nearest_vids]
    np.savetxt(
        deca_dir_path / 'kpts_flame.txt', 
        np.concatenate([nearest_vids[:, np.newaxis], nearest_v], axis=-1),
    )
    landmarks3d_flame_mesh = build_spheres(nearest_v, 0.002, (117, 250, 97))
    landmarks3d_flame_mesh.export(deca_dir_path / 'landmarks3d_flame.ply')


def get_singlefit_data(
        skull_name: str, 
        face_name: str, 
        cfg_dataset: dict, 
        deca, 
        device: str, 
        force_reconstruct: bool = False
    ):

    deca_dir_path = Path(cfg_dataset.face_deca_dir) / Path(face_name).stem
    codedict_path = deca_dir_path / 'codedict.pt'
    neutral_face = 'neutral' in face_name
    if force_reconstruct or not codedict_path.exists() :
        deca_dir_path.mkdir(parents=True, exist_ok=True)

        face_img_path = Path(cfg_dataset.face_img_dir) / face_name
        build_3dmm_data(deca, deca_dir_path, face_img_path, device, neutral_pose=True, neutral_face=neutral_face)

    codedict = torch.load(codedict_path, map_location='cpu')
    codedict_norm = normalize_codedict(
        codedict, 
        keep_exp=cfg_dataset.keep_exp, 
        keep_pose=cfg_dataset.keep_pose 
    )
    if 'neutral' in face_name:
        codedict_norm['shape'] = codedict_norm['shape'] * 0.0
        
    detaildict = np.load(deca_dir_path / 'face_detail.npy', allow_pickle=True)
    detaildict = detaildict.item()

    # Load skull info
    skull_dir = Path(cfg_dataset.skull_dir)
    print((skull_dir / skull_name).with_suffix('.ply'))
    skull_mesh = trimesh.load_mesh(
        (skull_dir / skull_name).with_suffix('.ply'),
        maintain_order=True, 
        skip_materials=True, 
        process=False
    )

    skin_mesh = trimesh.load_mesh(
        (skull_dir / skull_name.replace('Skull', 'Skin')).with_suffix('.ply'),
        maintain_order=True, 
        skip_materials=True, 
        process=False
    )
    
    lmks_dict = np.load(cfg_dataset.lmks_dict_path, allow_pickle=True).item()
    skull_idx = lmks_dict['skull_names'].index(f'{skull_name}.ply')

    if cfg_dataset.use_lmk_gt:
        print('Load GT depth')
        lmk_depth = lmks_dict['depth'][skull_idx].reshape(-1)
    else:
        print('Load depth samples', cfg_dataset.lmk_depth_samples_path, 'Use index', cfg_dataset.lmk_depth_sample_idx)
        lmk_depth_samples = np.load(cfg_dataset.lmk_depth_samples_path)
        lmk_depth = lmk_depth_samples[cfg_dataset.lmk_depth_sample_idx]
    
    lmk_on_skull = lmks_dict['on_skull'][skull_idx]
    lmk_target = lmk_on_skull + lmks_dict['direction'][skull_idx] * lmk_depth[:, np.newaxis]
    
    skin_size = np.loadtxt(skull_dir / (skull_name.replace('Skull', 'Skin') + '_size.txt'))

    batch = {
        'codedict': codedict_norm,
        'detaildict': detaildict,
        'lmk_on_skull': lmk_on_skull,
        'lmk_depth': lmk_depth,
        'lmk_direction': lmks_dict['direction'][skull_idx],
        'lmk_target': torch.from_numpy(lmk_target).float(),
        'skin_size': torch.from_numpy(skin_size).float(),
        'deca_dir_path': deca_dir_path,
        'skull_name': skull_name,
        'skin_mesh': skin_mesh,
        'skull_mesh': skull_mesh
    }

    return batch


def get_singleface_data(
    face_name: str, 
    cfg_dataset: dict, 
    deca, 
    device: str, 
    force_reconstruct: bool = False
):

    deca_dir_path = Path(cfg_dataset.face_deca_dir) / Path(face_name).stem
    codedict_path = deca_dir_path / 'codedict.pt'
    if force_reconstruct or not codedict_path.exists() :
        deca_dir_path.mkdir(parents=True, exist_ok=True)

        face_img_path = Path(cfg_dataset.face_img_dir) / face_name
        build_3dmm_data(deca, deca_dir_path, face_img_path, device, neutral_pose=True)

    codedict = torch.load(codedict_path, map_location='cpu')
    codedict_norm = normalize_codedict(
        codedict, 
        keep_exp=cfg_dataset.keep_exp, 
        keep_pose=cfg_dataset.keep_pose
    )
    detaildict = np.load(deca_dir_path / 'face_detail.npy', allow_pickle=True)
    detaildict = detaildict.item()
    face_neuralpose_mesh = trimesh.load_mesh(deca_dir_path / 'face_neutralpose.obj', process=False, maintain_order=True, skip_materials=True)
    
    batch = {
        'codedict': codedict_norm,
        'detaildict': detaildict,
        'deca_dir_path': deca_dir_path,
        'face_neuralpose_mesh': face_neuralpose_mesh
    }

    return batch


def get_realskull_data(
        skull_name: str, 
        face_name: str, 
        cfg_dataset: dict, 
        deca, 
        device: str, 
        force_reconstruct: bool = False
    ):

    deca_dir_path = Path(cfg_dataset.face_deca_dir) / Path(face_name).stem
    codedict_path = deca_dir_path / 'codedict.pt'
    if force_reconstruct or not codedict_path.exists() :
        deca_dir_path.mkdir(parents=True, exist_ok=True)

        face_img_path = Path(cfg_dataset.face_img_dir) / face_name
        build_3dmm_data(deca, deca_dir_path, face_img_path, cfg_dataset.dense_face_fids_path, device, neutral_pose=True)

    codedict = torch.load(codedict_path, map_location='cpu', weights_only=True)
    codedict_norm = normalize_codedict(
        codedict, 
        keep_exp=cfg_dataset.keep_exp, 
        keep_pose=cfg_dataset.keep_pose
    )
    detaildict = np.load(deca_dir_path / 'face_detail.npy', allow_pickle=True)
    detaildict = detaildict.item()

    # Load skull info
    skull_dir = Path(cfg_dataset.skull_dir)
    
    skull_mesh = trimesh.load_mesh(
        skull_dir / cfg_dataset.skull_name,
        maintain_order=True, 
        skip_materials=True, 
        process=False
    )

    # print('Load depth samples', cfg_dataset.lmk_depth_samples_path, 'Use index', cfg_dataset.lmk_depth_sample_idx)
    # lmk_depth_samples = np.load(cfg_dataset.lmk_depth_samples_path)
    # lmk_depth = lmk_depth_samples[cfg_dataset.lmk_depth_sample_idx]
    
    lmk_path = skull_dir / 'skull_lmks.txt'
    lmk_ids = np.loadtxt(lmk_path, dtype=np.int32)
    lmk_on_skull = skull_mesh.vertices[lmk_ids]
    skin_size = np.ones(4)

    batch = {
        'codedict': codedict_norm,
        'detaildict': detaildict,
        'lmk_on_skull': lmk_on_skull,
        'skin_size': torch.from_numpy(skin_size).float(),
        'deca_dir_path': deca_dir_path,
        'skull_name': skull_name,
        'skull_mesh': skull_mesh
    }

    return batch