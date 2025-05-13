import os
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import shutil
import pickle
import argparse
import json
from datetime import datetime
from loguru import logger
from pathlib import Path

from DECA.decalib.deca import DECA
from DECA.decalib.utils.config import cfg as deca_cfg

from GAN.trainer_fitskull import TrainerFitSkull
from GAN.utils import parse_args, build_cylinders, build_spheres
from GAN.dataset_singlefit import get_realskull_data


def main(cfg):
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    json.encoder.FLOAT_REPR = lambda o: format(o, '.8f')

    # creat folders for each face
    unique_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f'output/{unique_str}')
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.add(run_dir / 'edit_3dface.log')
    logger.info(f'Log Dir: {run_dir}')
    with open(run_dir / 'full_config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    shutil.copy(cfg.cfg_file, run_dir / 'train_config.yaml')
    
    # cudnn related setting 
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    deca_cfg.pretrained_modelpath = str('assets/deca_model.tar')
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.use_tex = True
    deca_cfg.model.extract_tex = True
    deca_cfg.model.tex_type = 'FLAME'
    deca_cfg.model.flame_model_path = str('assets/generic_model.pkl')
    deca_cfg.model.flame_tex_path = str('assets/FLAME_texture.npz')
    deca_cfg.model.fr_model_path = str('assets/resnet50_ft_weight.pkl')
    deca = DECA(deca_cfg)
    device = f'{cfg.device}:{cfg.device_id}'
    
    if cfg.dataset.face_names == 'all':
        face_paths = Path(cfg.dataset.face_img_dir).glob('*.png')
        cfg.dataset.face_names = [face_path.name for face_path in face_paths]

    with open(cfg.dataset.lmk_depth_pca_model, 'rb') as f:
        pca = pickle.load(f)

    lmks_dict = np.load(cfg.dataset.lmks_dict_path, allow_pickle=True).item()
    lmks_dir_flag = 'avg' # 'normal' or 'avg'
    if lmks_dir_flag == 'avg':
        lmks_avg_dir = lmks_dict['direction'].mean(axis=0)
        lmks_avg_dir = lmks_avg_dir / np.linalg.norm(lmks_avg_dir, axis=-1, keepdims=True)

        # with open(cfg.dataset.lmk_path, 'r') as f:
            # deca_lmk_ids = json.load(f)

    skull_name = Path(cfg.dataset.skull_dir).stem
    scale = 500

    for face_name in cfg.dataset.face_names:    

        batch_original = get_realskull_data(
            skull_name,
            face_name, 
            cfg.dataset, 
            deca, 
            device, 
            force_reconstruct=cfg.dataset.force_reconstruct
        )

        trainer = TrainerFitSkull(deca, cfg, device=device)
        batch_original = trainer.move_data_to_device(batch_original)
        
        if lmks_dir_flag == 'normal':
            lmks_avg_dir = batch_original['skull_mesh'].vertex_normals[trainer.deca_lmk_ids['vids']]
        
        dir_id = 'avg'

        lmk_deltas = [0.1]
        for i, lmk_delta in enumerate(lmk_deltas):
            
            logger.info(f'{i}, Current lmk delta: {lmk_delta:.2f}')
            
            batch = batch_original.copy()
            
            res_faces_dir = run_dir / f'edit_{skull_name}_dir_{dir_id}_{i}'
            res_faces_dir.mkdir(parents=True, exist_ok=True)

            lmk_depth_new = pca.inverse_transform(lmk_delta) * scale
            batch['lmk_target'] = batch_original['lmk_on_skull'] + lmks_avg_dir * lmk_depth_new[0][..., np.newaxis]
            
            sticks = build_cylinders(
                batch_original['lmk_on_skull'], 
                batch['lmk_target'], 
                0.001 * scale, (50, 130, 246)
            )
            sticks.export(res_faces_dir / 'skull_lmk_tissues.ply')
            spheres_target = build_spheres(
                batch['lmk_target'], 
                0.002 * scale, (117, 250, 97)
            )
            spheres_target.export(res_faces_dir / 'skull_lmk_target.ply')

            batch['lmk_target'] = torch.from_numpy(batch['lmk_target']).float()
            batch['skin_size'] = torch.zeros(4, dtype=torch.float32)
            trainer.fit_by_lmk(batch, res_faces_dir)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Edit Celebrity Face")
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--mode', type=str, default='train', help='deca mode')
    parser.add_argument('--force-reconstruct', action='store_true', help='Force to reconstruct 3D face from DECA.')
    # parser.add_argument('--pretrained', type=str, default='wandb/offline-run-20231108_163201-16xq03g8/files/last.pt', help='pretrained model path')
    cfg = parse_args(parser)
    
    main(cfg)
