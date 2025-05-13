import torch
import torch.nn as nn
from torch.optim import lr_scheduler, AdamW
import os
import numpy as np
import json
import trimesh
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from DECA.decalib.utils import util
from GAN.model import GAN1D
from GAN.dataset_singlefit import get_singlefit_data
from GAN.utils import interpolate, build_cylinders, build_spheres, chamfer_dist
from pysdf import SDF


def rotation_matrix_to_euler_xyz(R):
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])

    return roll, pitch, yaw


class TrainerFitSkull:

    def __init__(self, deca, cfg, device='cuda'):
        
        self.deca = deca.to(device)
        self.deca.eval()

        with open(cfg.dataset.lmk_path, 'r') as f:
            self.deca_lmk_ids = json.load(f)
        self.lmk_num = len(self.deca_lmk_ids['vids'])
        self.deca_lmk_ids['right_left_pair_vids'] = torch.from_numpy(
            np.array(self.deca_lmk_ids['vids'])[self.deca_lmk_ids['right_left_pairs']]
        )
        self.deca_lmk_ids['mid_vids'] = torch.from_numpy(
            np.array(self.deca_lmk_ids['vids'])[self.deca_lmk_ids['mid_ids']]
        )

        self.dense_face_fids = np.load(cfg.dataset.dense_face_fids_path)

        self.gan = GAN1D(cfg, self.deca_lmk_ids).train()

        self.cfg = cfg
        self.device = device

        # self.loss_d_real = nn.BCELoss().to(device)
        # self.loss_d_gen = nn.BCELoss().to(device)
        self.loss_g_lmk = nn.MSELoss().to(device)
        # self.loss_g_gen = nn.BCELoss().to(device)
        
        self.gan = self.gan.to(device)
        self.build_optimizer()
        
        
    def move_data_to_device(self, batch: dict):
        
        self.batch = batch
        # print('deca_dir_path', self.batch['deca_dir_path'])
        # print('skull_name', self.batch['skull_name'])

        for k in self.batch.keys():
            if k in ['lmk_target', 'skin_size']:
                self.batch[k] = self.batch[k].unsqueeze(0)
            
            if k in ['lmk_on_skull', 'lmk_direction', 'lmk_depth', 'deca_dir_path', 'skull_name', 'skin_mesh', 'detaildict', 'skull_mesh', 'face_neuralpose_mesh']: 
                continue
            if k == 'codedict':
                for key, val in self.batch['codedict'].items():
                    self.batch['codedict'][key] = val.to(self.device)
            else:
                self.batch[k] = self.batch[k].to(self.device)

        return self.batch


    def build_optimizer(self):
        # self.optD = AdamW(self.gan.discriminator.parameters(), lr=self.cfg.train.D_lr)
        self.optG = AdamW(self.gan.generator.parameters(), lr=self.cfg.train.G_lr)
        # self.optD_lr = lr_scheduler.MultiStepLR(self.optD, self.cfg.train.lr_milestones, gamma=0.2)
        self.optG_lr = lr_scheduler.MultiStepLR(self.optG, self.cfg.train.lr_milestones, gamma=0.2)
        # self.optG_lr = lr_scheduler.CosineAnnealingLR(self.optG, self.cfg.train.max_epochs, eta_min=1e-6)
 

    def train_step(self, skull_data_aligned: dict, step: int):
        
        self.gan.train()

        outputs = self.gan.finetinue(
            self.batch, 
            skull_data_aligned['lmk_target_aligned'], 
            self.deca,
            no_gan=False
        )
        loss_dict = dict()
        
        self.optG.zero_grad(set_to_none=True)
        loss_chamfer, _, _ = chamfer_dist(outputs['verts_out'][0], skull_data_aligned['lmk_target_aligned'][0])
        if self.batch['skin_size'].sum() > 0:
            lmk_diff = torch.norm(outputs['lmk_new'] - skull_data_aligned['lmk_target_aligned'], dim=-1) / self.batch['skin_size'][:, -1]
        else:
            lmk_diff = torch.norm(outputs['lmk_new'] - skull_data_aligned['lmk_target_aligned'], dim=-1).mean()
        # print(lmk_diff.shape, torch.norm(outputs['lmk_new'] - outputs['lmk_target'], dim=-1).shape, batch['skin_size'].shape)
        
        lmk_mid_v = outputs['verts_out'][0][self.deca_lmk_ids['mid_vids']].mean(dim=0)
        lmk_sym_pairs = outputs['verts_out'][0][self.deca_lmk_ids['right_left_pair_vids']]
        lmk_sym_vec = lmk_sym_pairs[:, 0] - lmk_sym_pairs[:, 1]
        lmk_sym_vec = lmk_sym_vec / torch.norm(lmk_sym_vec, dim=-1, keepdim=True)
        loss_lmk_sym_normal = 1 - lmk_sym_vec[:, 0].mean()

        lmk_sym_mid = (lmk_sym_pairs[:, 0] + lmk_sym_pairs[:, 1]) / 2
        loss_lmk_sym_mid = torch.abs(lmk_sym_mid[:, 0].mean() - lmk_mid_v[0])

        loss_lmk = self.loss_g_lmk(outputs['lmk_new'], skull_data_aligned['lmk_target_aligned'])
        loss_code_delta_norm = torch.norm(outputs['code_delta'], dim=1).mean()
        loss_g = \
            self.cfg.train.scale_chamfer * loss_chamfer + \
            self.cfg.train.scale_lmk * loss_lmk + \
            self.cfg.train.scale_sym * (loss_lmk_sym_mid + loss_lmk_sym_normal)
        loss_g.backward()
        self.optG.step()

        loss_dict.update({
            'loss_g': loss_g.item(),
            'loss_lmk': loss_lmk.item(),
            'loss_code_delta_norm': loss_code_delta_norm.item(),
            'lmk_diff': lmk_diff.mean().item(),
            'loss_chamfer': loss_chamfer.item(),
            'loss_lmk_sym_normal': loss_lmk_sym_normal.item(),
            'loss_lmk_sym_mid': loss_lmk_sym_mid.item()
        })

        return loss_dict
    

    def eval_step(self, data_aligned: dict, step: int, res_faces_dir: str, no_gan: bool):
        
        lmk_target_aligned = data_aligned['lmk_target_aligned']
        self.gan.eval()

        with torch.no_grad():
            outputs = self.gan.finetinue(
                self.batch, 
                lmk_target_aligned, 
                self.deca,
                no_gan=no_gan
            )
            loss_dict = dict()

            loss_chamfer, _, _ = chamfer_dist(outputs['verts_out'][0], lmk_target_aligned[0])
            lmk_diff = torch.norm(outputs['lmk_new'] - lmk_target_aligned, dim=-1) / self.batch['skin_size'][:, -1]

            loss_lmk = self.loss_g_lmk(outputs['lmk_new'], lmk_target_aligned)
            loss_code_delta_norm = torch.norm(outputs['code_delta'], dim=1).mean()
            loss_g = self.cfg.train.scale_chamfer * loss_chamfer + self.cfg.train.scale_lmk * loss_lmk

        loss_dict.update({
            'loss_g': loss_g.item(),
            'loss_lmk': loss_lmk.item(),
            'loss_code_delta_norm': loss_code_delta_norm.item(),
            'lmk_diff': lmk_diff.mean().item(),
            'loss_chamfer': loss_chamfer.item(),
        })

        face_v = outputs['verts_out'][0].cpu().detach().numpy()
        face_v = (data_aligned['R_inv'] @ (face_v.T - data_aligned['t'] / data_aligned['c'])).T
        lmk_new = outputs['lmk_new'][0].cpu().detach().numpy()
        lmk_new = (data_aligned['R_inv'] @ (lmk_new.T - data_aligned['t'] / data_aligned['c'])).T

        face_mesh = trimesh.Trimesh(face_v, self.deca.render.faces.cpu().numpy()[0])
        export_path = res_faces_dir / f'{step:05d}_reconstructed.ply'

        err_mean = self.eval_accuracy(face_mesh, str(export_path), lmk_new)
        return err_mean


    def eval_accuracy(self, face_mesh, export_path: str, lmk_new):
        
        detaildict = self.batch['detaildict']
        oec = self.batch['skin_size'][0][0].item()
        # face_sdf = SDF(face_mesh.vertices, face_mesh.faces)
        
        # nn_pids = face_sdf.nn(self.batch['skin_mesh'].vertices)
        # nn_pts = face_mesh.vertices[nn_pids]
        # d = np.linalg.norm(nn_pts - self.batch['skin_mesh'].vertices, axis=-1)

        # d = face_sdf(self.batch['skin_mesh'].vertices)

        res_pq = trimesh.proximity.ProximityQuery(face_mesh)
        _, d, _ = res_pq.on_surface(self.batch['skin_mesh'].vertices)

        d = np.abs(d) / oec
        err_mean = d.mean()
        # print('Skin2Face-Diff:', 'Min', d.min(), 'Max', d.max(), 'Abs Mean', err_mean)
        
        normalize_scale = 1e1
        err_colors = interpolate(np.abs(d), normalize='0to1', normalize_scale=normalize_scale, color_map='jet')  # viridis

        err_mesh = trimesh.Trimesh(
            vertices=self.batch['skin_mesh'].vertices,
            faces=self.batch['skin_mesh'].faces,
            vertex_colors=err_colors,
        )
        err_mesh.export(export_path.replace('_reconstructed.ply', '_err.ply'))
        face_mesh.export(export_path)

        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(
            face_mesh.vertices, 
            detaildict['normals'], 
            None, 
            detaildict['displacement_map'], 
            detaildict['texture'], 
            self.deca.dense_template
        )

        dense_mesh = trimesh.Trimesh(
            vertices=dense_vertices,
            faces=dense_faces,
            vertex_colors=dense_colors / 255.0,
        )
        dense_mesh.export(export_path.replace('.ply', '_detail.ply'))

        dense_mesh.update_faces(self.dense_face_fids)
        dense_mesh.export(export_path.replace('.ply', '_detail_face.ply'))

        lmk_target_np = self.batch['lmk_target'][0].cpu().detach().numpy()
        # print(lmk_target_np.shape, lmk_new.shape)

        spheres_target = build_spheres(lmk_target_np, 2e-3, (200, 0, 0))
        spheres_new = build_spheres(lmk_new, 2e-3, (0, 200, 0))
        arrows = build_cylinders(lmk_new, lmk_target_np, 1e-3, (0, 0, 200))
        scene = trimesh.util.concatenate([
            spheres_target, spheres_new, arrows
        ])
        scene.export(export_path.replace('_reconstructed.ply', '_lmk_err.ply'))
        spheres_new.export(export_path.replace('_reconstructed.ply', '_lmk_on_face.ply'))

        return err_mean


    def export_mesh_only(self, data_aligned: dict, step: int, res_faces_dir: str, no_gan: bool):
        
        lmk_target_aligned = data_aligned['lmk_target_aligned']
        self.gan.eval()

        with torch.no_grad():
            outputs = self.gan.finetinue(
                self.batch, 
                lmk_target_aligned, 
                self.deca,
                no_gan=no_gan
            )

        face_v = outputs['verts_out'][0].cpu().detach().numpy()
        face_v = (data_aligned['R_inv'] @ (face_v.T - data_aligned['t']) / data_aligned['c']).T
        lmk_new = outputs['lmk_new'][0].cpu().detach().numpy()
        lmk_new = (data_aligned['R_inv'] @ (lmk_new.T - data_aligned['t']) / data_aligned['c']).T

        face_mesh = trimesh.Trimesh(face_v, self.deca.render.faces.cpu().numpy()[0])
        export_path = str(res_faces_dir / f'{step:05d}_reconstructed.ply')
        
        detaildict = self.batch['detaildict']

        face_mesh.export(export_path)

        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(
            face_mesh.vertices, 
            detaildict['normals'], 
            None, 
            detaildict['displacement_map'], 
            detaildict['texture'], 
            self.deca.dense_template
        )

        dense_mesh = trimesh.Trimesh(
            vertices=dense_vertices,
            faces=dense_faces,
            vertex_colors=dense_colors / 255.0,
        )
        dense_mesh.export(export_path.replace('.ply', '_detail.ply'))

        dense_mesh.update_faces(self.dense_face_fids)
        dense_mesh.export(export_path.replace('.ply', '_detail_face.ply'))

        lmk_target_np = self.batch['lmk_target'][0].cpu().detach().numpy()

        spheres_target = build_spheres(lmk_target_np, 2e-3, (200, 0, 0))
        spheres_new = build_spheres(lmk_new, 2e-3, (0, 200, 0))
        arrows = build_cylinders(lmk_new, lmk_target_np, 1e-3, (0, 0, 200))
        scene = trimesh.util.concatenate([
            spheres_target, spheres_new, arrows
        ])
        scene.export(export_path.replace('_reconstructed.ply', '_lmk_err.ply'))
        spheres_new.export(export_path.replace('_reconstructed.ply', '_lmk_on_face.ply'))


    def fit(self, batch, res_faces_dir, run_dir):
        
        self.move_data_to_device(batch)
        
        best_loss = None

        with torch.no_grad():
            data_aligned = self.gan.align_global(self.batch, self.deca)
        err_mean = self.eval_step(data_aligned, 0, res_faces_dir, no_gan=True)
        logger.info(f'Init error_mean: {err_mean}')

        for epoch in tqdm(range(1, self.cfg.train.max_epochs + 1)):

            self.gan.train()
            lr = self.optG_lr.get_last_lr()[0]
            
            loss_dict = self.train_step(data_aligned, epoch)

            # write summary
            if (epoch % self.cfg.train.log_steps == 0) or (epoch == self.cfg.train.max_epochs):
                err_mean = self.eval_step(data_aligned, epoch, res_faces_dir, no_gan=False)
            
            self.optG_lr.step()

            if epoch % self.cfg.train.log_steps == 0:
                model_dict = {
                    'gan': self.gan.model_dict(),
                    'optG': self.optG.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss
                }

                torch.save(model_dict, os.path.join(run_dir,  f'last.pt'))

        logger.info(f'Final error_mean: {err_mean}')
        return err_mean
    

    def fit_by_lmk(self, batch, res_faces_dir):
        
        self.move_data_to_device(batch)
        
        with torch.no_grad():
            data_aligned = self.gan.align_global(self.batch, self.deca)

        for epoch in tqdm(range(1, self.cfg.train.max_epochs + 1)):

            self.gan.train()
            loss_dict = self.train_step(data_aligned, epoch)

            # write summary
            if (epoch % self.cfg.train.log_steps == 0) or (epoch == self.cfg.train.max_epochs):
                
                self.export_mesh_only(data_aligned, epoch, res_faces_dir, no_gan=False)
                print(loss_dict)
            
            self.optG_lr.step()

        return 0
    