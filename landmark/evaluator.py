from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam, SGD, AdamW
from torch.utils.data import DataLoader
import os
import wandb
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger

from landmark.model import LandmarkModel


class LMKEvaluator:
    def __init__(self, cfg, device='cuda'):
        
        self.device = device
        self.model = LandmarkModel(cfg.model).to(device)
        if cfg.pretrained and os.path.exists(cfg.pretrained):
            print('Load pretrained weights', cfg.pretrained)
            ckpt = torch.load(cfg.pretrained)
            self.model.load_state_dict(ckpt['model'])
            print('Epoch', ckpt['epoch'])
        
        self.model.eval()

    def move_to_device(self, batches):
        for k in batches.keys():
            if k == 'id': continue
            batches[k] = batches[k].to(self.device)

        return batches

    def infer(self, batches):
        
        batches = self.move_to_device(batches)
        
        with torch.no_grad():
            bs = batches['lmk'].shape[0]

            # pred_d = self.model(batches['lmk'].view(bs, -1))
            # pred_d = pred_d.view(bs, -1, 3)
            # pred = batches['lmk'][:, :, :3] + pred_d
            
            # pred_d_norm = torch.norm(pred_d, 'fro', dim=-1)
            # print('norm', pred_d_norm)

            pred_d = self.model(batches['lmk'].view(bs, -1))
            # print('est', pred_d)
            pred = batches['lmk_dir'] * pred_d.reshape(bs, -1, 1) + batches['lmk'][:, :, :3]

        return pred, pred_d
    