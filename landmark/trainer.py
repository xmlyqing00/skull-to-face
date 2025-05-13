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

from landmark.dataset import DatasetLMK
from landmark.model import LmkDistModel


class Trainer:
    def __init__(self, config, device='cuda'):
        
        self.model = LmkDistModel(config.model).to(device)
        # self.D = Discriminator(config.D).to(device)

        self.cfg = config
        self.device = device
        print('device', device)
        self.bs = config.dataset.batch_size
        self.max_epochs = config.train.max_epochs

        # self.difficulty = 0.2
        # self.difficulty_scheduler = dict()
        # for i in range(10):
        #     epoch = int(self.max_epochs * i / 10)
        #     d = min(1, self.difficulty + i / 5)
        #     self.difficulty_scheduler[epoch] = d
        #     if d == 1:
        #         break
        
        # self.lmk_range = config.dataset.lmk_range

        self.loss_regression = nn.MSELoss().to(device)
        # self.loss_delta = nn.L1Loss().to(device)
        # self.loss_d_real = nn.BCELoss().to(device)
        # self.loss_d_gen = nn.BCELoss().to(device)
        # self.loss_g_gen = nn.BCELoss().to(device)

        self.regression_scale = self.cfg.train.regression_scale
        # self.penalty_scale = self.cfg.train.penalty_scale

        # self.labels = {
            # 'real': torch.ones(self.bs, 1).to(device),
            # 'fake': torch.zeros(self.bs, 1).to(device)
        # }
        
        self.start_epoch = 0

        self.optimizer()
        self.prepare_data()
        
    def prepare_data(self):

        self.train_dataset = DatasetLMK('train', self.cfg.dataset)
        self.val_dataset = DatasetLMK('val', self.cfg.dataset)
        logger.info(f'Training data numbers: {len(self.train_dataset)}')
        logger.info(f'Validation data numbers: {len(self.val_dataset)}')

        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.bs, 
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=self.cfg.dataset.pin_memory,
            drop_last=self.cfg.train.drop_last
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.bs, 
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=self.cfg.dataset.pin_memory,
            drop_last=self.cfg.val.drop_last
        )

    def optimizer(self):
        self.opt = AdamW(self.model.parameters(), float(self.cfg.train.lr))
        # self.optD = AdamW(self.D.parameters(), float(self.cfg.train.lr) * 0.1)
        # self.opt_lr = lr_scheduler.MultiStepLR(self.opt, self.cfg.train.lr_milestones)
        self.opt_lr = lr_scheduler.CosineAnnealingLR(self.opt, self.max_epochs)
        # self.optD_lr = lr_scheduler.MultiStepLR(self.optD, self.cfg.train.lr_milestones)


    def set_requires_grad(self, net, requires_grad: bool):

        for param in net.parameters():
            param.requires_grad = requires_grad

    def move_to_device(self, batches):
        for k in batches.keys():
            if k == 'id': continue
            batches[k] = batches[k].to(self.device)

        return batches

    def training_step(self, batches):
            
        batches = self.move_to_device(batches)
        
        pred = self.model(
            batches['lmk_source'].view(self.bs, -1),
            batches['lmk_delta_input'].view(self.bs, -1)
        )

        # generator loss
        self.opt.zero_grad(set_to_none=True)
        loss_regression = self.loss_regression(
            pred, 
            batches['lmk_target'].view(self.bs, -1)
        )
        loss_g = self.regression_scale * loss_regression
        loss_g.backward()
        self.opt.step()
        
        results = {
            'loss_regression': loss_regression.item(),
            'loss_g': loss_g.item(),
        }

        return results
    

    def validataion_step(self, batches):

        batches = self.move_to_device(batches)

        with torch.no_grad():
            pred = self.model(
                batches['lmk_source'].view(self.bs, -1),
                batches['lmk_delta_input'].view(self.bs, -1)
            )
            
            loss_regression = self.loss_regression(
                pred, 
                batches['lmk_target'].view(self.bs, -1)
            )
            loss_g = self.regression_scale * loss_regression

        results = {
            'loss_regression': loss_regression.item(),
            'loss_g': loss_g.item(),
        }

        return results
    

    def fit(self):

        best_loss = 100000000

        for epoch in range(self.start_epoch, self.max_epochs):
            
            lr = self.opt_lr.get_last_lr()[0]
            info = f"ExpName: {self.cfg.exp_name}. LR: {lr:.2e}.\n"
            wandb.log({'train/lr': lr},  step=epoch)
            logger.info(info)
            self.model.train()
            loss_g_list = []

            for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):

                # continue
                results = self.training_step(batch)
                loss_g_list.append(results['loss_g'])
            
            # write summary
            loss_g_np = np.array(loss_g_list)
            wandb.log({'train/loss_avg': loss_g_np.mean()}, step=epoch)                    
            wandb.log({'train/loss_max': loss_g_np.max()}, step=epoch)                    
            logger.info(f'loss_avg: {loss_g_np.mean():.4f}. loss_max: {loss_g_np.max():.4f}')
               
            self.opt_lr.step()
            
            if epoch % self.cfg.val.val_epoch == 0:
                self.model.eval()
                loss_g_list = []

                for step, batch in enumerate(tqdm(self.val_dataloader, desc=f"Val: {epoch}/{self.cfg.train.max_epochs}")):
                    
                    # continue
                    results = self.validataion_step(batch)
                    loss_g_list.append(results['loss_g'])

                loss_g_np = np.array(loss_g_list)
                wandb.log({'val/loss_avg': loss_g_np.mean()}, step=epoch)                    
                wandb.log({'val/loss_max': loss_g_np.max()}, step=epoch)                    
                logger.info(f'Validation loss_avg: {loss_g_np.mean():.4f}. loss_max: {loss_g_np.max():.4f}')

            val_loss = loss_g_np.mean()
            model_dict = {
                'model': self.model.state_dict(),
                'opt': self.opt.state_dict(),
                'epoch': epoch,
                'loss': (val_loss, best_loss),
            }

            torch.save(model_dict, os.path.join(wandb.run.dir,  f'last.pt'))
            logger.info(f'Save last model weights to {wandb.run.dir}. Cur loss {val_loss:.5f}')

            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                
                torch.save(model_dict, os.path.join(wandb.run.dir,  f'best.pt'))
                logger.info(f'Update best model weights to {wandb.run.dir}. Best loss {best_loss:.5f}')
