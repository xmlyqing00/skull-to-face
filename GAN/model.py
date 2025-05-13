import torch
import torch.nn as nn
import numpy as np

from DECA.decalib.utils.lossfunc import VGGFace2Loss
from DECA.decalib.utils import util, renderer
from landmark.umeyama import umeyama


class Discriminator(nn.Module):
    '''
    Input: VGG features of size
    '''
    def __init__(self, model_config):
        super(Discriminator, self).__init__()
        
        channels = model_config.channels
        mlp = []
        for i in range(1, len(channels)):
            mlp.append(nn.Sequential(
                nn.Linear(channels[i-1], channels[i], bias=False),
                # nn.BatchNorm1d(channels[i]),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.mlp = nn.Sequential(*mlp)
        self.fc = nn.Linear(model_config.channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        return self.sigmoid(self.fc(x))
        

class Generator1D(nn.Module):
    '''
    Input: 2 1D tensor
    Output: 1 1D tensor
    '''
    def __init__(self, model_config):
        super(Generator1D, self).__init__()
        lmk_len = model_config.lmk_len
        shp_len = model_config.shape_len
        hidden_channel = model_config.hidden_channel
        out_channel = shp_len
        self.lmk_fc = nn.Sequential(
            nn.Linear(lmk_len, hidden_channel[0]//2),
            # nn.InstanceNorm1d(hidden_channel[0]//2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.shp_fc = nn.Sequential(
            nn.Linear(shp_len, hidden_channel[0]//2),
            # nn.InstanceNorm1d(hidden_channel[0]//2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        mlp = []
        for i in range(1, len(hidden_channel)):
            mlp.append(nn.Sequential(
                nn.Linear(hidden_channel[i-1], hidden_channel[i]),
                # nn.InstanceNorm1d(hidden_channel[i]),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        self.fc = nn.Sequential(*mlp)
        self.out_fc = nn.Linear(hidden_channel[-1], out_channel)
        # nn.init.zeros_(self.out_fc.weight)
    
    def forward(self, lmk, shp):
        lmk = self.lmk_fc(lmk)
        shp = self.shp_fc(shp)
        x = torch.cat([lmk, shp], dim=1)
        x = self.fc(x)
        x = self.out_fc(x)
        return x


class GAN1D(nn.Module):
    def __init__(self, config, deca_lmk_ids):

        super(GAN1D, self).__init__()
        
        self.generator = Generator1D(config.model.generator)
        self.discriminator = Discriminator(config.model.discriminator)
        self.deca_lmk_ids = deca_lmk_ids
        self.cfg = config
 
        # if config.mode == 'train':
            # self.vgg = VGGFace2Loss(pretrained_model=config.model.fr_model_path).eval()
            # for param in self.vgg.parameters():
                # param.requires_grad = False
        
            # renderer.set_rasterizer(self.cfg.rasterizer_type)
            # self.render = renderer.SRenderY(
            #     config.renderer.image_size, 
            #     obj_filename=config.renderer.topology_path, 
            #     uv_size=config.renderer.uv_size, 
            #     rasterizer_type=config.renderer.rasterizer_type
            # )
            # self.image_size = config.renderer.image_size
    
    def load_chkp(self, path):
        model_chkp = path
        chkp = torch.load(model_chkp, map_location='cpu')
        cur_dict = self.model_dict()
        print(f'trained model found. load {model_chkp}')
        for k in chkp.keys():
            if k in cur_dict.keys():
                print('load module dict', k, 'from checkpoint')
                util.copy_state_dict(cur_dict[k], chkp[k])
    
    def model_dict(self):
        d =  dict(
            generator = self.generator.state_dict(),
            discriminator = self.discriminator.state_dict(),
        )
        return d

    def forward(self, batch):
        
        codedict = batch['codedict']
        albedo = batch['albedo']
        bs = albedo.shape[0]

        verts_in, _, _ = self.deca.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        lmk_init = verts_in[:, self.deca_lmk_ids['vids']]

        lmk_target = lmk_init[list(range(bs//2, bs)) + list(range(bs//2))]
        lmk_delta_target = lmk_target - lmk_init
        
        # generate fake shape params
        gan_input = torch.concat([codedict['shape'], codedict['exp'], codedict['pose']], dim=1)
        # gan_input = torch.concat([codedict['shape'], codedict['exp']], dim=1)
        codedict_out = self.generator(lmk_delta_target.view(bs, -1), gan_input) + gan_input
        shape_out = codedict_out[:, :100]
        exp_out = codedict_out[:, 100:150]
        pose_out = codedict_out[:, 150:]

        verts_out, _, _ = self.deca.flame(shape_params=shape_out, expression_params=exp_out, pose_params=pose_out)
        # verts_out, _, _ = self.deca.flame(shape_params=shape_out, expression_params=exp_out, pose_params=codedict['pose'])
        lmk_new = verts_out[:, self.deca_lmk_ids['vids']]
        lmk_delta_new = lmk_new - lmk_init

        cam = shape_out.new_tensor([[10.0, 0.0, 0.0]]).repeat(bs, 1)
        trans_verts_in = util.batch_orth_proj(verts_in, cam)
        trans_verts_in[:,:,1:] = -trans_verts_in[:,:,1:]
        # render_res = self.render(verts_in, trans_verts_in, albedo, h=self.image_size, w=self.image_size)
        # rendered_image = render_res['images']
        rendered_image = batch['rendered_image']

        trans_verts_out = util.batch_orth_proj(verts_out, cam)
        trans_verts_out[:,:,1:] = -trans_verts_out[:,:,1:]
        render_res = self.render(verts_out, trans_verts_out, albedo, h=self.image_size, w=self.image_size)
        
        rendered_image_new = render_res['images']
        F_real = self.vgg.forward_features(rendered_image)
        F_gen = self.vgg.forward_features(rendered_image_new)

        logdict = {
            'shape_out change max': (verts_out / verts_in - 1).abs().max().item(),
            'shape_out change mean': (verts_out / verts_in - 1).abs().mean().item(),
            'landmark_init_depth': lmk_delta_target.norm(dim=-1).mean().item(),
        }

        outputs = {
            'F_real': F_real, 
            'F_gen': F_gen, 
            'lmk_delta_target': lmk_delta_target, 
            'lmk_delta_new': lmk_delta_new
        }
       
        vis_dict = {
            'images': rendered_image.detach(),
            'gen_images': rendered_image_new.detach(),
            'verts_in': verts_in.detach(),
            'verts_out': verts_out.detach(),
            'trans_verts_in': trans_verts_in.detach(),
            'trans_verts_out': trans_verts_out.detach()
        }

        return outputs, vis_dict, logdict


    def align_global(self, batch: dict, deca):
        codedict = batch['codedict']
        lmk_target = batch['lmk_target'][0]

        verts_in, _, _ = deca.flame(
            shape_params=codedict['shape'], 
            expression_params=codedict['exp'], 
            pose_params=codedict['pose']
        )
        lmk_init = verts_in[0, self.deca_lmk_ids['vids']]

        # Adjust global alignment
        lmk_target_np = lmk_target.cpu().detach().numpy()
        lmk_init_np = lmk_init.cpu().detach().numpy()

        c, R, t = umeyama(lmk_target_np.T, lmk_init_np.T)
        # skin_size_aligned = batch['skin_size'][0].clone() * c
        lmk_target_aligned = torch.from_numpy(
            (c * R @ lmk_target_np.T + t).T
        ).unsqueeze(0).to(lmk_target.device).float()

        # skin_v = np.array(batch['skin_mesh'].vertices)
        # skin_mesh_aligned = batch['skin_mesh'].copy()
        # skin_mesh_aligned.vertices = (c * R @ skin_v.T + t).T

        # skull_v = np.array(batch['skull_mesh'].vertices)
        # skull_mesh_aligned = batch['skull_mesh'].copy()
        # skull_mesh_aligned.vertices = (c * R @ skull_v.T + t).T

        # lmk_on_skull_aligned = (c * R @ batch['lmk_on_skull'][0].cpu().detach().numpy().T + t).T
        # lmk_target_gt = (c * R @ batch['lmk_target_gt'][0].cpu().detach().numpy().T + t).T


        return {
            'lmk_target_aligned': lmk_target_aligned.contiguous(),
            'c': c,
            'R': R,
            't': t,
            'R_inv': np.linalg.inv(R),
            # 'skin_size_aligned': skin_size_aligned.contiguous(),
            # 'lmk_on_skull_aligned': lmk_on_skull_aligned,
            # 'lmk_target_gt': lmk_target_gt,
            # 'skin_mesh_aligned': skin_mesh_aligned,
            # 'skull_mesh_aligned': skull_mesh_aligned,
            # 'skin_vertices': torch.from_numpy(skin_mesh_aligned.vertices).float().to(lmk_target.device),
            # 'skull_vertices': torch.from_numpy(skull_mesh_aligned.vertices).float().to(lmk_target.device)
        }
    

    def finetinue(self, batch: dict, lmk_target_aligned: torch.Tensor, deca, no_gan):
        
        codedict = batch['codedict']

        # generate fake shape params
        codedict_in = torch.concat([codedict['shape'], codedict['exp'], codedict['pose']], dim=1)
        # codedict_in = codedict['shape']
        # codedict_out = self.generator(lmk_delta_target.view(bs, -1), codedict_in) + codedict_in
        # code_delta = self.generator(lmk_target.view(bs, -1), codedict_in)
        # codedict_out = code_delta + codedict_in
        if no_gan:
            codedict_out = codedict_in
        else:
            codedict_out = self.generator(lmk_target_aligned.reshape(codedict_in.shape[0], -1), codedict_in)

        shape_out = codedict_out[:, :100]
        exp_out = codedict_out[:, 100:150]
        pose_out = codedict_out[:, 150:]
        # exp_out = codedict['exp']
        # pose_out = codedict['pose']
        # pose_out[:, 4] = 0
        # pose_out[:, 5] = 10
        # print(pose_out)

        verts_out, _, _ = deca.flame(shape_params=shape_out, expression_params=exp_out, pose_params=pose_out)
        # verts_out, _, _ = self.deca.flame(shape_params=shape_out, expression_params=exp_out, pose_params=codedict['pose'])
        lmk_new = verts_out[:, self.deca_lmk_ids['vids']]
        # print('lmknew', lmk_new)
        # lmk_delta_new = lmk_new - lmk_init

        # cam = shape_out.new_tensor([[10.0, 0.0, 0.0]]).repeat(bs, 1)
        # trans_verts_in = util.batch_orth_proj(verts_in, cam)
        # trans_verts_in[:,:,1:] = -trans_verts_in[:,:,1:]
        # render_res = self.render(verts_in, trans_verts_in, albedo, h=self.image_size, w=self.image_size)
        # rendered_image = render_res['images']

        # trans_verts_out = util.batch_orth_proj(verts_out, cam)
        # trans_verts_out[:,:,1:] = -trans_verts_out[:,:,1:]
        # render_res = self.render(verts_out, trans_verts_out, albedo, h=self.image_size, w=self.image_size)
        # rendered_image_new = render_res['images']

        # F_real = self.vgg.forward_features(rendered_image)
        # F_gen = self.vgg.forward_features(rendered_image_new)

        outputs = {
            'verts_out': verts_out,
            'lmk_new': lmk_new,
            'code_delta': codedict_out - codedict_in,
        }

        return outputs
    

    def infer(self, batch):

        codedict = batch['codedict']
        bs = codedict['shape'].shape[0]

        verts_in, _, _ = self.deca.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        batch['lmk_init'] = verts_in[:, self.deca_lmk_ids['vids']]
        # print(batch['lmk_init'] - lmk_init)

        lmk_delta_target = batch['lmk_target'] - batch['lmk_init']
        # lmk_delta_target = batch['lmk_target'] - lmk_init
        
        # generate fake shape params
        gan_input = torch.concat([codedict['shape'], codedict['exp']], dim=1)
        # gan_input = torch.concat([codedict['shape'], codedict['exp'], codedict['pose']], dim=1)
        codedict_out = self.generator(lmk_delta_target.view(bs, -1), gan_input) + gan_input
        
        shape_out = codedict_out[:, :100]
        exp_out = codedict_out[:, 100:150]
        # pose_out = codedict_out[:, 150:]
        verts_out, _, _ = self.deca.flame(shape_params=shape_out, expression_params=exp_out, pose_params=codedict['pose'])
        # verts_out, _, _ = self.deca.flame(shape_params=shape_out, expression_params=exp_out, pose_params=pose_out)
        
        codedict_out = codedict.copy()
        codedict_out['shape'] = shape_out
        codedict_out['exp'] = exp_out
        # codedict_out['pose'] = pose_out
        
        return codedict_out, verts_out