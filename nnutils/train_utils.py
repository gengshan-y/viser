# Copyright 2021 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generic Training Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import time
import pdb
import numpy as np
from absl import flags

import cv2
import subprocess
import soft_renderer as sr
import subprocess
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributed as dist
import trimesh
from kmeans_pytorch import kmeans

from ext_utils.flowlib import flow_to_image, warp_flow
from ext_nnutils.train_utils import Trainer
from nnutils.geom_utils import label_colormap
from ext_nnutils import loss_utils as ext_loss_utils
from nnutils import loss_utils
from nnutils import mesh_net
from nnutils import geom_utils
from dataloader import vid as vid_data


#-------------- flags -------------#
#----------------------------------#
## Flags for training
flags.DEFINE_integer('local_rank', 0, 'for distributed training')
flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_integer('batch_size', 4, 'Size of minibatches')

## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', './logdir',
                    'Root directory for output files')
flags.DEFINE_integer('save_epoch_freq', 1, 'save model every k epochs')


citylabs = label_colormap()

def add_image(log,tag,img,step,scale=True):
    timg = img[0]
    if scale:
        timg = (timg-timg.min())/(timg.max()-timg.min())

    if len(timg.shape)==2:
        formats='HW'
    elif timg.shape[0]==3:
        formats='CHW'
    else:
        formats='HWC'
    log.add_image(tag,timg,step,dataformats=formats)


class LASRTrainer(Trainer):
    def define_model(self):
        opts = self.opts
        if opts.batch_size==3: 
            print('exiting')
            print('potential bug of flow rendering when using batch size=3')
            exit()
        self.symmetric = opts.symmetric

        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.LASR(
            img_size, opts, nz_feat=opts.nz_feat)
        
        if opts.model_path!='':
            self.load_network(self.model, model_path = opts.model_path)

        # ddp
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = self.model.to(device)

        self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[opts.local_rank],
                output_device=opts.local_rank,
                find_unused_parameters=True,
        )
        self.define_criterion_ddp()
        return

    def define_criterion_ddp(self):
        self.model.module.triangle_loss_fn_sr = ext_loss_utils.LaplacianLoss(self.model.module.mean_v[0,0].cpu(), \
                                                                             self.model.module.faces[0,0].cpu()).cuda()
        self.model.module.arap_loss_fn = loss_utils.ARAPLoss(self.model.module.mean_v[0,0].cpu(), \
                                                             self.model.module.faces[0,0].cpu()).cuda()
        self.model.module.flatten_loss = ext_loss_utils.FlattenLoss(self.model.module.faces[0,0].cpu()).cuda() 
        from PerceptualSimilarity.models import dist_model
        self.model.module.ptex_loss = dist_model.DistModel()
        self.model.module.ptex_loss.initialize(model='net', net='alex', use_gpu=False)
        self.model.module.ptex_loss.cuda(self.opts.local_rank)
    
    def set_input(self, batch):
        opts = self.opts
        batch_size = batch['img'].shape[0]
        self.model.module.is_canonical = torch.cat([batch['is_canonical'][:batch_size], batch['is_canonicaln'][:batch_size]],0)
        self.model.module.frameid = torch.cat([    batch['id0'][:batch_size], batch['id1'][:batch_size]],0)

        # Image with annotations.
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        input_imgn_tensor = batch['imgn'].type(torch.FloatTensor)
        for b in range(input_imgn_tensor.size(0)):
            input_imgn_tensor[b] = self.resnet_transform(input_imgn_tensor[b])
        input_img_tensor = torch.cat([input_img_tensor, input_imgn_tensor],0)
        self.model.module.input_imgs = input_img_tensor.cuda()

        img_tensor = batch['img'].type(torch.FloatTensor)
        imgn_tensor = batch['imgn'].type(torch.FloatTensor)
        img_tensor = torch.cat([            img_tensor, imgn_tensor      ],0)
        self.model.module.imgs = img_tensor.cuda()

        shape = img_tensor.shape[2:]

        cam_tensor = batch['cam'].type(torch.FloatTensor)
        camn_tensor = batch['camn'].type(torch.FloatTensor)
        cam_tensor = torch.cat([            cam_tensor, camn_tensor      ],0)
        self.model.module.cams = cam_tensor.cuda()

        flow_tensor = batch['flow'].type(torch.FloatTensor)
        flown_tensor = batch['flown'].type(torch.FloatTensor)
        self.model.module.flow = torch.cat([flow_tensor, flown_tensor],0).cuda()
        self.model.module.occ = torch.cat([  batch['occ'].type(torch.FloatTensor),
                                batch['occn'].type(torch.FloatTensor),],0).cuda()
        self.model.module.oriimg_shape = batch['shape'][:1,:2].repeat(batch_size*2,1).cuda()
                            
        batch_input = {}
        batch_input['input_imgs  '] = self.model.module.input_imgs  
        batch_input['imgs        '] = self.model.module.imgs        
        batch_input['masks       '] = batch['mask'].type(torch.FloatTensor).cuda().permute(1,0,2,3).reshape(batch_size*2,shape[0],shape[1])
        batch_input['cams        '] = self.model.module.cams        
        
        batch_input['flow        '] = self.model.module.flow        
        batch_input['ddts_barrier'] = batch['dmask_dts'].type(torch.FloatTensor).cuda().permute(1,0,2,3).reshape(batch_size*2,1,shape[0],shape[1])
        batch_input['pp          '] = batch['pps'].type(torch.FloatTensor).cuda().permute(1,0,2).reshape(batch_size*2,-1) # bs, 2, x 
        batch_input['occ         '] = self.model.module.occ              
        batch_input['oriimg_shape'] = self.model.module.oriimg_shape     
        batch_input['is_canonical'] = self.model.module.is_canonical 
        batch_input['frameid']      = self.model.module.frameid
        batch_input['dataid']      = torch.cat([    batch['dataid'][:batch_size], batch['dataid'][:batch_size]],0)
        batch_input['rtk']      = torch.cat([    batch['rtk'][:batch_size], batch['rtkn'][:batch_size]],0)
        for k,v in batch_input.items():
            #v: 2xBxk
            batch_input[k] = v.view(2,batch_size,-1).permute(1,0,2).reshape(v.shape)
        return batch_input 

    def init_training(self):
        opts = self.opts
        self.init_dataset()    
        self.define_model()
        cnn_params=[]
        nerf_params=[]
        nerf_shape_params=[]
        for name,p in self.model.module.named_parameters():
            if name == 'mean_v': print('found mean v'); continue
            if name == 'rots': print('found rots'); continue
            if name == 'focal': print('found fl');continue
            if name == 'pps': print('found pps');continue
            if name == 'tex': print('found tex');continue
            if name == 'body_score': print('found body_score');continue
            if name == 'skin': print('found skin'); continue
            if name == 'rest_rs': print('found rest rotation'); continue
            if name == 'ctl_rs': print('found ctl rotation'); continue
            if name == 'ctl_ts': print('found ctl points'); continue
            if name == 'joint_ts': print('found joint points'); continue
            if name == 'transg': print('found global translation'); continue
            if name == 'log_ctl': print('found log ctl'); continue
            if 'nerf_tex' in name or 'nerf_feat' in name or 'nerf_coarse' in name or 'nerf_fine' in name: 
                print('found %s'%name); nerf_params.append(p); continue
            if 'nerf_shape' in name or 'nerf_mshape' in name: 
                print('found %s'%name); nerf_shape_params.append(p); continue
            cnn_params.append(p)
        self.optimizer = torch.optim.AdamW(
            [{'params': cnn_params},
             {'params': nerf_params, 'lr': 10*opts.learning_rate},
             {'params': nerf_shape_params, 'lr': 10*opts.learning_rate},
             {'params': self.model.module.mean_v, 'lr':50*opts.learning_rate},
             {'params': self.model.module.pps, 'lr':50*opts.learning_rate},
             {'params': self.model.module.tex, 'lr':50*opts.learning_rate},
             {'params': self.model.module.ctl_rs, 'lr':50*opts.learning_rate},
             {'params': self.model.module.ctl_ts, 'lr':50*opts.learning_rate},
             {'params': self.model.module.joint_ts, 'lr':50*opts.learning_rate},
             {'params': self.model.module.log_ctl, 'lr':50*opts.learning_rate},
            ],
            lr=opts.learning_rate,betas=(0.9, 0.999),weight_decay=1e-4)
            
        lr_meanv = 50*opts.learning_rate
        cnnlr = opts.learning_rate
        nerflr= 10*opts.learning_rate
        nerf_shape_lr= 10*opts.learning_rate
        conslr = 50*opts.learning_rate

        pct_start = 0.01
        div_factor = 25

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
        [cnnlr,
        nerflr, # nerf-params
        nerf_shape_lr, # nerf shape params
        lr_meanv,
        conslr, # pps
        conslr, # tex
        conslr, # ctl rs 
        conslr, # ctl ts 
        conslr, # joint ts 
        conslr, # log ctl
        ],
        200*len(self.dataloader), pct_start=pct_start, cycle_momentum=False, 
       anneal_strategy='linear',final_div_factor=1./25, div_factor = div_factor)

    def train(self):
        opts = self.opts
        if opts.local_rank==0:
            log = SummaryWriter('%s/%s'%(opts.checkpoint_dir,opts.name), comment=opts.name)
        total_steps = 0
        dataset_size = len(self.dataloader)
        def weights_init(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        if opts.local_rank==0:        self.save('0')

        # initial params
        init_mean_v = self.model.module.mean_v.data.clone()
        init_tex = self.model.module.tex.data.clone()
        init_ctl_rs = self.model.module.ctl_rs.data.clone()
        init_ctl_ts = self.model.module.ctl_ts.data.clone()
        init_joint_ts = self.model.module.joint_ts.data.clone()
        init_log_ctl = self.model.module.log_ctl.data.clone()
        for epoch in range(0, opts.num_epochs):
            self.model.module.epoch = epoch
            epoch_iter = 0

            # reinit bones
            if (self.model.module.reinit_bones or not opts.finetune) and (self.opts.local_rank==0 and epoch==0 and self.opts.n_bones>1):
                for idx in range(self.model.module.mean_v.shape[0]):
                    for jdx in range(self.model.module.mean_v.shape[1]):
                        mean_shape = self.model.module.mean_v[idx,jdx].clone()
                        if self.opts.catemodel:
                            pred_v_symm = mean_shape[None].clone()
                            shape_delta = run_network(
                            self.model.module.nerf_mshape[idx+jdx],
                            pred_v_symm,
                            None,
                            131072,
                            self.model.module.encode_position_fn_shape,
                            None,
                            code=torch.zeros(1,self.model.module.codedim).cuda(),
                            )[:,:,:3]
                            mean_shape = mean_shape + shape_delta[0]

                        cluster_ids_x, cluster_centers = kmeans(
                        X=mean_shape, num_clusters=self.opts.n_bones-1, distance='euclidean', device=torch.device('cuda:%d'%(opts.local_rank)))
                        self.model.module.ctl_ts.data[idx,jdx] = cluster_centers.cuda()
                        self.model.module.joint_ts.data[idx,jdx] = cluster_centers.cuda()
                self.model.module.ctl_rs.data[:,:,:] = torch.Tensor([0,0,0,1]).cuda()
                self.model.module.log_ctl.data[:]= 0
            dist.barrier()
            dist.broadcast(self.model.module.joint_ts, 0)
            dist.broadcast(self.model.module.ctl_ts, 0)
            dist.broadcast(self.model.module.ctl_rs, 0)
            dist.broadcast(self.model.module.log_ctl, 0)
            print('new bone locations')
            
            # TODO reset the baselise and directlist of dataloader
            if self.opts.use_inc: # assuming dataset goes from 0 to -1
                for i in range(len(self.dataloader.dataset.datasets)):
                    vid_length = len(self.dataloader.dataset.datasets[i].imglist)
                    if self.opts.end_idx == -1: end_idx = vid_length
                    else: end_idx = self.opts.end_idx
                    start_idx = self.opts.start_idx

                    delta_max = max(start_idx, vid_length - self.opts.end_idx)
                    if self.opts.delta_max_cap>0: delta_max = self.opts.delta_max_cap
                    inc_frame = int(delta_max * epoch / self.opts.num_epochs)
                    start_idx = max(0,start_idx - inc_frame)
                    end_idx =   min(vid_length-1, end_idx + inc_frame)
                    self.dataloader.dataset.datasets[i].start_idx = start_idx
                    self.dataloader.dataset.datasets[i].end_idx   = end_idx

            self.model.module.ep_iters = len(self.dataloader)
            for i, batch in enumerate(self.dataloader):
                self.model.module.iters=i
                input_batch = self.set_input(batch)

                # set pose-shape indicator: 0: match, 1: reprojection
                if self.opts.use_inc and i<len(self.dataloader)//2:
                    self.model.module.use_kproj = True
                else:
                    self.model.module.use_kproj = False

                # skip a batch
                if (input_batch['masks       '].sum(-1).sum(-1) < 100).sum()>0: 
                    print(input_batch['masks       '].sum(-1).sum(-1) < 100)
                    continue

                if self.opts.debug:
                    torch.cuda.synchronize()
                    start_time = time.time()

                self.model.module.total_steps = total_steps
                self.optimizer.zero_grad()
                total_loss,aux_output = self.model(input_batch)
                total_loss.mean().backward()
                
                if self.opts.debug:
                    torch.cuda.synchronize()
                    print('forward back time:%.2f'%(time.time()-start_time))

                cam_grad = []
                nerf_mshape_grad = []
                nerf_tex_grad = []
                for name,p in self.model.module.named_parameters():
                    #print(name)
                    if 'mean_v' == name and p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, 1.)
                        self.grad_meanv_norm = p.grad.view(-1).norm(2,-1)
                    elif p.grad is not None and ('nerf_mshape' in name):
                        nerf_mshape_grad.append(p)
                    elif p.grad is not None and ('nerf_tex' in name):
                        nerf_tex_grad.append(p)
                    elif p.grad is not None and ('code_predictor' in name or 'encoder' in name):
                        cam_grad.append(p)
                    if (not p.grad is None) and (torch.isnan(p.grad).sum()>0):
                        self.optimizer.zero_grad()
                self.grad_cam_norm = torch.nn.utils.clip_grad_norm_(cam_grad, 10.)
                self.grad_nerf_tex_norm = torch.nn.utils.clip_grad_norm_(nerf_tex_grad, 1.)
                self.grad_nerf_mshape_norm = torch.nn.utils.clip_grad_norm_(nerf_mshape_grad, 1)

                if opts.local_rank==0 and torch.isnan(self.model.module.total_loss):
                    pdb.set_trace()
                self.optimizer.step()
                self.scheduler.step()
                total_steps += 1
                epoch_iter += 1

                if opts.local_rank==0:
                    if i==0:
                        gu = np.asarray(self.model.module.flow[0,0,:,:].detach().cpu()) 
                        gv = np.asarray(self.model.module.flow[0,1,:,:].detach().cpu())
                        
                        flow = np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis]),-1)
                        warped = warp_flow(np.asarray(255*self.model.module.imgs[opts.batch_size].permute(1,2,0).detach().cpu()).astype(np.uint8), flow/2*opts.img_size)
                        add_image(log,'train/warped_flow', warped[None],epoch, scale=False)
                        try:
                            mask = aux_output['vis_mask'][:,0]
                            mask = np.asarray(mask[0].float().cpu())
                        except: mask = np.zeros((opts.img_size, opts.img_size))
                        
                        gu[~mask.astype(bool)] = 0.;  gv[~mask.astype(bool)] = 0.
                        add_image(log,'train/flowobs', flow_to_image(np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis],mask[:,:,np.newaxis]),-1))[np.newaxis],epoch)
                        try:
                            gu = np.asarray(aux_output['flow_rd'].view(2*opts.batch_size,1,opts.img_size,opts.img_size,2)[0,0,:,:,0].detach().cpu())
                            gv = np.asarray(aux_output['flow_rd'].view(2*opts.batch_size,1,opts.img_size,opts.img_size,2)[0,0,:,:,1].detach().cpu())
                            gu[~mask.astype(bool)] = 0.;  gv[~mask.astype(bool)] = 0.
                            add_image(log,'train/flowrd', flow_to_image(np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis],mask[:,:,np.newaxis]),-1))[np.newaxis],epoch)
                            error = np.asarray(aux_output['flow_rd_map'][:1,0].detach().cpu()); error=error*mask
                            add_image(log,'train/flow_error', opts.img_size*error,epoch)
                            add_image(log,'train/mask', 255*np.asarray(aux_output['mask_pred'][:1].detach().cpu()),epoch)
                        except: pass
                        add_image(log,'train/maskgt', 255*np.asarray(self.model.module.masks[:1].detach().cpu()),epoch)
                        img1_j = np.asarray(255*self.model.module.imgs[:1].permute(0,2,3,1).detach().cpu()).astype(np.uint8)
                        add_image(log,'train/img1', img1_j,epoch      ,scale=False)
                        add_image(log,'train/img2', np.asarray(255*self.model.module.imgs[opts.batch_size:opts.batch_size+1].permute(0,2,3,1).detach().cpu()).astype(np.uint8),epoch, scale=False)
                        if opts.n_bones>1 and 'part_render' in aux_output.keys():
                            add_image(log,'train/part', np.asarray(255*aux_output['part_render'][:1].detach().cpu().permute(0,2,3,1), dtype=np.uint8),epoch)
                        if hasattr(self.model.module, 'imatch_vis'):
                            add_image(log,'train/imatch_vis', self.model.module.imatch_vis[None],epoch, scale=True)

                        if 'texture_render' in aux_output.keys():
                            texture_j = np.asarray(aux_output['texture_render'][:1].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8)
                            if opts.n_bones>1:
                                for k in range(aux_output['ctl_proj'].shape[1]):
                                    texture_j[0] = cv2.circle(texture_j[0].copy(),tuple(128+128*np.asarray(aux_output['ctl_proj'][0].detach().cpu())[k,:2]),3,citylabs[k].tolist(),3)
                            add_image(log,'train/texture', texture_j,epoch,scale=False)
                            # joints
                            texture_j = np.asarray(aux_output['texture_render'][:1].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8)
                            if opts.n_bones>1:
                                for k in range(aux_output['joint_proj'].shape[1]):
                                    texture_j[0] = cv2.circle(texture_j[0].copy(),tuple(128+128*np.asarray(aux_output['joint_proj'][0].detach().cpu())[k,:2]),3,citylabs[k].tolist(),3)
                            add_image(log,'train/texture_j', texture_j,epoch,scale=False)
                        if 'texture_render_pred' in aux_output.keys():
                            texture_j = np.asarray(aux_output['texture_render_pred'][:1].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8)
                            if opts.n_bones>1:
                                for k in range(aux_output['ctl_proj_pred'].shape[1]):
                                    texture_j[0] = cv2.circle(texture_j[0].copy(),tuple(128+128*np.asarray(aux_output['ctl_proj_pred'][0].detach().cpu())[k,:2]),3,citylabs[k].tolist(),3)
                            add_image(log,'train/texture_pred', texture_j,epoch,scale=False)
                    log.add_scalar('train/total_loss',  aux_output['total_loss'].mean()  , total_steps)
                    if 'mask_loss' in aux_output.keys():
                        log.add_scalar('train/mask_loss' ,  aux_output['mask_loss'].mean()   , total_steps)
                    if 'flow_rd_loss' in aux_output.keys():
                        log.add_scalar('train/flow_rd_loss',aux_output['flow_rd_loss'].mean(), total_steps)
                    if 'skin_ent_loss' in aux_output.keys():
                        log.add_scalar('train/skin_ent_loss',aux_output['skin_ent_loss'].mean(), total_steps)
                    if 'arap_loss' in aux_output.keys():
                        log.add_scalar('train/arap_loss',aux_output['arap_loss'].mean(), total_steps)
                    if 'cam_loss' in aux_output.keys():
                        log.add_scalar('train/cam_loss',aux_output['cam_loss'].mean(), total_steps)
                    if 'match_loss' in aux_output.keys():
                        log.add_scalar('train/match_loss',aux_output['match_loss'].mean(), total_steps)
                    if 'imatch_loss' in aux_output.keys():
                        log.add_scalar('train/imatch_loss',aux_output['imatch_loss'].mean(), total_steps)
                    if 'kreproj_loss' in aux_output.keys():
                        log.add_scalar('train/kreproj_loss',aux_output['kreproj_loss'].mean(), total_steps)
                    if 'texture_loss' in aux_output.keys():
                        log.add_scalar('train/texture_loss',aux_output['texture_loss'].mean(), total_steps)
                    if 'triangle_loss' in aux_output.keys():
                        log.add_scalar('train/triangle_loss',aux_output['triangle_loss'], total_steps)
                    if 'lmotion_loss' in aux_output.keys():
                        log.add_scalar('train/lmotion_loss', aux_output['lmotion_loss'], total_steps)
                    if 'nerf_tex_loss' in aux_output.keys():
                        log.add_scalar('train/nerf_tex_loss', aux_output['nerf_tex_loss'], total_steps)
                    if 'nerf_shape_loss' in aux_output.keys():
                        log.add_scalar('train/nerf_shape_loss', aux_output['nerf_shape_loss'], total_steps)
                    if 'l1_deform_loss' in aux_output.keys():
                        log.add_scalar('train/l1_deform_loss', aux_output['l1_deform_loss'], total_steps)
                    if hasattr(self, 'grad_meanv_norm'): log.add_scalar('train/grad_meanv_norm',self.grad_meanv_norm, total_steps)
                    if hasattr(self, 'grad_cam_norm'):log.add_scalar('train/grad_cam_norm',self.grad_cam_norm, total_steps)
                    if hasattr(self, 'grad_nerf_mshape_norm'):log.add_scalar('train/grad_nerf_mshape_norm',self.grad_nerf_mshape_norm, total_steps)
                    if hasattr(self, 'grad_nerf_tex_norm'):log.add_scalar('train/grad_nerf_tex_norm',self.grad_nerf_tex_norm, total_steps)
                        
                    if hasattr(self.model.module, 'sampled_img_obs_vis'):
                        if i%10==0:
                            add_image(log,'train/sampled_img_obs_vis', np.asarray(255*self.model.module.sampled_img_obs_vis[0:1, 0].detach().cpu()).astype(np.uint8),epoch, scale=False)
                            add_image(log,'train/sampled_img_rdc_vis', np.asarray(255*self.model.module.sampled_img_rdc_vis[0:1, 0].detach().cpu()).astype(np.uint8),epoch, scale=False)
                            add_image(log,'train/sampled_img_rdf_vis', np.asarray(255*self.model.module.sampled_img_rdf_vis[0:1, 0].detach().cpu()).astype(np.uint8),epoch, scale=False)
                        log.add_scalar('train/coarse_loss',self.model.module.coarse_loss, total_steps)
                        log.add_scalar('train/sil_coarse_loss',self.model.module.sil_coarse_loss, total_steps)
                        log.add_scalar('train/fine_loss',self.model.module.fine_loss, total_steps)

            if (epoch+1) % opts.save_epoch_freq == 0:
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                self.save('latest')
                self.save(epoch+1)
    
    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, local_rank=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = network.state_dict()

        if 'latest' not in save_path:
            save_dict = {k:v for k,v in save_dict.items() if 'uncertainty_predictor' not in k}
        save_dict['faces'] = self.model.module.faces.cpu()
        save_dict['full_shape'] = [self.model.module.symmetrize(   i.cuda()).cpu() for i in self.model.module.mean_v][0]
        save_dict['full_tex'] =   [self.model.module.symmetrize_color(i.cuda()).cpu()  for i in  self.model.module.tex][0]
        torch.save(save_dict, save_path)
        return

    # helper loading function that can be used by subclasses
    def load_network(self, network, model_path=None):
        opts = self.opts
        save_path = model_path
        pretrained_dict = torch.load(save_path,map_location='cpu')
        
        states = pretrained_dict
        numvid, n_hypo = states['mean_v'].shape[:2]
        if opts.finetune:
            pretrained_dict = states
       
        # remesh     
        if int(self.opts.n_faces)!=states['faces'].shape[2]:
            sr.Mesh(states['mean_v'][:1,0], states['faces'][:1,0]).save_obj('tmp/input-%d.obj'%(self.opts.local_rank))
            print(subprocess.check_output(['./Manifold/build/manifold', 'tmp/input-%d.obj'%(self.opts.local_rank), 'tmp/output-%d.obj'%(self.opts.local_rank), '10000']))
            print(subprocess.check_output(['./Manifold/build/simplify', '-i', 'tmp/output-%d.obj'%(self.opts.local_rank), '-o', 'tmp/simple-%d.obj'%(self.opts.local_rank), '-m', '-f', self.opts.n_faces]))
            loadmesh = sr.Mesh.from_obj('tmp/simple-%d.obj'%(self.opts.local_rank))
            self.model.num_verts =   loadmesh.vertices[0].shape[-2]
            self.model.num_faces =   loadmesh.faces[0].shape[-2]
            self.model.mean_v.data = loadmesh.vertices[None]
            self.model.faces.data  = loadmesh.faces   [None]
        else:
            self.model.num_verts = states['mean_v'].shape[-2]
            self.model.num_faces = states['faces'].shape[-2]
            self.model.mean_v.data = states['mean_v'] 
            self.model.faces.data  = states['faces']
        self.model.tex.data = torch.zeros_like(self.model.mean_v.data)
        del states['mean_v']
        del states['tex']
        del states['faces']

        # change number of bones
        self.model.reinit_bones = False
        num_bones_checkpoint = states['code_predictor.0.depth_predictor.pred_layer.bias'].shape[0]
        if num_bones_checkpoint != self.opts.n_bones:
            self.model.reinit_bones = True
            nfeat = states['code_predictor.0.quat_predictor.pred_layer.weight'].shape[-1]
            quat_weights = torch.cat( [states['code_predictor.0.quat_predictor.pred_layer.weight'].view(-1,4,nfeat)[:1], self.model.code_predictor[0].quat_predictor.pred_layer.weight.view(self.opts.n_bones,4,-1)[1:]],0).view(self.opts.n_bones*4,-1)
            quat_bias =    torch.cat( [states['code_predictor.0.quat_predictor.pred_layer.bias'].view(-1,4)[:1],         self.model.code_predictor[0].quat_predictor.pred_layer.bias.view(self.opts.n_bones,-1)[1:]],0).view(-1)
            states['code_predictor.0.quat_predictor.pred_layer.weight'] = quat_weights
            states['code_predictor.0.quat_predictor.pred_layer.bias'] = quat_bias
            
            tmp_weights = torch.cat( [states['code_predictor.0.trans_predictor.pred_layer.weight'].view(-1,2,nfeat)[:1], self.model.code_predictor[0].trans_predictor.pred_layer.weight.view(self.opts.n_bones,2,-1)[1:]],0).view(self.opts.n_bones*2,-1)
            tmp_bias =    torch.cat( [states['code_predictor.0.trans_predictor.pred_layer.bias'].view(-1,2)[:1],         self.model.code_predictor[0].trans_predictor.pred_layer.bias.view(self.opts.n_bones,-1)[1:]],0).view(-1)
            states['code_predictor.0.trans_predictor.pred_layer.weight'] = tmp_weights
            states['code_predictor.0.trans_predictor.pred_layer.bias'] =   tmp_bias
            
            tmp_weights = torch.cat( [states['code_predictor.0.depth_predictor.pred_layer.weight'].view(-1,1,nfeat)[:1], self.model.code_predictor[0].depth_predictor.pred_layer.weight.view(self.opts.n_bones,1,-1)[1:]],0).view(self.opts.n_bones*1,-1)
            tmp_bias =    torch.cat( [states['code_predictor.0.depth_predictor.pred_layer.bias'].view(-1,1)[:1],         self.model.code_predictor[0].depth_predictor.pred_layer.bias.view(self.opts.n_bones,-1)[1:]],0).view(-1)
            states['code_predictor.0.depth_predictor.pred_layer.weight'] = tmp_weights
            states['code_predictor.0.depth_predictor.pred_layer.bias'] =   tmp_bias

            ## initialize skin based on mean shape 
            #np.random.seed(18)
            del states['ctl_rs']
            del states['log_ctl']
            del states['ctl_ts']
            del states['joint_ts']

        if numvid < self.model.numvid:
            self.model.tex.data = self.model.tex.data[:1].repeat(self.model.numvid, 1,1,1)
            self.model.mean_v.data = self.model.mean_v.data[:1].repeat(self.model.numvid, 1,1,1)
            self.model.faces.data = self.model.faces.data[:1].repeat(self.model.numvid, 1,1,1)
            states['rotg'] = states['rotg'][:1].repeat(self.model.numvid, 1,1)
            states['joint_ts'] = states['joint_ts'][:1].repeat(self.model.numvid, 1,1,1)
            states['ctl_ts'] = states['ctl_ts'][:1].repeat(self.model.numvid, 1,1,1)
            states['ctl_rs'] = states['ctl_rs'][:1].repeat(self.model.numvid, 1,1,1)
            states['log_ctl'] = states['log_ctl'][:1].repeat(self.model.numvid,1, 1,1)

        network.load_state_dict(pretrained_dict,strict=False)
        return
