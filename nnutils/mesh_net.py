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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')

import cv2
import configparser
import numpy as np
import configparser
import time
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import trimesh
import pytorch3d
import pytorch3d.loss

import pdb
from ext_utils import mesh
from ext_utils.quatlib import q_rnd_m, q_scale_m
from ext_utils.util_rot import compute_geodesic_distance_from_two_matrices
from ext_utils import geometry as geom_utils
from ext_nnutils import net_blocks as nb
from ext_nnutils.mesh_net import MeshNet
from nerf import models as nerf_models, get_embedding_function, run_network
import kornia
import configparser
import soft_renderer as sr
from nnutils.geom_utils import pinhole_cam, obj_to_cam, render_flow_soft_3, render_multiplex
from nnutils.geom_utils import label_colormap
from nnutils.loss_utils import mesh_area
from nnutils.cenet import SurfaceMatchNet
from nnutils.net_blocks import CodePredictorTex
citylabs = label_colormap()

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_boolean('noise', True, 'Add random noise to pose')
flags.DEFINE_boolean('symmetric', False, 'Use symmetric mesh or not')
flags.DEFINE_boolean('symmetric_loss', False, 'Use symmetric loss or not')
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')
flags.DEFINE_boolean('debug', False, 'deubg')
flags.DEFINE_bool('finetune', False, 'whether to load the full model and finetune it')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_boolean('symmetric_texture', True, 'if true texture is symmetric!')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')
flags.DEFINE_integer('symidx', 0, 'symmetry index: 0-x 1-y 2-z')
flags.DEFINE_integer('n_bones', 1, 'num of bones')
flags.DEFINE_string('n_faces', '1280','number of faces for remeshing')
flags.DEFINE_integer('start_idx', 0, 'index for initial set of frames')
flags.DEFINE_integer('end_idx', -1,  'index for initial set of frames')
flags.DEFINE_integer('delta_max_cap', -1,  'maximum delta frames for incremental optimization')

flags.DEFINE_boolean('only_mean_sym', False, 'If true, only the meanshape is symmetric')
flags.DEFINE_boolean('ptex', True, 'If true, use implicit func for volume texture')
flags.DEFINE_boolean('pshape', False, 'If true, use implicit func for shape variation volume')
flags.DEFINE_string('dataname', 'fashion', 'name of the test data')
flags.DEFINE_boolean('use_inc', False, 'If true uses incremental during fine-tuning')
flags.DEFINE_string('opt_tex', 'yes', 'optimize texture')
flags.DEFINE_float('rscale', 1.0, 'scale random variance')
flags.DEFINE_float('sil_wt', 0.5, 'weight of flow loss')
flags.DEFINE_float('tex_wt', 0.25, 'weight of flow loss')
flags.DEFINE_float('flow_wt', 0.5, 'weight of flow loss')
flags.DEFINE_float('l1tex_wt', 1.0, 'weight of l1 texture')
flags.DEFINE_float('arap_wt', 1.0, 'weight of arap loss')
flags.DEFINE_float('sigval', 1e-4, 'blur radius of soft renderer')
flags.DEFINE_float('l1_wt', 1.0, 'weight of arap loss')
flags.DEFINE_float('triangle_reg_wt', 0.005, 'weights to triangle smoothness prior')
flags.DEFINE_bool('catemodel', False, 'learn a category model')
flags.DEFINE_bool('cnnpp', False, 'cnn principle points')
flags.DEFINE_string('model_path', '',
                    'load model path')


def render_feature(pred_v, faces, vfeat, Rmat, Tmat, skin, ppoint, scale,
                renderer_softfls, local_batch_size, n_mesh):
    feats_render = []
    multip = vfeat.shape[-1]//3
    vfeat = vfeat[:,:,:multip*3].view(local_batch_size, vfeat.shape[1],multip,3).permute(2,0,1,3).reshape(local_batch_size*multip,-1,3)
    pred_v = pred_v.repeat(multip,1,1).clone()
    faces = faces.repeat(multip,1,1).clone()
    Rmat = Rmat.repeat(multip,1,1).clone()
    Tmat = Tmat.repeat(multip,1).clone()
    skin  = skin.repeat(multip,1,1,1).clone()
    ppoint  = ppoint.repeat(multip,1,1).clone()
    scale  = scale.repeat(multip,1,1).clone()

    feat_render = render_multiplex(pred_v, faces, vfeat, Rmat, Tmat, skin, ppoint, scale, renderer_softfls, local_batch_size*multip,n_mesh)
    feat_render = feat_render.view(multip, local_batch_size, 4, feat_render.shape[-2], feat_render.shape[-1])
    mask_render = feat_render[0,:,-1]
    feat_render = feat_render.permute(1,0,2,3,4)[:,:,:3].reshape(local_batch_size, -1, feat_render.shape[-2], feat_render.shape[-1])
    feat_render = feat_render * mask_render[:,None]
    return feat_render, mask_render

def reg_decay(curr_steps, max_steps, min_wt,max_wt):
    """
    max weight to min weight
    """
    if curr_steps>max_steps:current = min_wt
    else:
        current = np.exp(curr_steps/float(max_steps)*(np.log(min_wt)-np.log(max_wt))) * max_wt 
    return current

class LASR(MeshNet):
    def __init__(self, input_shape, opts, nz_feat=100):
        super(LASR, self).__init__(input_shape, opts, nz_feat)
        self.opts = opts
        self.symmetric = opts.symmetric
        
        # Input shape is H x W of the image.
        super(MeshNet, self).__init__()
        self.reinit_bones = True

        # multi-vid
        sym_angle = 0
        ppx=960; ppy=540
        if hasattr(opts, 'dataset'): # training
            config = configparser.RawConfigParser()
            config.read('data/%s.config'%opts.dataname)
            numvid = int(config.get('meta', 'numvid'))
            try: sym_angle = int(config.get('data_0', 'sym_angle'))
            except: sym_angle = 0
            total_fr = int(config.get('data_0', 'end_frame'))
            try:
                ppx=int(config.get('data_0', 'ppx'))
                ppy=int(config.get('data_0', 'ppy'))
            except: pass
        else:
            numvid = 1
        self.numvid = numvid

        # rest shape
        if osp.exists('tmp/sphere_%d.npy'%(opts.subdivide)):
            sphere = np.load('tmp/sphere_%d.npy'%(opts.subdivide),allow_pickle=True)[()]
            verts = sphere[0]
            faces = sphere[1]
        else:
            verts, faces = mesh.create_sphere(opts.subdivide)
            np.save('tmp/sphere_%d.npy'%(opts.subdivide),[verts,faces])
        self.num_verts = verts.shape[0]
        self.num_faces = faces.shape[0]

        self.mean_v = nn.Parameter(torch.Tensor(verts)[None])
        faces = Variable(torch.LongTensor(faces), requires_grad=False)
        self.texture_type = 'vertex'
        if self.opts.opt_tex=='yes':
            self.tex = nn.Parameter(torch.normal(torch.zeros(1,self.num_verts,3).cuda(),1))
        else:
            self.tex = torch.normal(torch.zeros(self.num_verts,3).cuda(),1)
        self.nfeat=16

        self.mean_v.data = self.mean_v.data.repeat(1,1,1).view(1, 1,self.num_verts,3)
        self.tex.data = self.tex.data.repeat(1,1,1).view(1, 1, self.num_verts,3)  # id, hypo, F, 3
        faces = faces[None].repeat(1,1,1).view(1, 1, self.num_faces,3)
        self.joint_ts =  torch.zeros(1*(opts.n_bones-1),3).cuda().view(1, 1, -1, 3)
        self.ctl_ts =    torch.zeros(1*(opts.n_bones-1),3).cuda().view(1, 1, -1, 3)
        self.ctl_rs =  torch.Tensor([[0,0,0,1]]).repeat(1*(opts.n_bones-1),1).cuda().view(1, 1, -1, 4)
        self.log_ctl = torch.Tensor([[0,0,0]]).repeat(  1*(opts.n_bones-1),1).cuda().view(1, 1, -1, 3)  # control point varuance
       
        self.faces=faces

        if self.opts.n_bones>1:
            self.ctl_rs  = nn.Parameter(self.ctl_rs) 
            self.ctl_ts  = nn.Parameter(self.ctl_ts) 
            self.joint_ts  = nn.Parameter(self.joint_ts) 
            self.log_ctl = nn.Parameter(self.log_ctl)

        # shape basis
        self.shape_code_fix = nn.Parameter(torch.zeros(1, 1, 60).cuda()) # vid, hp, 8

        self.resnet_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

        # pose net 
        self.encoder = nn.ModuleList([nb.Encoder(input_shape, n_blocks=4, nz_feat=nz_feat) for i in range(1)])
        self.csmnet = SurfaceMatchNet()

        self.code_predictor = nn.ModuleList([nb.CodePredictor(nz_feat=nz_feat, \
                 num_verts=self.num_verts, n_bones=opts.n_bones, n_hypo = 1,
                    ) for i in range(1)])

        self.codedim = 60
        self.encoder_class = nb.Encoder(input_shape, n_blocks=4, nz_feat=nz_feat)
        self.code_predictor_class = CodePredictorTex(nz_feat=nz_feat, \
          tex_code_dim=self.codedim, shape_code_dim=self.codedim)

        self.pps =   nn.Parameter(torch.Tensor([ppx, ppy]).cuda())
        self.light_params = nn.Parameter(torch.Tensor([1,0,-1,0]).cuda())  # intensity, x,y,z

        # For renderering.
        self.renderer_soft = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_softflf = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        self.renderer_softflb = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_softtex = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_softpart = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,gamma_val=1e-4, 
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_hardtex = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-12, 
               camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
               light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)

        self.renderer_kp = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
       
        self.renderer_softdepth = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-3,gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.) 

        if opts.ptex:
            self.nerf_tex = nn.ModuleList([getattr(nerf_models, 'CondNeRFModel')(
                        num_encoding_fn_xyz=10,
                        num_encoding_fn_dir=4,
                        include_input_xyz=False,
                        include_input_dir=False,
                        use_viewdirs=False,
                        codesize=self.codedim) for i in range(1)])
            self.nerf_feat = nn.ModuleList([getattr(nerf_models, 'CondNeRFModel')(
                        num_encoding_fn_xyz=10,
                        num_encoding_fn_dir=4,
                        include_input_xyz=False,
                        include_input_dir=False,
                        use_viewdirs=False,
                        out_channel=self.nfeat,
                        codesize=0) for i in range(1)])
            self.encode_position_fn_tex = get_embedding_function(
                num_encoding_functions=10,
                include_input=False,
                log_sampling=True,
            )

        if opts.catemodel:
            self.nerf_mshape = nn.ModuleList([getattr(nerf_models, 'CondNeRFModel')(
                        num_encoding_fn_xyz=10,
                        num_encoding_fn_dir=4,
                        include_input_xyz=True,
                        include_input_dir=False,
                        use_viewdirs=False,
                        codesize=self.codedim) for i in range(1)])
            self.nerf_shape = nn.ModuleList([getattr(nerf_models, 'CondNeRFModel')(
                        num_encoding_fn_xyz=10,
                        num_encoding_fn_dir=4,
                        include_input_xyz=False,
                        include_input_dir=False,
                        use_viewdirs=False,
                        codesize=0) for i in range(1)])
            self.encode_position_fn_shape = get_embedding_function(
                num_encoding_functions=10,
                include_input=True,
                log_sampling=True,
            )

        self.use_kproj = False # by default don't use keypoint reprojection

    def forward(self, batch_input):
        if self.training:
            local_batch_size = batch_input['input_imgs  '].shape[0]
            for k,v in batch_input.items():
                batch_input[k] = v.view(local_batch_size//2,2,-1).permute(1,0,2).reshape(v.shape)
            self.input_imgs   = batch_input['input_imgs  ']
            self.imgs         = batch_input['imgs        ']
            self.masks        = batch_input['masks       ']
            self.cams         = batch_input['cams        ']
            self.flow         = batch_input['flow        ']
            self.ddts_barrier = batch_input['ddts_barrier']
            self.pp           = batch_input['pp          ']
            self.occ          = batch_input['occ         ']
            self.oriimg_shape = batch_input['oriimg_shape']
            self.frameid      = batch_input['frameid']
            self.dataid      = batch_input['dataid']
            self.is_canonical = batch_input['is_canonical']
            self.rtk = batch_input['rtk']
        else:
            local_batch_size = len(batch_input)
            self.input_imgs = batch_input

        img = self.input_imgs
        opts = self.opts
        if opts.debug:
            torch.cuda.synchronize()
            start_time = time.time()

        # assign instance texture and shape
        pred_v = torch.zeros(local_batch_size, 1, self.num_verts,3).cuda()
        faces = torch.zeros(local_batch_size, 1, self.num_faces,3).cuda().long()
        tex = torch.zeros_like(pred_v)
        joint_ts =  torch.zeros(local_batch_size, 1,opts.n_bones-1,3).cuda()
        ctl_ts =  torch.zeros(local_batch_size, 1,opts.n_bones-1,3).cuda()
        ctl_rs =  torch.zeros(local_batch_size, 1,opts.n_bones-1,4).cuda()
        log_ctl = torch.zeros(local_batch_size, 1,opts.n_bones-1,3).cuda()
        for i in range(local_batch_size):
            # select the first ins
            pred_v[i] = self.mean_v[0]
            tex[i] = self.tex[0].sigmoid()
            faces[i] = self.faces[0]
            joint_ts[i] = self.joint_ts[0] # id, hp, nmesh-1, 3
            ctl_ts[i] = self.ctl_ts[0] # id, hp, nmesh-1, 3
            ctl_rs[i] = self.ctl_rs[0] # id, hp, nmesh-1, 3
            log_ctl[i] = self.log_ctl[0] # id, hp, nmesh-1, 3

        pred_v = pred_v.reshape(-1,self.num_verts,3)
        tex = tex.reshape(-1,self.num_verts,3)
        faces = faces.reshape(-1,self.num_faces,3)
        joint_ts = joint_ts.view(local_batch_size,1,opts.n_bones-1,3)
        ctl_ts = ctl_ts.view(local_batch_size,1,opts.n_bones-1,3)
        ctl_rs = ctl_rs.view(local_batch_size,1,opts.n_bones-1,4)
        log_ctl = log_ctl.view(local_batch_size,1,opts.n_bones-1,3)

        if opts.debug:
            torch.cuda.synchronize()
            print('before nerf time:%.2f'%(time.time()-start_time))

        
        # replace texture with nerf here
        if opts.ptex:
            img_feat_class = self.encoder_class.forward(img)
            tex_code, shape_code = self.code_predictor_class.forward(img_feat_class)
            #freq, 2, 3 (sinx, siny, sinz, cosx, cosy, cosz)
            pred_v_symm = pred_v[:local_batch_size*1].clone().detach()
            texture_sampled = torch.zeros_like(pred_v_symm)
            feat_sampled = torch.zeros(local_batch_size*1, 
                                       self.num_verts,self.nfeat).cuda()
            for i in range(local_batch_size):
                for j in range(1):
                    nerfidx = j  # select the first 
                    sampidx = i*1+j
                    nerf_out = run_network(
                        self.nerf_tex[nerfidx],
                        pred_v_symm[sampidx:sampidx+1],
                        None,
                        131072,
                        self.encode_position_fn_tex,
                        None,
                        code=tex_code[i:i+1],
                    )
                    texture_sampled[sampidx] = nerf_out[:,:,:-1]

                    featnerf=self.csmnet.featnerf
                    nerf_out = run_network(
                        featnerf,
                        pred_v_symm[sampidx:sampidx+1],
                        None,
                        131072,
                        self.encode_position_fn_tex,
                        None,
                        code=None,
                    )
                    feat_sampled[sampidx] = nerf_out[:,:,:-1]
                    #feat_sampled[sampidx] = F.relu(nerf_out[:,:,:-1])
                    #feat_sampled[sampidx] = nerf_out[:,:,:-1]*20
                            
            tex=texture_sampled[:,:,:3] # bs, nverts, 3
            tex=tex.sigmoid()
            vfeat = feat_sampled
        if opts.debug:
            torch.cuda.synchronize()
            print('before shape nerf time:%.2f'%(time.time()-start_time))
            
        # add shape deformation at instance level
        if opts.catemodel:
            # compute mean shape
            pred_v_symm = pred_v[:local_batch_size*1].clone().detach()
            shape_delta = torch.zeros_like(pred_v_symm)
            for i in range(local_batch_size):
                for j in range(1):
                    nerfidx = j
                    sampidx = i*1+j
                    shape_delta[sampidx] = run_network(
                        self.nerf_mshape[nerfidx],
                        pred_v_symm[sampidx:sampidx+1],
                        None,
                        131072,
                        self.encode_position_fn_shape,
                        None,
                        code=shape_code[i:i+1,:],
                    )[:,:,:3]
            pred_v = pred_v + shape_delta
        
        if self.training and self.use_kproj: 
            pred_v = pred_v.detach()
            ctl_ts = ctl_ts.detach()
            if 'shape_delta' in locals():
                shape_delta = shape_delta.detach()

        def skinning(pred_v, ctl_ts, ctl_rs, log_ctl, opts, local_batch_size, num_verts):
            log_ctl[:] = 1
            skin = torch.zeros(local_batch_size,1,opts.n_bones-1, num_verts,1).cuda()
            for i in range(local_batch_size):
                dis_norm = (ctl_ts[i].view(1,-1,1,3) - pred_v.view(local_batch_size,1,-1,3)[i,:,None].detach()) # p-v, H,J,1,3 - H,1,N,3
                dis_norm = dis_norm.matmul(kornia.quaternion_to_rotation_matrix(ctl_rs[i]).view(1,-1,3,3)) # h,j,n,3
                dis_norm = log_ctl[i].exp().view(1,-1,1,3) * dis_norm.pow(2) # (p-v)^TS(p-v) 
                dis_norm = (-10 * dis_norm.sum(3))
                
                topk, indices = dis_norm.topk(3, 1, largest=True)
                res = torch.zeros_like(dis_norm).fill_(-np.inf)
                res = res.scatter(1, indices, topk)
                dis_norm = res
                skin[i] = dis_norm.softmax(1)[:,:,:,None] # h,j,n,1

            skin = skin.view(-1,opts.n_bones-1,pred_v.shape[-2],1)
            return skin

        # skin computation
        if opts.n_bones>1:
            skin = skinning(pred_v, ctl_ts, ctl_rs, log_ctl, opts, local_batch_size, self.num_verts)
        else:skin=None                

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()
        self.apply(set_bn_eval)
        self.csmnet.apply(set_bn_train)


        scale =  torch.zeros(1, local_batch_size, 2).cuda()
        ppoint = torch.zeros(1, local_batch_size, 2).cuda()
        quat =   torch.zeros(1, local_batch_size, opts.n_bones * 9).cuda()
        trans =  torch.zeros(1, local_batch_size, opts.n_bones * 2).cuda()
        depth =  torch.zeros(1, local_batch_size, opts.n_bones).cuda()
        for i in range(1):
            img_feat       = self.encoder[i].forward(img)
            scalex, transx, quatx, depthx, ppointx = self.code_predictor[i].forward(img_feat)
            
            quatx = quatx.view(-1,opts.n_bones,3,3)
            quat_body = quatx[:,:1]
            quat_body = quat_body.clone()

            quatx = torch.cat([quat_body, quatx[:,1:]],1)

            scale[i] = scalex.view(local_batch_size,-1)
            trans[i] = transx.view(local_batch_size,-1)
            quat[i] = quatx.view(local_batch_size,-1)
            depth[i] = depthx.view(local_batch_size,-1)
            ppoint[i] = ppointx.view(local_batch_size,-1)

        scale = scale.permute(1,0,2)
        ppoint=ppoint.permute(1,0,2)
        quat=quat.permute(1,0,2)
        trans=trans.permute(1,0,2)
        depth=depth.permute(1,0,2)

        # equal focal length
        scale[:,:,1] = scale[:,:,0].clone() * self.cams[:,None,1] / self.cams[:,None,0]

        if not opts.cnnpp:
            # fix pp
            ppoint = (self.pps[None,None]-self.pp[:,None])/128. * self.cams[:,None,:2] - 1
            ppoint_raw = ppoint.clone()
        if self.training and opts.cnnpp:
            ppb1 = self.cams[:local_batch_size//2,None,:2]*self.pp[:local_batch_size//2,None]/(opts.img_size/2.)
            ppb2 = self.cams[local_batch_size//2:,None,:2]*self.pp[local_batch_size//2:,None]/(opts.img_size/2.)
            ppa1 = ppoint[:local_batch_size//2] + ppb1 + 1
            ppa2 = ppa1 * (self.cams[local_batch_size//2:,None,:2] / self.cams[:local_batch_size//2,None,:2]) 
            ppoint[local_batch_size//2:]= ppa2 - ppb2 -1

        if not self.training:
            self.uncrop_scale = scale.clone() / self.cams[None] * 128
            self.uncrop_pp = (ppoint[0,0] + 1)*128/self.cams[0] + self.pp[0]
        
        # change according to intrinsics 
        quat = quat.reshape(-1,9)
        depth = depth.reshape(-1,1)
        trans = trans.reshape(-1,2)
        ppoint = ppoint.reshape(ppoint.shape)
        scale = scale.reshape(scale.shape)

        noise_rot = torch.eye(3).cuda()[None]
        ## rendering
        if self.training and opts.noise and self.epoch>2 and self.iters<100 and self.iters>1:
            # add noise
            decay_factor = 0.2*(1e-4)**(self.iters/100)
            decay_factor_r = decay_factor * np.ones(quat.shape[0])
            ### smaller noise for bones
            decay_factor_r = decay_factor_r.reshape((-1,opts.n_bones))
            rotmag = pytorch3d.transforms.quaternion_to_axis_angle(pytorch3d.transforms.matrix_to_quaternion(quat.view(-1,3,3))).norm(2,-1)/(np.pi*2)
            decay_factor_r[:,1:] *= 0
            decay_factor_r[:,0] *=  1
            decay_factor_r = decay_factor_r.flatten()
            noise = q_scale_m(q_rnd_m(b=quat.shape[0]), decay_factor_r)  # wxyz
            noise = torch.Tensor(noise).cuda()
            noise = torch.cat([noise[:,1:], noise[:,:1]],-1)
            noise_rot = kornia.quaternion_to_rotation_matrix(noise)
            quat = quat.view(-1,3,3).matmul(noise_rot).view(-1,9)

            decay_factor_s = decay_factor
            noise = (decay_factor_s*torch.normal(torch.zeros(scale.shape).cuda(),opts.rscale)).exp()
            scale = scale * noise
           
        if self.training and opts.rtk_path!='none':
            # w/ gt cam
            quat_pred = quat.clone()
            # replace with rtk
            quat = quat.view(local_batch_size, opts.n_bones,-1)
            quat[:,:1] = self.rtk[:,:3,:3].reshape(-1,1,9)
            quat = quat.view(quat_pred.shape)
    
        ## rendering
        # obj-cam rigid transform;  proj_cam: [focal, tx,ty,qw,qx,qy,qz]; 
        # 1st/2nd frame stored as 1st:[0:local_batch_size//2], 2nd: [local_batch_size//2:-1]
        # transforms [body-to-cam, part1-to-body, ...]
        Rmat = quat.clone().view(-1,3,3).permute(0,2,1)
        Tmat = torch.cat([trans, depth],1)
        joint_ts = joint_ts.view(-1,opts.n_bones-1,3,1)
        ctl_ts = ctl_ts.view(-1,opts.n_bones-1,3,1)
        if opts.n_bones>1:
            # part transform
            # Gg*Gx*Gg_inv
            Rmat = Rmat.view(-1,opts.n_bones,3,3)
            Tmat = Tmat.view(-1,opts.n_bones,3,1)
            Tmat[:,1:] = -Rmat[:,1:].matmul(joint_ts)+Tmat[:,1:]+joint_ts
            Rmat[:,1:] = Rmat[:,1:].permute(0,1,3,2)
            Rmat = Rmat.view(-1,3,3)
            Tmat = Tmat.view(-1,3)

            self.ctl_proj =    obj_to_cam(ctl_ts[:,:,:,0],  Rmat.detach(), Tmat[:,np.newaxis].detach(), opts.n_bones, 1, torch.eye(opts.n_bones-1)[None,:,:,None].cuda())
            self.ctl_proj = pinhole_cam(self.ctl_proj, ppoint.detach(), scale.detach())
            self.joint_proj =    obj_to_cam(joint_ts[:,:,:,0],  Rmat.detach(), Tmat[:,np.newaxis].detach(), opts.n_bones, 1, torch.eye(opts.n_bones-1)[None,:,:,None].cuda())
            self.joint_proj = pinhole_cam(self.joint_proj, ppoint.detach(), scale.detach())

        aug_img   = img.clone()
        gt_csmmasks = self.masks.clone()
        csm_Rmat  =Rmat.clone()
        csm_Tmat  =Tmat.clone()
        csm_skin  =skin.clone()
        csm_ppoint=ppoint.clone()
        csm_scale =scale.clone()
        
        if self.training:
            with torch.no_grad():
                csm_gt = render_multiplex(pred_v, faces, pred_v, csm_Rmat, csm_Tmat, csm_skin, csm_ppoint, csm_scale, self.renderer_hardtex, 1,opts.n_bones).detach()
                csm_mask = csm_gt[:,-1]
                csm_gt = csm_gt[:,:3]

        factor = 0
        M = torch.eye(3).cuda()[None].repeat(local_batch_size,1,1)
        if self.training and not (self.use_kproj): 
            # color aug
            if np.random.rand()>0.:
                color_aug = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                aug_img = color_aug(self.imgs)
                for i in range(local_batch_size):
                    aug_img[i,:3] = aug_img[i,:3]*gt_csmmasks[i,None] + aug_img[i].permute(1,2,0)[~self.masks[i].bool()].mean(0)[:,None,None] * (1-gt_csmmasks[i,None])
                aug_img = torch.stack([self.resnet_transform(aug_img[i]) for i in range(aug_img.shape[0])],0)

            for k in range(local_batch_size):
                M[k,:2,-1] -= opts.img_size/2.
                trnM = torch.Tensor([[1,0,(np.random.rand()-0.5)*opts.img_size*0.2], [0,1,(np.random.rand()-0.5)*opts.img_size*0.2],[0,0,1]]).cuda()
                rotM = torch.Tensor(cv2.Rodrigues(np.asarray([0,0,np.random.rand()*2*np.pi*factor]))[0]).cuda()
                shrM = kornia.get_shear_matrix2d(torch.tensor([[0., 0.]]), sx=(torch.rand(1)-0.5)*np.pi*0.5, sy=(torch.rand(1)-0.5)*np.pi*0.5)[0].cuda()
                rotM = rotM.matmul(shrM)
                scaM = torch.Tensor([[np.exp(np.random.normal(0,0.1)),0,0], [0,np.exp(np.random.normal(0,0.1)),0], [0,0,1]]).cuda()
                M[k] = trnM.matmul(M[k].inverse().matmul(scaM.matmul(rotM).matmul(M[k])))

                if np.random.rand()>0.5:
                    # rand mask
                    rct = (np.random.rand(2) *      opts.img_size).astype(int)
                    rsz = ((np.random.rand(2)*0.2+0.1) *opts.img_size).astype(int)
                    lb = np.minimum(np.maximum(rct-rsz, [0,0]), [opts.img_size, opts.img_size])
                    ub = np.minimum(np.maximum(rct+rsz, [0,0]), [opts.img_size, opts.img_size])
                    aug_img[k,:,lb[0]:ub[0], lb[1]:ub[1]] = torch.Tensor(np.random.rand(3)).cuda()[:,None,None]
                    csm_mask[k,lb[0]:ub[0], lb[1]:ub[1]] = -0.1
        aug_img = kornia.warp_perspective(aug_img, M, dsize=(opts.img_size, opts.img_size)).clone()

        if self.training and (not opts.finetune) and self.epoch<5: csmdetach=True
        else: csmdetach=False
        csm_pred, icsm_pred, icsm_pts, _, _ = self.csmnet.forward(aug_img, gt_csmmasks, 
                 pred_v, faces, self.resnet_transform, detach=csmdetach)
        # don't update feature through reprojection loss
        csm_pred_clean, _, _, _, csmfb_w_clean = self.csmnet.forward(img,
                    self.masks, pred_v, faces, self.resnet_transform, detach=True)
           
        #self.feat = csm_feat.detach()
        #featfac = float(opts.img_size)/self.feat.shape[2]
        Minverse=M.inverse()
        #Minverse_sub = Minverse.clone()
        #Minverse_sub[:,:2,-1] = Minverse_sub[:,:2,-1] / featfac
        #self.feat = kornia.warp_perspective(self.feat, Minverse_sub, dsize=(self.feat.shape[2],self.feat.shape[3]))
        #featdim = csm_feat.shape[2]
            
        csm_pred = kornia.warp_perspective(csm_pred, Minverse, dsize=(opts.img_size, opts.img_size))
        icsm_pred = (icsm_pred+1)*128
        icsm_pred = Minverse[:,:2,:2].matmul(icsm_pred) + Minverse[:,:2,-1:]
        icsm_pred = icsm_pred/128-1

        # for output at test time
        self.csm_pred = csm_pred_clean
        self.csm_conf = csmfb_w_clean


        if not self.training:
           return [scale, trans, quat, depth, ppoint], pred_v, faces, tex, skin, Rmat, Tmat, vfeat
        

        self.deform_v = obj_to_cam(pred_v, Rmat.view(-1,3,3), Tmat[:,np.newaxis,:],opts.n_bones, 1,skin,tocam=False)


        if opts.debug:
            torch.cuda.synchronize()
            print('before rend time:%.2f'%(time.time()-start_time))
        
        # 1) flow rendering 
        verts_fl = obj_to_cam(pred_v, Rmat, Tmat[:,np.newaxis,:],opts.n_bones, 1,skin)
        verts_fl = torch.cat([verts_fl,torch.ones_like(verts_fl[:, :, 0:1])], dim=-1)
        verts_pos0 = verts_fl.view(local_batch_size,1,-1,4)[:local_batch_size//2].clone().view(local_batch_size//2*1,-1,4)
        verts_pos1 = verts_fl.view(local_batch_size,1,-1,4)[local_batch_size//2:].clone().view(local_batch_size//2*1,-1,4)
        verts_fl = pinhole_cam(verts_fl, ppoint, scale)

        dmax=verts_fl[:,:,-2].max()
        dmin=verts_fl[:,:,-2].min()
        self.renderer_softflf.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softflf.rasterizer.far= dmax+(dmax-dmin)/2
        self.renderer_softflb.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softflb.rasterizer.far= dmax+(dmax-dmin)/2
        self.renderer_softtex.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softtex.rasterizer.far= dmax+(dmax-dmin)/2
        if opts.finetune and '-ft' in opts.name:
            self.renderer_soft.rasterizer.sigma_val= 1e-5
            self.renderer_softflf.rasterizer.sigma_val= 1e-5
            self.renderer_softflb.rasterizer.sigma_val= 1e-5
            self.renderer_softtex.rasterizer.sigma_val= 1e-5

        self.renderer_kp.rasterizer.sigma_val = reg_decay(self.iters, self.ep_iters, 1e-5, 1e-3)

        self.flow_fw, self.bgmask_fw, self.fgmask_flowf = render_flow_soft_3(self.renderer_softflf,
                verts_fl.view(local_batch_size,1,-1,4)[:local_batch_size//2].view(-1,verts_fl.shape[1],4),
                verts_fl.view(local_batch_size,1,-1,4)[local_batch_size//2:].view(-1,verts_fl.shape[1],4),
                faces.view(local_batch_size,1,-1,3)[:local_batch_size//2].view(-1,faces.shape[1],3))
        self.flow_bw, self.bgmask_bw, self.fgmask_flowb = render_flow_soft_3(self.renderer_softflb, 
                verts_fl.view(local_batch_size,1,-1,4)[local_batch_size//2:].view(-1,verts_fl.shape[1],4),
                verts_fl.view(local_batch_size,1,-1,4)[:local_batch_size//2].view(-1,verts_fl.shape[1],4),
                faces.view(local_batch_size,1,-1,3)[local_batch_size//2:].view(-1,faces.shape[1],3))
        self.bgmask =  torch.cat([self.bgmask_fw, self.bgmask_bw],0) 
        self.fgmask_flow =  torch.cat([self.fgmask_flowf, self.fgmask_flowb],0) 
        self.flow_rd = torch.cat([self.flow_fw, self.flow_bw    ],0) 
        if opts.debug:
            torch.cuda.synchronize()
            print('after flow rend time:%.2f'%(time.time()-start_time))
              
        # 2) silhouette
        Rmat_mask = Rmat.clone().view(-1,opts.n_bones,3,3)
        Rmat_mask = torch.cat([Rmat_mask[:,:1], Rmat_mask[:,1:]],1).view(-1,3,3)
        if not opts.finetune:
            Rmat_mask = Rmat_mask.detach()
        verts_mask = obj_to_cam(pred_v, Rmat_mask, Tmat[:,np.newaxis,:],opts.n_bones, 1,skin)
        verts_mask = torch.cat([verts_mask,torch.ones_like(verts_mask[:, :, 0:1])], dim=-1)
        verts_mask = pinhole_cam(verts_mask, ppoint, scale)

        # softras
        offset = torch.Tensor( self.renderer_soft.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
        verts_pre = verts_mask[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
        self.mask_pred = self.renderer_soft.render_mesh(sr.Mesh(verts_pre,faces))[:,-1]


        if opts.debug:
            torch.cuda.synchronize()
            print('after sil rend time:%.2f'%(time.time()-start_time))

        if opts.opt_tex=='yes':
            if not opts.finetune and self.epoch<5: 
                Rmat_tex = Rmat.clone().view(-1,opts.n_bones,3,3)
                Rmat_tex = torch.cat([Rmat_tex[:,:1].detach(), Rmat_tex[:,1:]],1).view(-1,3,3)
            else:
                Rmat_tex = Rmat.clone().view(-1,3,3)
            verts_tex = obj_to_cam(pred_v, Rmat_tex, Tmat[:,np.newaxis,:],opts.n_bones, 1,skin)
            verts_tex = torch.cat([verts_tex,torch.ones_like(verts_tex[:, :, 0:1])], dim=-1)
            verts_tex = pinhole_cam(verts_tex, ppoint, scale)
            offset = torch.Tensor( self.renderer_softtex.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
            verts_pre = verts_tex[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
            self.renderer_softtex.rasterizer.background_color = [1,1,1]
            self.texture_render = self.renderer_softtex.render_mesh(sr.Mesh(verts_pre, faces, textures=tex,  texture_type=self.texture_type)).clone()
            self.fgmask_tex = self.texture_render[:,-1]
            self.texture_render = self.texture_render[:,:3]
            img_obs = self.imgs[:]*(self.masks[:]>0).float()[:,None]
            img_rnd = self.texture_render*(self.fgmask_tex)[:,None]
            img_white = 1-(self.masks[:]>0).float()[:,None] + img_obs

        if opts.n_bones>1 and self.iters==0:
            # part rendering
            colormap = torch.Tensor(citylabs[:opts.n_bones-1]).cuda() # 5x3
            skin_colors = (skin[0] * colormap[:,None]).sum(0)/256.
            self.part_render = self.renderer_softpart.render_mesh(sr.Mesh(verts_pre.view(local_batch_size,1,-1,3)[:1,0].detach(), faces[:1], textures=skin_colors[None], texture_type='vertex'))[:,:3].detach()

        # depth rendering
        verts_depth = obj_to_cam(pred_v, Rmat.detach(), Tmat[:,np.newaxis,:].detach(),opts.n_bones, 1,skin.detach())
        verts_depth = 1./verts_depth
        self.depth_render = self.renderer_softdepth.render_mesh(sr.Mesh(verts_pre, faces, textures=verts_depth,  texture_type=self.texture_type)).clone()
        depth_mask = self.depth_render[:,3]
        self.depth_render = self.depth_render[:,2]
        try:
            self.depth_render[depth_mask==0] = self.depth_render[depth_mask>0].min()
        except:
            pass#pdb.set_trace()

        
        if opts.debug:
            torch.cuda.synchronize()
            print('after tex+part render time:%.2f'%(time.time()-start_time))
        
        # pixel weights
        weight_mask = torch.ones(local_batch_size, opts.img_size, opts.img_size).cuda()
        weight_mask[self.occ==0] = 0

        # 1) mask loss
        mask_pred = self.mask_pred.view(local_batch_size,-1,opts.img_size, opts.img_size)
        self.mask_loss_sub = (mask_pred - self.masks[:,None]).pow(2)

        if not opts.finetune:
            self.mask_loss_sub = 0
            for i in range (5): # 256,128,64,32,16
                diff_img = (F.interpolate(mask_pred         , scale_factor=(0.5)**i,mode='area')
                          - F.interpolate(self.masks[:,None], scale_factor=(0.5)**i,mode='area')).pow(2)
                self.mask_loss_sub += F.interpolate(diff_img, mask_pred.shape[2:4])
            self.mask_loss_sub *= 0.2
        
        tmplist = torch.zeros(local_batch_size, 1).cuda()
        for i in range(local_batch_size):
            for j in range(1):
                tmplist[i,j] = (self.mask_loss_sub[i,j]*weight_mask).mean()
        self.mask_loss_sub = opts.sil_wt * tmplist
        self.mask_loss = self.mask_loss_sub.mean()  # get rid of invalid pixels (out of border)
        self.total_loss = self.mask_loss.clone()

        # 2) flow loss
        flow_rd = self.flow_rd.view(local_batch_size,-1,opts.img_size, opts.img_size,2)
        mask = (~self.bgmask).view(local_batch_size,-1,opts.img_size, opts.img_size) & ((self.occ!=0)[:,None] &  (self.masks[:]>0) [:,None]).repeat(1,1,1,1)
        self.flow_rd_map = torch.norm((flow_rd-self.flow[:,None,:2].permute(0,1,3,4,2)),2,-1)
        self.vis_mask = mask.clone()
        weights_flow = (-self.occ).sigmoid()[:,None].repeat(1,1,1,1)
        weights_flow = weights_flow / weights_flow[mask].mean()
        self.flow_rd_map = self.flow_rd_map * weights_flow
    
        tmplist = torch.zeros(local_batch_size, 1).cuda()
        for i in range(local_batch_size):
            for j in range(1):
                tmplist[i,j] = self.flow_rd_map[i,j][mask[i,j]].mean()
                if mask[i,j].sum()==0: tmplist[i,j]=0
        self.flow_rd_loss_sub = tmplist
    
        self.flow_rd_loss = self.flow_rd_loss_sub.mean() * opts.flow_wt
        self.total_loss += self.flow_rd_loss
   
        # feature consistency loss
        #self.feat = self.feat.view(local_batch_size, -1, featdim, featdim)
        #self.renderer_kp.rasterizer.image_size=featdim
        #if not opts.finetune and self.epoch<5: 
        #    feats_render, mask_render = render_feature(pred_v.detach(), faces, vfeat, Rmat.detach(), Tmat.detach(), skin.detach(), ppoint.detach(), scale.detach(),\
        #        self.renderer_kp, local_batch_size, opts.n_mesh)
        #else:
        #    feats_render, mask_render = render_feature(pred_v.detach(), faces, vfeat, Rmat, Tmat, skin, ppoint, scale,\
        #        self.renderer_kp, local_batch_size, opts.n_mesh)
        #mask_obs =((self.occ!=0)[:,None] &  (self.masks[:]>0) [:,None]).float()
        #mask_obs = F.interpolate(mask_obs, self.feat.shape[2:], mode='nearest')
        #feat_obs = self.feat[:,:feats_render.shape[1]]
        #feat_mask = mask_obs*mask_render[:,None] > 0
        #self.feat_loss = (F.normalize(feat_obs, 2,1) * \
        #                  F.normalize(feats_render, 2,1)).sum(1)
        #self.feat_loss = (1-(self.feat_loss+1)/2) 
        #self.feat_loss = self.feat_loss[feat_mask[:,0]].mean()
  
        # 3) texture loss
        if opts.opt_tex=='yes':
            imgobs_rep = img_obs[:,None].repeat(1,1,1,1,1).view(-1,3,opts.img_size,opts.img_size)
            imgwhite_rep = img_white[:,None].repeat(1,1,1,1,1).view(-1,3,opts.img_size,opts.img_size)
            obspair = torch.cat([imgobs_rep, imgwhite_rep],0) 
            rndpair = torch.cat([img_rnd, self.texture_render],0) 
    
            tmplist = torch.zeros(local_batch_size, 1).cuda()
            for i in range(local_batch_size):
                for j in range(1):
                    #tmplist[i,j] += ((img_obs[i] - img_rnd.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j]).abs().mean(0)*weight_mask[i]).mean()  
                    tmplist[i,j] += 0.75*((img_obs[i] - img_rnd.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j]).pow(2).sum(0)*weight_mask[i]).mean()  
                    tmplist[i,j] += ((img_white[i] - self.texture_render.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j]).abs().mean(0)*weight_mask[i]).mean()  
            if self.use_kproj and self.reinit_bones:
                percept_loss = self.ptex_loss.forward_pair(2*obspair-1, 2*rndpair-1, reweight=True)
            else:
                percept_loss = self.ptex_loss.forward_pair(2*obspair-1, 2*rndpair-1)
            tmplist +=  0.005*percept_loss.view(2,-1).sum(0).view(local_batch_size,1)
            self.texture_loss_sub = opts.tex_wt*tmplist
            self.texture_loss = self.texture_loss_sub.mean()
    
            self.total_loss += self.texture_loss

        # 4) shape smoothness
        factor=int(opts.n_faces)/1280
        if not opts.finetune:
            factor = reg_decay(self.epoch, opts.num_epochs, 0.1, 1)
        else: 
            factor = reg_decay(self.epoch, opts.num_epochs, 0.05, 0.5)
   
        self.triangle_loss_sub = torch.zeros(2*1*local_batch_size//2).cuda()
        for idx in range(local_batch_size): 
            predv_batch = self.deform_v[idx*1:(idx+1)*1]
            self.triangle_loss_sub[idx*1:(idx+1)*1] = factor*opts.triangle_reg_wt*\
                self.triangle_loss_fn_sr(predv_batch)*(4**opts.subdivide)/64.
            self.triangle_loss_sub[idx*1:(idx+1)*1] +=factor*0.1*opts.triangle_reg_wt*\
                self.flatten_loss(predv_batch)*(2**opts.subdivide/8.0)
        self.triangle_loss_sub = self.triangle_loss_sub.view(local_batch_size,1)
        self.triangle_loss = self.triangle_loss_sub.mean()
        self.total_loss += self.triangle_loss

        # 5) matching loss
        mask = (csm_mask>0) & ((self.occ!=0) &  (gt_csmmasks>0))

        if opts.local_rank==0 and mask.sum()==0:pdb.set_trace()
        self.match_loss = (csm_pred - csm_gt).norm(2,1)[mask].mean() * 0.1

        ## icsm match loss
        icsm_skin = skinning(icsm_pts, ctl_ts, ctl_rs, log_ctl, opts, local_batch_size, icsm_pts.shape[-2])
        icsm_proj = obj_to_cam(icsm_pts, Rmat, Tmat[:,None,:],opts.n_bones, 1,icsm_skin)
        icsm_depth = icsm_proj[:,:,2].clone()
        icsm_proj = torch.cat([icsm_proj,torch.ones_like(icsm_proj[:, :, 0:1])], dim=-1)
        icsm_proj = pinhole_cam(icsm_proj, ppoint, scale)
        icsm_proj = icsm_proj.detach()[:,:,:2].permute(0,2,1)

        ## need to weight by depth
        icsm_depthgt = F.grid_sample(1./ self.depth_render[:,None], icsm_proj.permute(0,2,1)[:,None])[:,0,0]
        depth_weight = (-F.relu(icsm_depth - icsm_depthgt)).exp().detach()
        self.imatch_loss = ((icsm_pred - icsm_proj).norm(2,1) * depth_weight).mean() * 0.05
        # visualize matches
        imatch_vis = self.imgs[0].permute(1,2,0).flip(-1).cpu().numpy() * 255
        depthw_vis = self.imgs[0].permute(1,2,0).flip(-1).cpu().numpy() * 255
        for point in icsm_pred[0].T:
            point = point.detach().cpu().numpy()
            point = (point+1) * 128
            cv2.circle(imatch_vis,tuple(point),1,(0,0,255))
        for point in icsm_proj[0].T:
            point = point.detach().cpu().numpy()
            point = (point+1) * 128
            cv2.circle(imatch_vis,tuple(point),1,(255,0,0))
        for i in range(len(icsm_proj[0].T)):
            point = icsm_proj[0,:,i].detach().cpu().numpy()
            point = (point+1) * 128
            c_den = int(depth_weight[0,i]*255)
            cv2.circle(depthw_vis,tuple(point),1,(c_den,c_den,c_den))
        self.imatch_vis = imatch_vis/255
        self.depthw_vis = depthw_vis/255
        #cv2.imwrite('0.png', self.imatch_vis*255)
        self.imatch_vis = self.imatch_vis[:,:,::-1]
        self.depthw_vis = self.depthw_vis[:,:,::-1]
        
        # 6) reprojection loss
        self.kreproj_loss = torch.zeros(local_batch_size).cuda()
        for i in range(local_batch_size):
            reproj_mask = self.masks[i]>0
            csm_points = csm_pred_clean[i].permute(1,2,0)[reproj_mask].detach()[None]# bs, h, w, 3
            dis_norm = (ctl_ts[i].view(1,-1,1,3) - csm_points[None]) # p-v, H,J,1,3 - H,1,N,3
            dis_norm = dis_norm.matmul(kornia.quaternion_to_rotation_matrix(ctl_rs[i]).view(1,-1,3,3)) # h,j,n,3
            dis_norm = log_ctl[i].exp().view(1,-1,1,3) * dis_norm.pow(2) # (p-v)^TS(p-v) 
            skin_points = (-10 * dis_norm.sum(3)).softmax(1)[:,:,:,None] # h,j,n,1
            skin_points = skin_points.detach()
            Rmat_points = Rmat[i*opts.n_bones:(i+1)*opts.n_bones] 
            Tmat_points = Tmat[i*opts.n_bones:(i+1)*opts.n_bones]
            csm_reproj = obj_to_cam(csm_points, Rmat_points, 
                                                Tmat_points[:,None,:],opts.n_bones, 1,skin_points)
            csm_reproj = torch.cat([csm_reproj,torch.ones_like(csm_reproj[:, :, 0:1])], dim=-1)
            csm_reproj = pinhole_cam(csm_reproj, ppoint[i:i+1], scale[i:i+1])
            
            reproj_gt = torch.flip((torch.vstack(torch.where(reproj_mask))/opts.img_size*2-1),[0]).T
            kreproj_loss = (csm_reproj[0,:,:2] - reproj_gt).norm(2,-1)
            kreproj_loss = kreproj_loss * csmfb_w_clean[i,0][reproj_mask]
            kreproj_loss = kreproj_loss.mean()
            if torch.isnan(kreproj_loss): pdb.set_trace()
            self.kreproj_loss[i] = kreproj_loss

        self.kreproj_loss = self.kreproj_loss.mean()
        if self.training and self.use_kproj: 
            self.total_loss = self.total_loss * 0 + self.kreproj_loss  * 1
            self.total_loss += self.match_loss  * 0.0
            self.total_loss += self.imatch_loss  * 0.0
        else:
            self.total_loss += self.kreproj_loss  * 0.0
            self.total_loss += self.match_loss  * 1
            self.total_loss += self.imatch_loss  * 1


        # vis
        meanvmin = csm_pred.min()
        meanvmax = csm_pred.max()
        csm_vis = (csm_pred - meanvmin)/(meanvmax-meanvmin)
        csm_vis_gt = (csm_gt - meanvmin)/(meanvmax-meanvmin)
        csm_vis[gt_csmmasks[:,None].repeat(1,3,1,1)<=0] = 0
        self.cost_rd = torch.cat([csm_mask[:1,None].repeat(1,3,1,1), aug_img[:1], csm_vis_gt[:1], csm_vis[:1]],3)
   
        # 7) shape deformation loss
        if opts.n_bones>1:
            # bones
            self.bone_rot_l1 =  compute_geodesic_distance_from_two_matrices(
                        quat.view(-1,1,opts.n_bones,9)[:,:,1:].reshape(-1,3,3), 
             torch.eye(3).cuda().repeat(local_batch_size*1*(opts.n_bones-1),1,1)).mean() # small rotation
            self.bone_trans_l1 = torch.cat([trans,depth],1).view(-1,1,opts.n_bones,3)[:,:,1:].abs().mean()
            if not opts.finetune:
                factor = reg_decay(self.epoch, opts.num_epochs, 0.1, 1.0)
            else: 
                factor=0.001
            self.lmotion_loss_sub = factor*(self.deform_v - pred_v).norm(2,-1).mean(-1).view(local_batch_size,1)
            self.lmotion_loss = self.lmotion_loss_sub.mean()
            self.total_loss += self.lmotion_loss * opts.l1_wt
    
            # skins
            shape_samp = pytorch3d.ops.sample_points_from_meshes(pytorch3d.structures.meshes.Meshes(verts=pred_v, faces=faces), 1000, return_normals=False).detach()
            from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
            samploss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            self.skin_ent_loss = samploss(ctl_ts[:,:,:,0], shape_samp).mean()  # By default, use constant weights = 1/number of samples
            if opts.finetune:
                factor = 0.1
            else:
                factor = 0.
            self.total_loss += self.skin_ent_loss * factor
            self.arap_loss = torch.zeros(local_batch_size).cuda()
            for i in range(local_batch_size):
                self.arap_loss[i] = self.arap_loss_fn(self.deform_v[i:i+1], pred_v[i:i+1]).mean()*(4**opts.subdivide)/64.
                self.arap_loss[i] += (mesh_area(self.deform_v[i], faces[i]) - mesh_area(pred_v[i], faces[i])).abs().mean()
            self.arap_loss = self.arap_loss.mean()
            self.total_loss += opts.arap_wt * self.arap_loss * 10

        # 8) camera loss / smoothness
        if opts.rtk_path!='none':
            self.cam_loss = compute_geodesic_distance_from_two_matrices(quat.view(-1,3,3), quat_pred.view(-1,3,3)).mean()
            self.cam_loss = 0.2 * self.cam_loss
        else:
            self.smooth_sub = compute_geodesic_distance_from_two_matrices(quat.view(-1,1,opts.n_bones,9)[:local_batch_size//2,:].view(-1,3,3),
                                                                           quat.view(-1,1,opts.n_bones,9)[local_batch_size//2:,:].view(-1,3,3)).view(-1,1,opts.n_bones)
            self.cam_loss =  0.*self.smooth_sub.mean()
            #self.cam_loss =  0.01*self.smooth_sub.mean()
            #if opts.n_bones>1:
            #    self.cam_loss += 0.01*(trans.view(-1,1,opts.n_bones,2)[:local_batch_size//2,:,1:] - 
            #                  trans.view(-1,1,opts.n_bones,2)[local_batch_size//2:,:,1:]).abs().mean()
            #    self.cam_loss += 0.01*(depth.view(-1,1,opts.n_bones,1)[:local_batch_size//2,:,1:] - 
            #                  depth.view(-1,1,opts.n_bones,1)[local_batch_size//2:,:,1:]).abs().mean()
        self.total_loss += self.cam_loss

        # 9) aux losses
        # pull far away from the camera center
        self.total_loss += 0.02*F.relu(2-Tmat.view(-1, 1, opts.n_bones, 3)[:,:,:1,-1]).mean()
        
        # 10) texture loss between consecutive frames
        if opts.ptex:
            self.nerf_tex_loss = (texture_sampled[:local_batch_size//2*1] - texture_sampled[local_batch_size//2*1:])[:,:,:3].norm(1,-1).mean()
            self.total_loss += 0.2*self.nerf_tex_loss * 0.01

        # 11) reset shape across videos 
        if opts.catemodel:
            # pairwise loss
            self.nerf_shape_loss = []
            for i in set(np.asarray(self.dataid.cpu())):
                subid = np.asarray(torch.where(self.dataid==i)[0].cpu())
                # pairwise distance
                import itertools
                for j,k in itertools.combinations(subid, r=2):
                    self.nerf_shape_loss.append( (shape_delta[j*1:(j+1)*1] \
                                                 -shape_delta[k*1:(k+1)*1]).norm(2,-1))
            self.nerf_shape_loss = torch.stack(self.nerf_shape_loss,0).mean()
            factor = reg_decay(self.epoch, opts.num_epochs, 2, 0.02)
            print(factor)
            self.total_loss += factor*self.nerf_shape_loss
            self.l1_deform_loss = shape_delta.norm(2,-1).mean()
            self.total_loss += 0.02*self.l1_deform_loss

        if opts.debug:
            torch.cuda.synchronize()
            print('forward time:%.2f'%(time.time()-start_time))

        aux_output={}
        aux_output['flow_rd_map'] = self.flow_rd_map
        aux_output['flow_rd'] = self.flow_rd
        aux_output['vis_mask'] = self.vis_mask
        aux_output['mask_pred'] = self.mask_pred
        aux_output['total_loss'] = self.total_loss
        aux_output['mask_loss'] = self.mask_loss
        aux_output['texture_loss'] = self.texture_loss
        aux_output['flow_rd_loss'] = self.flow_rd_loss
        aux_output['skin_ent_loss'] = self.skin_ent_loss
        aux_output['arap_loss'] = self.arap_loss
        try:
            aux_output['cam_loss'] = self.cam_loss.mean()
        except: pass

        try:
            aux_output['match_loss'] = self.match_loss
            aux_output['imatch_loss'] = self.imatch_loss
        except: pass
        
        try:
            aux_output['kreproj_loss'] = self.kreproj_loss
        except: pass
        
        try:
            aux_output['depth_loss'] = self.depth_loss
            aux_output['flow_s_obs'] = flow_obs
            aux_output['flow_s'] = flow_s
            aux_output['mask_cost'] = mask_cost
            aux_output['score_s'] = score_s
        except: pass
        try:
            aux_output['flow_s_loss'] = self.flow_s_loss
        except: pass
        try:
            aux_output['feat_loss'] = self.feat_loss
            aux_output['rank_loss'] = self.rank_loss
        except: pass
        try:    
            aux_output['kp'] = self.kp
            aux_output['kp_loss'] = self.kp_loss
            aux_output['kp_pred'] = self.kp_pred
        except: pass
        aux_output['triangle_loss'] = self.triangle_loss
        if opts.n_bones>1:
            aux_output['lmotion_loss'] = self.lmotion_loss
        try: aux_output['nerf_tex_loss'] = self.nerf_tex_loss
        except: pass
        try: aux_output['l1_deform_loss'] = self.l1_deform_loss
        except: pass
        
        else:
            aux_output['current_nscore'] = self.texture_loss_sub.mean(0) + self.flow_rd_loss_sub.mean(0) + self.mask_loss_sub.mean(0)
        if 1 > 1:
            for ihp in range(1):
                aux_output['mask_hypo_%d'%(ihp)] = self.mask_loss_sub[:,ihp].mean()
                aux_output['flow_hypo_%d'%(ihp)] = self.flow_rd_loss_sub[:,ihp].mean()
                aux_output['tex_hypo_%d'%(ihp)] = self.texture_loss_sub[:,ihp].mean()
        try:
            aux_output['texture_render'] = self.texture_render
            aux_output['ctl_proj'] = self.ctl_proj
            aux_output['joint_proj'] = self.joint_proj
            aux_output['part_render'] = self.part_render
            aux_output['cost_rd'] = self.cost_rd
            aux_output['texture_render_hr'] = self.texture_render_hr
            aux_output['orth_loss'] = self.orth_loss
        except:pass
        try:
            aux_output['nerf_shape_loss'] = self.nerf_shape_loss
        except: pass
        return self.total_loss, aux_output
