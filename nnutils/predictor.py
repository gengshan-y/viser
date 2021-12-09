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
Takes an image, returns stuff.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import scipy.misc
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.io as sio

from nnutils.mesh_net import LASR
from nnutils import geom_utils
import pdb
import kornia
import soft_renderer as sr
from nnutils.geom_utils import label_colormap, obj_to_cam, pinhole_cam, \
                                render_multiplex
import configparser
citylabs = label_colormap()
import trimesh

from pytorch3d.renderer.mesh import TexturesAtlas, TexturesUV, TexturesVertex
from pytorch3d.structures.meshes import Meshes

# These options are off by default, but used for some ablations reported.
class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts
        self.symmetric = opts.symmetric

        img_size = (opts.img_size, opts.img_size)
        print('Setting up model..')
        self.model = LASR(img_size, opts, nz_feat=opts.nz_feat)

        self.load_network(self.model, self.opts.model_path)
        self.model.eval()
        self.model = self.model.cuda()

        self.renderer_softtex = sr.SoftRenderer(image_size=opts.img_size,  
                       camera_mode='look_at',perspective=False,
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        self.renderer_softpart = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,gamma_val=1e-4, 
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)

        # pytorch3d 
        from pytorch3d.renderer.mesh.shader import (
             BlendParams,
         )
        from pytorch3d.renderer import (
         PointLights, 
         RasterizationSettings, 
         MeshRenderer, 
         MeshRasterizer,  
         SoftPhongShader,
         SoftSilhouetteShader,
         )
        from pytorch3d.renderer.cameras import OrthographicCameras
        device = torch.device("cuda:0") 
        cameras = OrthographicCameras(device = device)
        lights = PointLights(
            device=device,
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((1.0, 1.0, 1.0),),
            specular_color=((1.0, 1.0, 1.0),),
        )
        self.renderer_pyr = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=opts.img_size,cull_backfaces=True)),
                shader=SoftPhongShader(device = device,cameras=cameras, lights=lights, blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)))
        )

        faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_network(self, network, save_path):
        print('loading {}..'.format(save_path))
        states = torch.load(save_path)
        insid=0
        idx=0
        optim_cam=0
      
        nfeat=states['code_predictor.0.quat_predictor.pred_layer.weight'].shape[-1]

        states['mean_v'] = states['mean_v'].view(-1,1,states['mean_v'].shape[-2],3)[insid][optim_cam:optim_cam+1]
        self.model.faces = states['faces'][insid:insid+1, optim_cam:optim_cam+1]
        self.model.num_verts = states['mean_v'].shape[-2]
        self.model.num_faces = states['faces'].shape[-2]

        for k in states.keys():
            if 'encoder.%d'%(optim_cam) in k:
                states[k.replace('encoder.%d.'%(optim_cam), 'encoder.0.')] = states[k]
            if 'code_predictor.%d'%(optim_cam) in k:
                states[k.replace('code_predictor.%d.'%(optim_cam), 'code_predictor.0.')] = states[k]
            if 'nerf_tex.%d'%(idx) in k:
                states[k.replace('nerf_tex.%d.'%(idx), 'nerf_tex.0.')] = states[k]
            if 'nerf_feat.%d'%(idx) in k:
                states[k.replace('nerf_feat.%d.'%(idx), 'nerf_feat.0.')] = states[k]
            if 'nerf_shape.%d'%(idx) in k:
                states[k.replace('nerf_shape.%d.'%(idx), 'nerf_shape.0.')] = states[k]
            if 'nerf_mshape.%d'%(idx) in k:
                states[k.replace('nerf_mshape.%d.'%(idx), 'nerf_mshape.0.')] = states[k]
        
        if 'tex' in states.keys():
            states['tex'] = states['tex'][insid][optim_cam:optim_cam+1]
            self.model.tex.data = states['tex']
            del states['tex']

        if 'ctl_rs' in states.keys():
            states['ctl_rs'] =   states['ctl_rs'][insid:insid+1,optim_cam:optim_cam+1]
            states['ctl_ts'] =   states['ctl_ts'][insid:insid+1,optim_cam:optim_cam+1]
            states['joint_ts'] =   states['joint_ts'][insid:insid+1,optim_cam:optim_cam+1]
            states['log_ctl'] = states['log_ctl'][insid:insid+1,optim_cam:optim_cam+1]

        # load mesh
        self.model.mean_v.data = states['mean_v']
        del states['mean_v']

        # load basis
        if 'shape_code_fix' in states.keys():
            self.model.shape_code_fix.data = states['shape_code_fix'].view(-1,1, states['shape_code_fix'].shape[-1])[insid:insid+1,optim_cam:optim_cam+1]
            del states['shape_code_fix']

        network.load_state_dict(states, strict=False)
        return

    def set_input(self, batch):
        opts = self.opts

        # original image where texture is sampled from.
        img_tensor = batch['img'].clone().type(torch.FloatTensor)
        imgn_tensor = batch['imgn'].clone().type(torch.FloatTensor)
        mask_tensor = batch['mask'].clone().type(torch.IntTensor)

        # input_img is the input to resnet
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        input_imgn_tensor = batch['imgn'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
            input_imgn_tensor[b] = self.resnet_transform(input_imgn_tensor[b])

        self.input_imgs = Variable(
            input_img_tensor.cuda(), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(), requires_grad=False)
        self.input_imgns = Variable(
            input_imgn_tensor.cuda(), requires_grad=False)
        self.imgns = Variable(
            imgn_tensor.cuda(), requires_grad=False)
        self.masks = Variable(
            mask_tensor.cuda(), requires_grad=False)
        self.pps = Variable(
            batch['pps'].clone().type(torch.FloatTensor).cuda(), requires_grad=False)
        self.ppsn = Variable(
            batch['ppsn'].clone().type(torch.FloatTensor).cuda(), requires_grad=False)

    def predict(self, batch,alp,frameid):
        """
        batch has B x C x H x W numpy
        """
        opts = self.opts
        self.set_input(batch)
        self.frameid = frameid
        self.cams = torch.Tensor(np.asarray([[1./alp[0], 1/alp[1]]])).cuda()
        self.forward()
        return self.collect_outputs()

    def forward(self):
        opts = self.opts
        self.model.frameid=torch.Tensor([self.frameid, self.frameid+1]).long().cuda()
        self.model.dataid = torch.Tensor([0]).long().cuda()
        self.model.imgs = self.imgs
        self.model.masks = self.masks
        self.model.cams = self.cams
        self.model.pp = torch.cat([self.pps, self.ppsn],0)
        
        # change to two-frame inputs
        self.input_imgs = torch.cat([self.input_imgs, self.input_imgns],0)
        self.model.dataid = torch.Tensor([0,0]).long().cuda()

        pred_codes, pred_v, faces, tex, skin, Rmat, Tmat, vfeat = self.model(self.input_imgs)
        scale, trans, quat, depth, ppoint = pred_codes
       
        self.depth = depth
        self.ppoint = ppoint
        self.scale = scale
        self.pred_v = pred_v
        self.tex = tex
        self.faces = faces

        print('focal: %f / depth: %f / ppx: %f / ppy: %f'%(scale[0,0,0],depth[0,0], ppoint[0,0,0], ppoint[0,0,1]))
        print(Tmat[:1])

        # create vis for skins
        if opts.n_bones>1:
            sphere_list = []
            sphere = trimesh.creation.uv_sphere(radius=0.05,count=[16, 16])
            for i in range(opts.n_bones-1):
                sphere_verts = sphere.vertices
                sphere_verts = sphere_verts / np.asarray((0.5*self.model.log_ctl.clamp(-2,2)).exp()[0,0,i,None].cpu())
                sphere_verts = sphere_verts.dot(np.asarray(kornia.quaternion_to_rotation_matrix(self.model.ctl_rs[0,0,i]).cpu()).T)
                sphere_verts = sphere_verts+np.asarray(self.model.ctl_ts[0,0,i,None].cpu())
                sphere_list.append( trimesh.Trimesh(vertices = sphere_verts, faces=sphere.faces) )
            self.sphere = trimesh.util.concatenate(sphere_list)
            # skin
            dis_norm = (self.model.ctl_ts[0,0,:,None] - torch.Tensor(self.sphere.vertices)[None].cuda()) # p-v, J,1,3 - 1,N,3
            dis_norm = dis_norm.matmul(kornia.quaternion_to_rotation_matrix(self.model.ctl_rs[0,0]))
            dis_norm = self.model.log_ctl.exp()[0,0,:,None] * dis_norm.pow(2) # (p-v)^TS(p-v)
            self.gauss_skin = (-10 * dis_norm.sum(2)).softmax(0)[None,:,:,None]

        verts = obj_to_cam(self.pred_v, Rmat, Tmat[:,np.newaxis,:], opts.n_bones,1,skin)
        if opts.n_bones>1:
            # joints to skin
            self.joints_proj = obj_to_cam(self.model.ctl_ts[0].repeat(2*opts.batch_size,1,1), Rmat, Tmat[:,np.newaxis], opts.n_bones,1, torch.eye(opts.n_bones-1)[None,:,:,None].cuda())
            self.joints_proj = pinhole_cam(self.joints_proj, ppoint, scale)[0]
            self.bones_3d = obj_to_cam(self.model.ctl_ts[0].repeat(2*opts.batch_size,1,1), Rmat, Tmat[:,np.newaxis], opts.n_bones,1, torch.eye(opts.n_bones-1)[None,:,:,None].cuda(),tocam=False)
            self.nsphere_verts = self.sphere.vertices.shape[0] // (opts.n_bones-1)
            self.gaussian_3d = obj_to_cam(torch.Tensor(self.sphere.vertices).cuda()[None], Rmat, Tmat[:,np.newaxis], opts.n_bones, 1,
                                torch.eye(opts.n_bones-1)[None].repeat(self.nsphere_verts,1,1).permute(1,2,0).reshape(1,opts.n_bones-1,-1,1).cuda(),tocam=False)
        else:
            self.joints_proj = torch.zeros(0,3).cuda()
            self.bones_3d = None

        self.verts = obj_to_cam(self.pred_v, Rmat, Tmat[:,np.newaxis,:], opts.n_bones,1,skin,tocam=False)
        self.verts_canonical = self.verts.clone()
        self.Tmat, self.Rmat = Tmat[::opts.n_bones,:,None][0].clone(), Rmat[::opts.n_bones][0].clone()
        self.Tmat = self.Tmat/self.pred_v[0].abs().max() # to scale the reconstruction

        
        self.gaussian_3d = obj_to_cam(torch.Tensor(self.sphere.vertices).cuda()[None], Rmat, Tmat[:,np.newaxis], opts.n_bones, 1,
                                torch.eye(opts.n_bones-1)[None].repeat(self.nsphere_verts,1,1).permute(1,2,0).reshape(1,opts.n_bones-1,-1,1).cuda(),tocam=True)
        self.verts = obj_to_cam(self.pred_v, Rmat, Tmat[:,np.newaxis,:], opts.n_bones,1,skin,tocam=True)
        self.Tmat_iden = torch.zeros(3,1).cuda()
        self.Rmat_iden = torch.eye(3).cuda()

        # texture
        proj_pp =    self.ppoint.clone()  
        self.model.texture_type='vertex'
        self.tex = self.tex
        self.texture_render = render_multiplex(self.pred_v, self.faces, self.tex, Rmat, Tmat, skin, proj_pp, scale, self.renderer_softtex, 1, opts.n_bones)
        self.mask_pred = torch.zeros_like(self.texture_render[:,:1])
        

        colormap = torch.Tensor(citylabs[:skin.shape[1]]).cuda() # 5x3
        skin_color = (skin==skin.max(1)[0][:,None]).float()
        vfeat_norm = (skin_color[0] * colormap[:,None]).sum(0)
        self.skin_colors = vfeat_norm

        # render csm
        norm_meanv = self.model.mean_v[0]
        lb = norm_meanv.min()
        ub = norm_meanv.max()
        norm_meanv = (norm_meanv - lb)/(ub-lb)
        self.csm_pred = render_multiplex(self.pred_v, self.faces, norm_meanv, Rmat, Tmat, skin, proj_pp, scale, self.renderer_softtex, 1, opts.n_bones)[:,:3]
        self.csmnet_pred = (self.model.csm_pred[:1] - lb)/(ub-lb)
        self.csmnet_pred[self.masks[:1,None].repeat(1,3,1,1)<=0]=1
        self.csmnet_conf = self.model.csm_conf[:1]
        self.csmnet_conf[self.masks[:1,None]<=0]=0.
        
        Rmat_tex = Rmat.clone()
        Rmat_tex[:1] = Rmat[:1].clone().matmul(kornia.quaternion_to_rotation_matrix(torch.Tensor([[0,-0.707,0,0.707]]).cuda()))
        self.texture_vp2 = render_multiplex(self.pred_v, self.faces, norm_meanv, Rmat_tex, Tmat, skin, proj_pp, scale, self.renderer_softtex, 1, opts.n_bones)[:,:3]
        self.verts_vp2 = obj_to_cam(self.pred_v, Rmat_tex, Tmat[:,np.newaxis,:], opts.n_bones,1,skin,tocam=True)
        
        Rmat_tex[:1] = Rmat[:1].clone().matmul(kornia.quaternion_to_rotation_matrix(torch.Tensor([[-0.707,0,0,0.707]]).cuda()))
        self.texture_vp3 = render_multiplex(self.pred_v, self.faces, norm_meanv, Rmat_tex, Tmat, skin, proj_pp, scale, self.renderer_softtex, 1, opts.n_bones)[:,:3]
        self.verts_vp3 = obj_to_cam(self.pred_v, Rmat_tex, Tmat[:,np.newaxis,:], opts.n_bones,1,skin,tocam=True)

        # flow rendering
        verts_fl = obj_to_cam(self.pred_v, Rmat, Tmat[:,None],nmesh=opts.n_bones,n_hypo=1,skin=skin)
        verts_fl = torch.cat([verts_fl,torch.ones_like(verts_fl[:, :, 0:1])], dim=-1)
        verts_pos = verts_fl.clone()
        verts_fl = pinhole_cam(verts_fl, proj_pp, scale)
        # end rendering

        Rmat_tex = Rmat.clone()
        verts_tex = obj_to_cam(self.pred_v, Rmat_tex, Tmat[:,np.newaxis,:], opts.n_bones,1,skin)
        verts_tex = torch.cat([verts_tex,torch.ones_like(verts_tex[:, :, 0:1])], dim=-1)
        verts_tex = pinhole_cam(verts_tex, proj_pp, scale)
        offset = torch.Tensor( self.renderer_softtex.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
        verts_pre = verts_tex[:,:,:3]-offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]

        self.skin_vis=[]
        if self.opts.n_bones>1:
            for i in range(skin.shape[1]):
                self.skin_vis.append( self.renderer_softtex.render_mesh(sr.Mesh(verts_pre, self.faces, textures=skin[:1,i]*torch.Tensor([1,0,0]).cuda()[None,None],texture_type='vertex'))[:,:3].clone() )
            # color palette
            colormap = torch.Tensor(citylabs[:skin.shape[1]]).cuda() # 5x3
            skin_colors = (skin[0] * colormap[:,None]).sum(0)/256.
            self.part_render = self.renderer_softpart.render_mesh(sr.Mesh(verts_pre.detach(), self.faces, textures=skin_colors[None], texture_type='vertex'))[:,:3].detach()
        self.skin = skin        

    def collect_outputs(self):
        outputs = {
            'verts_canonical': self.verts_canonical.data,
            'verts': self.verts.data,
            'verts_vp2': self.verts_vp2.data,
            'verts_vp3': self.verts_vp3.data,
            'joints': self.joints_proj.data,
            'mask_pred': self.mask_pred[0].data,
            'tex': self.tex.data[0],
        }

        return outputs
