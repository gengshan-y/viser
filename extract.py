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

from absl import flags, app
import sys
sys.path.insert(0,'third_party')

import numpy as np
import skimage.io as io
import torch
import os
import glob
import pdb
import cv2
import matplotlib.pyplot as plt
import soft_renderer as sr
import trimesh

from nnutils.train_utils import LASRTrainer
from nnutils import predictor as pred_util
from nnutils.geom_utils import label_colormap
from ext_utils import fusion
from ext_utils.flowlib import point_vec
from ext_utils import image as img_util
import configparser
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

opts = flags.FLAGS

citylabs = label_colormap()

def preprocess_pair(img_path, imgn_path, img_size, dframe=1):
    img, alp, img_vis, mask,flow, center, length,pps = preprocess_image(img_path, dframe=dframe)
    imgn, alpn, img_visn, maskn,flown, centern, lengthn,ppsn = preprocess_image(imgn_path)
    #_, _, img_vis, _,_, _, _ = preprocess_image(img_path, equal=True)

    A = np.eye(3)
    B = np.asarray([[alp[0],0,(center[0]-length[0])],
                        [0,alp[1],(center[1]-length[1])],
                        [0,0,1]]).T
    Ap = np.eye(3)
    Bp = np.asarray([[alpn[0],0,(centern[0]-lengthn[0])],
                        [0,alpn[1],(centern[1]-lengthn[1])],
                        [0,0,1]]).T

    x0,y0  =np.meshgrid(range(img_size),range(img_size))
    hp0 = np.stack([x0,y0,np.ones_like(x0)],-1)
    hp0 = np.dot(hp0,A).dot(B)                   # image coord
    hp1c = np.concatenate([flow[:,:,:2] + hp0[:,:,:2], np.ones_like(hp0[:,:,:1])],-1) # image coord
    hp1c = hp1c.dot(np.linalg.inv(Ap.dot(Bp)))   # screen coord
    flow[:,:,:2] = hp1c[:,:,:2] - np.stack([x0,y0],-1)
    
    # draw optical flow
    flow[~mask.astype(bool)] = 0
    img_vis = np.transpose(img_vis[::-1],[1,2,0])
    img_vis = point_vec(img_vis*255, flow)
    img_vis = img_vis/255
    img_vis = np.transpose(img_vis,[2,0,1])
    return img, alp, img_vis, mask, imgn, alpn, img_visn, maskn, pps, ppsn

def preprocess_image(img_path, img_size=256, equal=False, dframe=1):
    img = cv2.imread(img_path)[:,:,::-1] / 255.

    #img[:,:,0]=img[:,:,0].mean()
    #img[:,:,1]=img[:,:,1].mean()
    #img[:,:,2]=img[:,:,2].mean()
    #img = cv2.blur(img, (9,9))
    #img = cv2.imread(img_path) / 255.

    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

    mask = cv2.imread(img_path.replace('JPEGImages', 'Annotations').replace('.jpg','.png'),0)
    mask = mask/np.sort(np.unique(mask))[1]
    mask[mask==255] = 0
    if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
        mask = cv2.resize(mask, img.shape[:2][::-1])
    mask = np.expand_dims(mask, 2)

    # flow    
    seqname = img_path.split('/')[-2]
    try:
        if dframe==1:
            suffix=''
        else:
            suffix='_%02d'%dframe
        flow = readPFM(img_path.replace('JPEGImages', 'FlowFW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s%s/flo-'%(seqname,suffix)))[0]
    except: flow = np.zeros((img_size, img_size, 3))

    color = img[mask[:,:,0].astype(bool)].mean(0)
    #img =   img*(mask>0).astype(float) + np.random.rand(mask.shape[0],mask.shape[1],1)*(1-(mask>0).astype(float))
    img =   img*(mask>0).astype(float) + (1-color )[None,None,:]*(1-(mask>0).astype(float))
    img_vis =   img*(mask>0).astype(float) + (1-(mask>0).astype(float))

    # crop box
    indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
    center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
    length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)

    if equal:
        maxlength = int(1.2*max(length))
        length = (maxlength,maxlength)
    else:
        length = (int(1.2*length[0]), int(1.2*length[1]))

    maxw=img_size;maxh=img_size
    orisize = (2*length[0], 2*length[1])
    alp =  [orisize[0]/maxw  ,orisize[1]/maxw]
    x0,y0  =np.meshgrid(range(maxw),range(maxh))
    # geometric augmentation for img, mask, flow, occ
    A = np.eye(3)
    B = np.asarray([[alp[0],0,(center[0]-length[0])],
                    [0,alp[1],(center[1]-length[1])],
                    [0,0,1]]).T

    hp0 = np.stack([x0,y0,np.ones_like(x0)],-1)  # screen coord
    hp0 = np.dot(hp0,A).dot(B)                   # image coord
    x0 = hp0[:,:,0].astype(np.float32)
    y0 = hp0[:,:,1].astype(np.float32)

    img = cv2.remap(img,x0,y0,interpolation=cv2.INTER_LINEAR,borderValue=(1-color))
    img_vis = cv2.remap(img_vis,x0,y0,interpolation=cv2.INTER_LINEAR,borderValue=img_vis[0,0])
    flow = cv2.remap(flow,x0,y0,interpolation=cv2.INTER_LINEAR,borderValue=0)
    mask = cv2.remap(mask.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))
    img_vis = np.transpose(img_vis, (2, 0, 1))

    pps = np.asarray([float( center[0] - length[0] ), float( center[1] - length[1]  )])

    return img, alp, img_vis, mask, flow, center, length, pps


def visualize(img, outputs, predictor,ipath,saveobj=False,frameid=None):
    faces = np.asarray(predictor.faces.cpu()[0])
    vert_vp1 = np.asarray(outputs['verts'][0]    .cpu()) 
    vert_vp2 = np.asarray(outputs['verts_vp2'][0].cpu())
    vert_vp3 = np.asarray(outputs['verts_vp3'][0].cpu())

    if frameid is None:
        frameid=int(ipath.split('/')[-1].split('.')[0])
    if saveobj or predictor.opts.n_bones>1:
        seqname = predictor.opts.dataname
        save_dir = predictor.opts.checkpoint_dir
        vp1_mesh = trimesh.Trimesh(vert_vp1, faces, process=False,vertex_colors=255*outputs['tex'].cpu())
        vp2_mesh = trimesh.Trimesh(vert_vp2, faces, process=False,vertex_colors=255*outputs['tex'].cpu())
        vp3_mesh = trimesh.Trimesh(vert_vp3, faces, process=False,vertex_colors=255*outputs['tex'].cpu())

        vp1_mesh.export('%s/%s-vp1pred%d.obj'%(save_dir, seqname,frameid))
        vp2_mesh.export('%s/%s-vp2pred%d.obj'%(save_dir, seqname,frameid))
        vp3_mesh.export('%s/%s-vp3pred%d.obj'%(save_dir, seqname,frameid))
        trimesh.Trimesh(np.asarray(outputs['verts_canonical'][0].cpu()), 
            np.asarray(predictor.faces.cpu()[0]), process=False).\
        export('%s/%s-mesh-%05d.obj'%(save_dir, seqname, frameid))
        if predictor.bones_3d is not None:
            colormap = torch.Tensor(citylabs[:predictor.bones_3d.shape[1]]).cuda() # 5x3
            skin_colors = predictor.skin_colors
            bone_colors = colormap[None].repeat(predictor.nsphere_verts,1,1).permute(1,0,2).reshape(-1,3)
            bone_mesh = trimesh.Trimesh( np.asarray(predictor.gaussian_3d[0].cpu()),
                                         predictor.sphere.faces,
                                         process=False,
                                        vertex_colors=np.asarray(bone_colors.cpu())) 
            bone_mesh.export('%s/%s-gauss%d.obj'%(save_dir, seqname, frameid))
                       
            skin_mesh = trimesh.Trimesh(vert_vp1, faces, process=False,vertex_colors=skin_colors.cpu())
            skin_mesh.export('%s/%s-skinpred%d.obj'%(save_dir, seqname,  frameid))
            
        # camera
        K = np.asarray(torch.cat([predictor.model.uncrop_scale[0,0,:], predictor.model.uncrop_pp],-1).view(-1,4).cpu())
        RT_iden = np.asarray(torch.cat([predictor.Rmat_iden, predictor.Tmat_iden],-1).cpu())
        RTK_iden = np.concatenate([RT_iden,K],0)
        RT = np.asarray(torch.cat([predictor.Rmat.T, predictor.Tmat],-1).cpu())
        RTK = np.concatenate([RT,K],0)
        np.savetxt('%s/%s-cam%d.txt'%(save_dir,    seqname,frameid),RTK_iden)
        np.savetxt('%s/%s-cam-%05d.txt'%(save_dir, seqname,frameid),RTK)

    mask_pred = np.asarray(predictor.mask_pred[0][0].detach().cpu())*255 
    csmnet_pred = np.asarray(predictor.csmnet_pred.data[0].permute(1,2,0).cpu())
    csmnet_conf = np.asarray(predictor.csmnet_conf.data[0,0][...,None].repeat(1,1,3).cpu())
    vp1 = np.asarray(predictor.texture_render.data[0].permute(1,2,0).cpu())
    vp2 = np.asarray(predictor.texture_vp2.data[0].permute(1,2,0).cpu())
    vp3 = np.asarray(predictor.texture_vp3.data[0].permute(1,2,0).cpu())
    csm_pred = np.asarray(predictor.csm_pred.data[0].permute(1,2,0).cpu())

    # resize
    vp1 = cv2.resize(vp1, (img.shape[2], img.shape[1]))
    vp2 = cv2.resize(vp2, (img.shape[2], img.shape[1]))
    vp3 = cv2.resize(vp3, (img.shape[2], img.shape[1]))
    mask_pred = cv2.resize(mask_pred, (img.shape[2], img.shape[1]))

    img = np.transpose(img, (1, 2, 0))

    redImg = np.zeros(img.shape, np.uint8)
    redImg[:,:] = (0, 0, 255)
    redMask = (redImg * mask_pred[:,:,np.newaxis]/255).astype(np.uint8)
    redMask = cv2.addWeighted(redMask, 0.5, (255*img).astype(np.uint8), 1, 0, (255*img).astype(np.uint8))

    plt.ioff()
    plt.figure(figsize=(21,3))
    plt.clf()
    plt.subplot(171)
    plt.imshow(redMask)
    plt.gca().set_title('input [frame %d]'%frameid)
    plt.axis('off')
    
    plt.subplot(172)
    plt.imshow(csmnet_pred)
    plt.gca().set_title('CSM prediction')
    plt.axis('off')
    
    plt.subplot(173)
    plt.imshow(csmnet_conf)
    plt.gca().set_title('CSM confidence')
    plt.axis('off')

    plt.subplot(174)
    plt.imshow(vp1)
    plt.gca().set_title('rendered')
    plt.axis('off')

    plt.subplot(175)
    plt.imshow(csm_pred)
    plt.gca().set_title('skin weights')
    plt.axis('off')

    plt.subplot(176)
    plt.imshow(vp2)
    plt.gca().set_title('right view')
    plt.axis('off')
    plt.subplot(177)
    plt.imshow(vp3)
    plt.gca().set_title('top view')
    plt.axis('off')

    plt.gca().set_facecolor('white')
    plt.draw()
    plt.savefig('%s/render-%s.png'%(save_dir, ipath.split('/')[-1].split('.')[0]), dpi=200)
    plt.close()
    
    plt.figure(figsize=(16,16))
    plt.clf()
    for i in range(len(predictor.skin_vis)):
        plt.subplot(6,7,i+1)
        skinvis = np.asarray(predictor.skin_vis[i][0].permute(1,2,0).cpu())
        plt.imshow(skinvis)
        plt.axis('off')
    plt.draw()
    plt.savefig('%s/renderskin-%s.png'%(save_dir, ipath.split('/')[-1].split('.')[0]))
    plt.close()

    # visualize skin
        

def main(_):
    config = configparser.RawConfigParser()
    config.read('configs/%s.config'%opts.dataname)
    datapath = str(config.get('data', 'datapath'))
    canonical_frame = int(config.get('data', 'can_frame'))
    dframe = int(config.get('data', 'dframe'))
    init_frame = int(config.get('data', 'init_frame'))
    end_frame = int(config.get('data', 'end_frame'))
   
    predictor = pred_util.MeshPredictor(opts)
    pathlist = sorted(glob.glob('%s/*'%datapath))
    for i,ipath in enumerate(pathlist):
        if (i%dframe!=init_frame%dframe) \
                or (i<init_frame) \
                or (end_frame>=0 and i >= end_frame):continue
        if i+dframe<len(pathlist):
            img, alp, imgb, mask, imgn,alpn,imgbn,maskn,pps,ppsn = preprocess_pair(\
                                pathlist[i], pathlist[i+dframe], \
                                img_size=opts.img_size, dframe=dframe)
        else:
            img,alp,imgb,mask,_,_,_,pps = preprocess_image(\
                                ipath, img_size=opts.img_size)

        batch = {'img': torch.Tensor(np.expand_dims(img, 0)),
                'imgn': torch.Tensor(np.expand_dims(imgn, 0)),
                'mask': torch.Tensor(np.expand_dims(mask, 0)),
                'pps': torch.Tensor(np.expand_dims(pps, 0)),
                'ppsn': torch.Tensor(np.expand_dims(ppsn, 0))}

        print('frame-id:%d'%i)
        with torch.no_grad():
            outputs = predictor.predict(batch,alp,frameid=i)
        visualize(imgb, outputs, predictor,ipath,saveobj=True)

if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
