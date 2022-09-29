import time
import sys, os
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
sys.path.insert(0,'third_party')
from ext_utils.badja_data import BADJAData
from ext_utils.joint_catalog import SMALJointInfo
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pdb
import soft_renderer as sr
import argparse
import trimesh
from nnutils.geom_utils import obj_to_cam, pinhole_cam, orthographic_cam, render_flow_soft_3

parser = argparse.ArgumentParser(description='BADJA')
parser.add_argument('--testdir', default='',
                    help='path to test dir')
parser.add_argument('--seqname', default='camel',
                    help='sequence to test')
parser.add_argument('--type', default='mesh',
                    help='load mesh data or flow or zero')
parser.add_argument('--cam_type', default='perspective',
                    help='camera model, orthographic or perspective')
parser.add_argument('--vis', default='no',
                    help='whether to draw visualization')
parser.add_argument('--cse_mesh_name', default='smpl_27554',
                    help='whether to draw visualization')
args = parser.parse_args()

renderer_softflf = sr.SoftRenderer(image_size=256,dist_func='hard' ,aggr_func_alpha='hard',
               camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
               light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)

def preprocess_image(img,mask,imgsize):
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
    if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
        mask = cv2.resize(mask, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)[:,:,None]
    # crop box
    indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
    center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
    length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
    length = (int(1.2*length[0]), int(1.2*length[1]))

    alp = (2*length[0]/float(imgsize), 2*length[1]/float(imgsize))
    refpp = np.asarray(center)/(imgsize/2.) - 1
    return alp, refpp,center,length[0]


def draw_joints_on_image(rgb_img, joints, visibility, region_colors, marker_types,pred=None,correct=None):
    joints = joints[:, ::-1] # OpenCV works in (x, y) rather than (i, j)

    disp_img = rgb_img.copy()    
    i=0
    for joint_coord, visible, color, marker_type in zip(joints, visibility, region_colors, marker_types):
        if visible:
            joint_coord = joint_coord.astype(int)
            cv2.circle(disp_img, tuple(joint_coord),  radius=3, color=[255,0,0], thickness = 10)
            if pred is not None:
                if correct[i]:
                    color=[0,255,0]
                else:
                    color=[0,0,255]
                error = np.linalg.norm(joint_coord - pred[i,::-1],2,-1)
                cv2.circle(disp_img, tuple(joint_coord),  radius=int(error), color=color, thickness = 3)
                cv2.line(disp_img, tuple(joint_coord), tuple(pred[i,::-1]),color , thickness = 3)
        i+=1
    return disp_img

def main():
    smal_joint_info = SMALJointInfo()
    badja_data = BADJAData(args.seqname)
    data_loader = badja_data.get_loader()
    
    print(args.testdir)
    # store all the data
    all_anno = []
    all_mesh = []
    all_cam = []
    all_fr = []
    all_fl = []
    for anno in data_loader:
        all_anno.append(anno)
        rgb_img, sil_img, joints, visible, name = anno
        seqname = name.split('/')[-2]
        fr = int(name.split('/')[-1].split('.')[-2])
        all_fr.append(fr)
        print('%s/%d'%(seqname, fr))
        
        # load mesh data or flow
        if args.type=='mesh':
            if os.path.exists('%s/%s-pred%d.ply'%(args.testdir, seqname,  fr)):
                mesh = trimesh.load('%s/%s-pred%d.ply'%(args.testdir, seqname,  fr),process=False)
                cam = np.loadtxt('%s/%s-cam%d.txt'%(args.testdir,seqname, fr))
            elif os.path.exists('%s/r%s-pred%d.ply'%(args.testdir, seqname,  fr)):
                mesh = trimesh.load('%s/r%s-pred%d.ply'%(args.testdir, seqname,  fr),process=False)
                cam = np.loadtxt('%s/r%s-cam%d.txt'%(args.testdir,seqname, fr))
            else:
                print('data not found')
                exit()
            all_mesh.append(mesh)
            all_cam.append(cam)
      
    if args.type=='flow':
        import sys
        sys.path.insert(0,'data_gen')
        from models.VCNplus import VCN
        model = VCN([1, 256, 256], md=[int(4*(256/256)),4,4,4,4], fac=1)
        model = nn.DataParallel(model, device_ids=[0])
        model.cuda()
        pretrained_dict = torch.load('/data/gengshay/lasr_vcn/flow-rob-4th.pth',map_location='cpu') 
        mean_L=pretrained_dict['mean_L']
        mean_R=pretrained_dict['mean_R']
        model.load_state_dict(pretrained_dict['state_dict'],strict=False)

    if args.type=='cse':
        import sys
        sys.path.insert(0,'/data/gengshay/code/detectron2_previous/projects/DensePose') 
        from cselib import create_cse, run_cse
        if args.cse_mesh_name=='smpl_27554':
            model, embedder, mesh_vertex_embeddings = create_cse(isanimal=False)
        else:
            model, embedder, mesh_vertex_embeddings = create_cse(isanimal=True)
        import pickle
        with open('/data/gengshay/code/detectron2_previous/projects/DensePose/geodists_%s.pkl'%(args.cse_mesh_name), 'rb') as f: geodists=pickle.load(f)
        geodists = torch.Tensor(geodists).cuda()
        geodists[0,:] = np.inf
        geodists[:,0] = np.inf
        
    # store all the results
    pck_all = [] 
    for i in range(len(all_anno)):
        for j in range(len(all_anno)):
            if i!=j:
                # evaluate every two-frame
                refimg, refsil, refkp, refvis, refname = all_anno[i]
                tarimg, tarsil, tarkp, tarvis, tarname = all_anno[j]
                refseqname = refname.split('/')[-2]
                tarseqname = tarname.split('/')[-2]
                # control inter or inner; comment to compute inter
                #if refseqname!= tarseqname: continue
                print('%s vs %s'%(refname, tarname))
                
                if args.type=='mesh':
                    refmesh, tarmesh = all_mesh[i], all_mesh[j]
                    refcam, tarcam = all_cam[i], all_cam[j]
                    img_size = max(refimg.shape)
                    reffl, refpp, center, length = preprocess_image(refimg,refsil,img_size)
                    reffl2, refpp2, center2, length2 = preprocess_image(tarimg,tarsil,img_size)
                    reffl = np.stack([reffl,reffl2])
                    refpp = np.stack([refpp,refpp2])
                    renderer_softflf.rasterizer.image_size = img_size
                    # render flow between mesh 1 and 2
                    
                    refface = torch.Tensor(refmesh.faces[None]).cuda()
                    verts = torch.Tensor(np.concatenate([refmesh.vertices[None], tarmesh.vertices[None]],0)).cuda()
                    Rmat =  torch.Tensor(np.concatenate([refcam[None,:3,:3], tarcam[None,:3,:3]], 0)).cuda()
                    Tmat =  torch.Tensor(np.concatenate([refcam[None,:3,3], tarcam[None,:3,3]], 0)).cuda()
                    ppoint =  torch.Tensor(np.concatenate([refcam[None,3,2:], tarcam[None,3,2:]], 0)).cuda()
                    scale =  torch.Tensor(np.concatenate([refcam[None,3,:2], tarcam[None,3,:2]], 0)).cuda()
                    scale = scale/img_size*2
                    ppoint = ppoint/img_size * 2 -1
                    verts_fl = obj_to_cam(verts, Rmat, Tmat[:,None],nmesh=1,n_hypo=1,skin=None)
                    verts_fl = torch.cat([verts_fl,torch.ones_like(verts_fl[:, :, 0:1])], dim=-1)
                    verts_pos = verts_fl.clone()
                    if args.cam_type=='perspective':
                        verts_fl = pinhole_cam(verts_fl, ppoint, scale)
                        flow_fw, bgmask_fw, fgmask_flowf = render_flow_soft_3(renderer_softflf, verts_fl[:1], verts_fl[1:], refface)
                    elif args.cam_type=='orthographic':
                        verts_fl = orthographic_cam(verts_fl, ppoint, scale)
                        verts_fl[:,:,-2] += (-verts_fl[:,:,-2].min()+10)
                        renderer_softflf.rasterizer.far=verts_fl[:,:,2].max() + 10
                        renderer_softflf.rasterizer.near=verts_fl[:,:,2].min() - 10
                        flow_fw, bgmask_fw, fgmask_flowf = render_flow_soft_3(renderer_softflf, verts_fl[:1], verts_fl[1:], refface)
                    else:exit()
                    flow_fw[bgmask_fw]=0.
                    flow_fw[:,:,:,0] *= flow_fw.shape[2] / 2
                    flow_fw[:,:,:,1] *= flow_fw.shape[1] / 2
                    flow_fw = torch.cat([flow_fw, torch.zeros_like(flow_fw)[:,:,:,:1]],-1)[:,:refimg.shape[0],:refimg.shape[1]]
                elif args.type=='flow':
                    flow_fw = process_flow(model, refimg, tarimg, mean_L, mean_R)[0]
                    flow_fw = torch.Tensor(flow_fw[None]).cuda()
                elif args.type=='zero':
                    flow_fw = torch.zeros(refimg.shape).cuda()[None]
                elif args.type=='cse':
                    imh, imw = refimg.shape[:2]
                    csmfw1, csmbw1, image_bgr1, bbox1,bbl1 = run_cse(model, embedder, mesh_vertex_embeddings, refimg[:,:,::-1], refsil[:,:,0], mesh_name=args.cse_mesh_name)
                    csmfw2, csmbw2, image_bgr2, bbox2,bbl2 = run_cse(model, embedder, mesh_vertex_embeddings, tarimg[:,:,::-1], tarsil[:,:,0], mesh_name=args.cse_mesh_name)
                    csmfw1 = csmfw1.view(-1,1).repeat(1,112*112).view(-1,1) # dismat is f1xf2
                    csmfw2 = csmfw2.view(1,-1).repeat(112*112,1).view(-1,1)
                    idxmatch = geodists[csmfw1, csmfw2].view(112*112, 112*112).argmin(1)
                    tar_coord = torch.cat([idxmatch[:,None]%112, idxmatch[:,None]//112],-1).float()  # cx,cy
                    tar_coord[:,0]=tar_coord[:,0]*bbl2[0]/112 + bbox2[0]
                    tar_coord[:,1]=tar_coord[:,1]*bbl2[1]/112 + bbox2[1]
                    tar_coord = tar_coord.view(112, 112, 2)
                    ref_coord = torch.Tensor(np.meshgrid(range(112), range(112))).cuda().permute(1,2,0).view(-1,2)
                    ref_coord[:,0]=ref_coord[:,0]*bbl1[0]/112 + bbox1[0]
                    ref_coord[:,1]=ref_coord[:,1]*bbl1[1]/112 + bbox1[1]
                    ref_coord = ref_coord.view(112, 112, 2)
                    flow_fw = tar_coord - ref_coord
                    flow_fw = F.interpolate(flow_fw.permute(2,0,1)[None], [bbl1[1],bbl1[0]], mode='bilinear')[0].permute(1,2,0)
                    tmp = torch.zeros(refimg.shape[0], refimg.shape[1],3).cuda()
                    tmp[bbox1[1]:bbox1[3],bbox1[0]:bbox1[2],:2] = flow_fw
                    flow_fw = tmp
                    flow_fw[torch.Tensor(refsil).cuda().repeat(1,1,3)<=0] = 0
                    flow_fw = flow_fw[None]
                    
                
                refkpx = torch.Tensor(refkp.astype(float)).cuda()
                x0,y0=np.meshgrid(range(refimg.shape[1]),range(refimg.shape[0]))
                x0 = torch.Tensor(x0).cuda()
                y0 = torch.Tensor(y0).cuda()
                idx = ( (flow_fw[:,:,:,:2].norm(2,-1)<1e-6).float().view(1,-1)*1e6+ (torch.pow(refkpx[:,0:1]-y0.view(1,-1),2) + torch.pow(refkpx[:,1:2]-x0.view(1,-1),2))/1000 ).argmin(-1)
                samp_flow = flow_fw.view(-1,3)[idx][:,:2]
                tarkp_pred = refkpx.clone()
                tarkp_pred[:,0] = tarkp_pred[:,0] +(samp_flow[:,1])
                tarkp_pred[:,1] = tarkp_pred[:,1] +(samp_flow[:,0])
                tarkp_pred = np.asarray(tarkp_pred.cpu())

                diff = np.linalg.norm(tarkp_pred - tarkp, 2,-1)
                sqarea = np.sqrt((refsil[:,:,0]>0).sum())
                correct = diff < sqarea * 0.2
                correct = correct[np.logical_and(tarvis, refvis)]
                if args.vis=='yes':
                    rgb_vis = draw_joints_on_image(refimg, refkp, refvis, smal_joint_info.joint_colors, smal_joint_info.annotated_markers)
                    tarimg = draw_joints_on_image(tarimg, tarkp, refvis, smal_joint_info.joint_colors, smal_joint_info.annotated_markers, pred=tarkp_pred,correct=diff < sqarea * 0.2)
                    cv2.imwrite('%s/%s-%05d-%s-%05d-ref.png'%(args.testdir,refseqname, all_fr[i],tarseqname,all_fr[j]),rgb_vis[:,:,::-1]) 
                    cv2.imwrite('%s/%s-%05d-%s-%05d-tar.png'%(args.testdir,    refseqname, all_fr[i],tarseqname,all_fr[j]),tarimg[:,:,::-1])
                    write_pfm( '%s/%s-%05d-%s-%05d.pfm'%(args.testdir, refseqname, all_fr[i], tarseqname, all_fr[j]),np.asarray(flow_fw[0].detach().cpu()))
                    

                pck_all.append(correct)
    print('PCK %.02f'%(100*np.concatenate(pck_all).astype(float).mean()))

if __name__ == '__main__':
    main()
