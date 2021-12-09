from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import pdb
import kornia
import sys
import sys
sys.path.insert(0,'third_party')
from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse, get_joint_embedding_function, get_time_embedding_function,
      load_blender_data, load_llff_data, meshgrid_xy, models,
      run_network, run_one_iter_of_nerf) 
from nerf.nerf_helpers import positional_encoding

import cv2; import trimesh
from matplotlib import cm
viridis = cm.get_cmap('jet', 12)

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None,dilation=1,with_bn=True):
        super(residualBlock, self).__init__()
        if dilation > 1:
            padding = dilation
        else:
            padding = 1

        if with_bn:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1)
        else:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, padding, dilation=dilation,with_bn=False)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, with_bn=False)
        self.downsample = downsample
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True))


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()
        bias = not with_bn

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()
        bias = not with_bn
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.1, inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, with_bn=True, levels=4):
        super(pyramidPooling, self).__init__()
        self.levels = levels

        self.paths = []
        for i in range(levels):
            self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, with_bn=with_bn))
        self.path_module_list = nn.ModuleList(self.paths)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        for pool_size in np.linspace(1,min(h,w)//2,self.levels,dtype=int):
            k_sizes.append((int(h/pool_size), int(w/pool_size)))
            strides.append((int(h/pool_size), int(w/pool_size)))
        k_sizes = k_sizes[::-1]
        strides = strides[::-1]

        pp_sum = x

        for i, module in enumerate(self.path_module_list):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear')
            pp_sum = pp_sum + 1./self.levels*out
        pp_sum = self.relu(pp_sum/2.)

        return pp_sum

class pspnet(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """
    def __init__(self, is_proj=True,groups=1, out_channel=64):
        super(pspnet, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16,
                                                 padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,64,1,stride=2)
        self.res_block5 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block6 = self._make_layer(residualBlock,128,1,stride=2)
        self.res_block7 = self._make_layer(residualBlock,128,1,stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)

        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                                 padding=1, stride=1)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                 padding=1, stride=1)
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                 padding=1, stride=1))
        self.iconv2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
                                                 padding=1, stride=1)

        if self.is_proj:
            self.proj6 = nn.Conv2d(128,128,1, padding=0,stride=1)
            self.proj5 = nn.Conv2d(128,128,1, padding=0,stride=1)
            self.proj4 = nn.Conv2d(128,128,1, padding=0,stride=1)
            self.proj3 = nn.Conv2d(64, 64 ,1, padding=0,stride=1)
            self.proj2 = nn.Conv2d(64, out_channel ,1, padding=0,stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
       

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        conv6x = F.upsample(conv6, [conv5.size()[2],conv5.size()[3]],mode='bilinear')
        concat5 = torch.cat((conv5,self.upconv6[1](conv6x)),dim=1)
        conv5 = self.iconv5(concat5) 

        conv5x = F.upsample(conv5, [conv4.size()[2],conv4.size()[3]],mode='bilinear')
        concat4 = torch.cat((conv4,self.upconv5[1](conv5x)),dim=1)
        conv4 = self.iconv4(concat4) 

        conv4x = F.upsample(conv4, [rconv3.size()[2],rconv3.size()[3]],mode='bilinear')
        concat3 = torch.cat((rconv3,self.upconv4[1](conv4x)),dim=1)
        conv3 = self.iconv3(concat3) 

        conv3x = F.upsample(conv3, [pool1.size()[2],pool1.size()[3]],mode='bilinear')
        concat2 = torch.cat((pool1,self.upconv3[1](conv3x)),dim=1)
        conv2 = self.iconv2(concat2) 

        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
            proj2 = self.proj2(conv2)
            return proj6,proj5,proj4,proj3,proj2
        else:
            return conv6, conv5, conv4, conv3, conv2


class SurfaceMatchNet(nn.Module):
    def __init__(self):
        super(SurfaceMatchNet, self).__init__()
        self.nfeat=16

        self.featnet = pspnet(out_channel=self.nfeat)
        self.featnerf = getattr(models, 'CondNeRFModel')(
                    num_encoding_fn_xyz=10,
                    num_encoding_fn_dir=4,
                    include_input_xyz=False,
                    include_input_dir=False,
                    use_viewdirs=False,
                    out_channel=self.nfeat,
                    codesize=0)
        self.encode_position_fn = get_embedding_function(
            num_encoding_functions=10,
            include_input=False,
            log_sampling=True,
        )

        self.tau = torch.Tensor([0]) # temperature scaling
        self.tau = nn.Parameter(self.tau)
        
        self.tau_back = torch.Tensor([0]) # temperature scaling
        self.tau_back = nn.Parameter(self.tau_back)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = m.in_features
                m.weight.data.normal_(0.0,1/np.sqrt(y))
                m.bias.data.fill_(0)

    def forward(self, img, masks, verts, faces, transform, detach=False):
        # softmax(<CNN(I), MLP(V)>)@V # 64x64x1000x64 -> (640x640x200)
        import pytorch3d
        bs,_,h,w = img.shape
        rep = img.shape[0]//masks.shape[0]
        if self.training:
            npts = 200
            #npts = 1000
            pts, _= pytorch3d.ops.sample_points_from_meshes(pytorch3d.structures.meshes.Meshes(verts=verts, faces=faces), npts ,return_normals=True)
        else:
            pts = verts
            npts = verts.shape[1]

        feat = self.featnet(img)[-1]  # b,c,h,w

        # output feature is always not detached
        out_feat = feat.clone()
        if detach: feat=feat.detach()

        hf,wf = feat.shape[2:]
        feat_nerf = run_network(
            self.featnerf,
            pts.detach(),
            None,
            131072,
            self.encode_position_fn,
            None,
            code=None,
        )[:,:,:-1] # b,n,c

        # corr
        feat_n = F.normalize(feat,2,1)
        feat_nerf_n = F.normalize(feat_nerf, 2,-1)
        pointcorr = feat_n[:,None] * feat_nerf_n[:,:,:,None,None] # b,n,c,h,w
        pointcorr = (pointcorr.sum(2)).permute(0,2,3,1) # bnhw->bhwn range -1 to 1
        pointcorr = 10*self.tau.exp()*pointcorr.clone()
        pred = pointcorr.clone()

        ## dist
        #pointcorr = (feat)[:,None] - (20*feat_nerf)[:,:,:,None,None] # b,n,c,h,w
        #pointcorr = -(pointcorr.pow(2).sum(2)).permute(0,2,3,1) # bnhw->bhwn range -1 to 1
        #pointcorr = self.tau.exp()*pointcorr
        #pred = pointcorr.clone()

        #topk, indices = pred.topk(5, -1, largest=True)
        #res = torch.zeros_like(pred).fill_(-np.inf)
        #res = res.scatter(-1, indices, topk)
        #pred = res

        pred = torch.softmax(pred,-1)
        pred = pred.view(bs,-1,npts,1) * pts.detach()[:,None]
        pred = pred.sum(-2).view(bs, hf,wf,3).permute(0,3,1,2)

        # 3d-2d mapping
        #ipred = pointcorr_dot.view(bs,-1,npts)
        ipred = pointcorr.clone().view(bs,-1,npts) * self.tau_back.exp() 
        #topk, indices = ipred.topk(1, 1, largest=True)
        #res = torch.zeros_like(ipred).fill_(-np.inf)
        #res = res.scatter(1, indices, topk)
        #ipred = res
        
        ipred = ipred.softmax(1)
        meshgrid = torch.Tensor(np.meshgrid(range(hf), range(wf))).cuda().view(2,-1) # 2hw
        meshgrid = meshgrid/(hf/2)-1
        ipred = ipred[:,None] * meshgrid[None,:,:,None]
        ipred = ipred.sum(2) # bs, 2, n

        
        ### ipred vis
        #pdb.set_trace()
        #import cv2
        #mask = torch.zeros(hf,wf)      
        #for i in range(npts):
        #    mask[(ipred[0,1,i]*32+32).int(), (ipred[0,0,i]*32+32).int()] = 1
        #cv2.imwrite('tmp/0.png', np.asarray(mask.cpu())*200)


        with torch.no_grad():
            # bs, n, 3, h, w
            dis3d = (pred[:,None] - pts[:,:,:,None,None]).norm(2,2)
            dis3d = dis3d.argmin(1).view(bs,-1) # hw
            fberr = torch.zeros(bs,1,hf,wf).cuda()
            for i in range(bs):
                ipred_i = ipred[i].permute(1,0)[dis3d[i]] # n, 2
                fberr[i,0] = (meshgrid.T - ipred_i).norm(2,-1).view(hf,wf)
            conf = (-5*fberr).exp() 

        # uncertainty
        #conf = torch.softmax(pointcorr,-1)
        #max_entropy = np.log(npts)
        #conf = -(conf * torch.log(conf+ 1e-9)).sum(-1)[:,None] # entropy
        #conf = 1-conf/max_entropy
        #conf = (0.01*pointcorr).max(-1)[0][:,None].sigmoid() # max logits
        conf = F.interpolate(conf, (h,w), mode='bilinear').detach()
        conf[conf<0.5] = 0 # ~10px

        ## 2d-3d mapping vis
        #pdb.set_trace()
        #xcoord = 32
        #ycoord = 32
        #colors = pointcorr[0,ycoord,xcoord]
        #colors = colors-colors.mean()
        #colors = colors / colors.std()
        #colors = (colors-colors.min())/(colors.max()-colors.min())
        #colors = viridis(colors.cpu())[:,:3]
        #trimesh.Trimesh(verts.cpu()[0],faces.cpu()[0],vertex_colors=colors, process=False).export('/data/gengshay/0.ply')
        #featvis = F.normalize(feat.std(1),2,0)[0]
        #featvis[ycoord,xcoord] = 0
        #cv2.imwrite('/data/gengshay/0.png', np.asarray(featvis.cpu())*255)

        ## 3d-2d mapping vis
        #pdb.set_trace()
        #for i in range(npts):
        #    colors = pointcorr[0,:,:,i].view(-1)
        #    colors = colors-colors.mean()
        #    colors = colors / colors.std()
        #    colors = (colors-colors.min())/(colors.max()-colors.min())
        #    colors = viridis(colors.cpu())[:,:3].reshape((hf,wf,3))
        #    colors = cv2.resize(colors,(256,256))
        #    cv2.imwrite('/data/gengshay/%03d.png'%i, colors[:,:,::-1]*255)

        pred = F.interpolate(pred, (h,w), mode='bilinear')
        #TODO better consistency term
        #conf[:] = 0.1
        return pred, ipred, pts, out_feat, conf
