# MIT License
# 
# Copyright (c) 2018 akanazawa
# Copyright (c) 2021 Google LLC
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')

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
import kornia
import configparser
import soft_renderer as sr
from nnutils.geom_utils import pinhole_cam, obj_to_cam, render_flow_soft_3
from nnutils.geom_utils import label_colormap
citylabs = label_colormap()

class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100):
        super(MeshNet, self).__init__()

    def forward(self, batch_input):
        pass
    
    def symmetrize(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        if self.symmetric:
            if V.dim() == 2:
                # No batch
                V_left = self.flip.cuda() * V[-self.num_sym:]
                verts=torch.cat([V, V_left], 0)
                verts[:self.num_indept,self.opts.symidx]=0
                return verts 
            else:
                pdb.set_trace()
                # With batch
                V_left = self.flip * V[:, -self.num_sym:]
                verts = torch.cat([V, V_left], 1)
                verts[:,:self.num_indept, 0] = 0
                return verts
        else:
            return V
    
    def symmetrize_color(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        # No batch
        if self.symmetric:
            V_left = V[-self.num_sym:]
            verts=torch.cat([V, V_left], 0)
        else: verts = V
        return verts 

    def symmetrize_color_faces(self, tex_pred):
        if self.symmetric:
            tex_left = tex_pred[-self.num_sym_faces:]
            tex = torch.cat([tex_pred, tex_left], 0)
        else: tex = tex_pred
        return tex
    
    def get_mean_shape(self,local_batch_size):
        mean_v = torch.cat([self.symmetrize(i)[None] for i in self.mean_v],0)
        faces = self.faces
        if self.texture_type=='surface':
            tex = torch.cat([self.symmetrize_color_faces(i) for i in self.tex],0)
        else:
            tex = torch.cat([self.symmetrize_color(i)[None] for i in self.tex],0)
        
        faces = faces.repeat(2*local_batch_size,1,1)
        mean_v = mean_v[None].repeat(2*local_batch_size,1,1,1).view(2*local_batch_size*mean_v.shape[0],-1,3)
        if self.texture_type=='surface':
            tex = tex[np.newaxis].repeat(2*local_batch_size,1,1,1).sigmoid()
        else:
            tex = tex[np.newaxis].repeat(2*local_batch_size,1,1,1).sigmoid().view(2*local_batch_size*tex.shape[0],-1,3)
        return mean_v, tex, faces
