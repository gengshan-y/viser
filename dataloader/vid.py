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

import os.path as osp
import numpy as np
import random

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset

import pdb
from . import vidbase as base_data
import glob
from torch.utils.data import DataLoader
import configparser

flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_integer('n_data_workers', 1, 'Number of data loading workers')
flags.DEFINE_string('rtk_path', 'none', 'path to rtk files')
opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #
class VidDataset(base_data.BaseDataset):
    """
    Load video observations including images, flow, and silhouette.
    """

    def __init__(self, opts, filter_key=None, imglist=None, can_frame=0,
            dframe=1,init_frame=0, dataid=0, numvid=1, flip=0, rtk_path='none'):
        super(VidDataset, self).__init__(opts, filter_key=filter_key)
        
        self.flip=flip
        self.imglist = imglist
        self.can_frame = can_frame
        self.dframe = dframe
        seqname = imglist[0].split('/')[-2]
        
        if opts.sil_path=='none':
            self.masklist = [i.replace('JPEGImages', 'Annotations').replace('.jpg', '.png') for i in self.imglist]
        else:
            self.masklist = [('%s/%s/%s'%(opts.sil_path,i.split('/')[-2],i.split('/')[-1] )).replace('.jpg', '.png') for i in self.imglist]
        self.camlist =  [i.replace('JPEGImages', 'Camera').replace('.jpg', '.txt') for i in self.imglist]
      
        if dframe==1:
            self.flowfwlist = [i.replace('JPEGImages', 'FlowFW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s/flo-'%seqname) for i in self.imglist]
            self.flowbwlist = [i.replace('JPEGImages', 'FlowBW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s/flo-'%seqname) for i in self.imglist]
        else:
            self.flowfwlist = [i.replace('JPEGImages', 'FlowFW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s_%02d/flo-'%(seqname,self.dframe)) for i in self.imglist]
            self.flowbwlist = [i.replace('JPEGImages', 'FlowBW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s_%02d/flo-'%(seqname,self.dframe)) for i in self.imglist]

        if rtk_path == 'none':
            self.rtklist =[i.replace('JPEGImages', 'Cameras').replace('.jpg', '.txt') for i in self.imglist]
        elif rtk_path[:5] == 'gauss':
            self.rtklist = []
            for imgpath in self.imglist:
                rtkpath = imgpath.replace('JPEGImages', 'Cameras').replace('.jpg', '.txt') 
                rtkpath_a, rtkpath_b = rtkpath.rsplit('/',1)
                rtkpath = '%s/%s-%s'%(rtkpath_a, rtk_path, rtkpath_b)
                self.rtklist.append(rtkpath)
        else:
            self.rtklist =['%s/%s-%05d.txt'%(rtk_path, seqname, i) for i in range(len(self.imglist))]
        
        self.baselist = [i for i in range(len(self.imglist)-self.dframe)] +  [i+self.dframe for i in range(len(self.imglist)-self.dframe)]
        self.directlist = [1] * (len(self.imglist)-self.dframe) +  [0]* (len(self.imglist)-self.dframe)
        
        # to skip frames
        self.odirectlist = self.directlist.copy()
        len_list = len(self.baselist)//2
        self.baselist = self.baselist[:len_list][init_frame::self.dframe]  + self.baselist[len_list:][init_frame::self.dframe]
        self.directlist = self.directlist[:len_list][init_frame::self.dframe]  + self.directlist[len_list:][init_frame::self.dframe]
       
        self.baselist =   [self.baselist[0]]   + self.baselist   + [self.baselist[-1]]
        self.directlist = [self.directlist[0]] + self.directlist + [self.directlist[-1]]

        fac = (opts.batch_size*opts.ngpu*200)//len(self.directlist) // numvid
        if fac==0: fac=1
        self.directlist = self.directlist*fac
        self.baselist = self.baselist*fac
        # Load the annotation file.
        self.num_imgs = len(self.directlist)
        self.dataid = dataid
        self.start_idx = 0
        self.end_idx = len(self.imglist)-1
        print('%d paris of images' % self.num_imgs)


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True,capdata=None):
    num_workers = opts.n_data_workers * opts.batch_size
    num_workers = min(num_workers, 8)
    #num_workers = 0
    print('# workers: %d'%num_workers)
    print('# pairs: %d'%opts.batch_size)
   
    config = configparser.RawConfigParser()
    config.read('configs/%s.config'%opts.dataname)
    
    def get_config_info(opts, name, capdata,dataid):
        datapath =  str(config.get(name, 'datapath'))
        dframe =    [int(i) for i in config.get(name, 'dframe').split(',')]
        can_frame = int(config.get(name, 'can_frame'))
        init_frame =int(config.get(name, 'init_frame'))
        end_frame = int(config.get(name, 'end_frame'))
        imglist = sorted(glob.glob('%s/*'%datapath))
        numvid =  int(config.get('meta', 'numvid'))
        try: flip=int(config.get(name, 'flip'))
        except: flip=0

        if end_frame >0:
            imglist = imglist[:end_frame]
        print('init:%d, end:%d'%(init_frame, end_frame))
        datasets = []
        for df in dframe:
            dataset = VidDataset(opts, imglist = imglist, can_frame = can_frame,
                    dframe=df, init_frame=init_frame, dataid=dataid, 
                    numvid=numvid, flip=flip, rtk_path=opts.rtk_path) 
            datasets.append(dataset)
        return datasets

    numvid =  int(config.get('meta', 'numvid'))
    datalist = []
    for i in range(numvid):
        dataset = get_config_info(opts, 'data_%d'%i, capdata,  i)
        datalist = datalist + dataset
    data_inuse = torch.utils.data.ConcatDataset(datalist)
    def _init_fn(worker_id):
        np.random.seed(1003)
        random.seed(1003)
    sampler = torch.utils.data.distributed.DistributedSampler(
    data_inuse,
    num_replicas=opts.ngpu,
    rank=opts.local_rank,
    shuffle=True
    )
    data_inuse = DataLoader(data_inuse,
         batch_size= opts.batch_size, num_workers=num_workers, drop_last=True, worker_init_fn=_init_fn, pin_memory=True,sampler=sampler)
    return data_inuse, 0
