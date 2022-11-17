#    Copyright 2022 Institute of Medical Informatics, University of Luebeck, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


import numpy as np

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)



class UnrollMax(nn.Module):
    def __init__(self,):
        super(UnrollMax, self).__init__()
        
    def forward(self, centre, neighbour, inrel):
        concat = torch.zeros_like(centre)
        #print('new')
        for i in range(6):#
            #concat += checkpoint(self.inrel,centre+torch.roll(neighbour,int((i%2-.5)*2),dims=int(i//2)+2))/6
            concat = torch.maximum(concat,checkpoint(inrel,centre+torch.roll(neighbour,int((i%2-.5)*2),dims=int(i//2)+2)))
        return concat
        
        
        
class XEdgeConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(XEdgeConv3d, self).__init__(in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.mid_channels = max(8,(in_channels+out_channels)//2)
        self.conv1 = nn.Conv3d(in_channels, self.mid_channels,1,padding=(0,0,0))
        self.conv2 = nn.Conv3d(in_channels, self.mid_channels,1,padding=(0,0,0))
        self.conv3 = nn.Conv3d(self.mid_channels, out_channels,1,padding=(0,0,0))
        self.inrel = nn.Sequential(nn.InstanceNorm3d(self.mid_channels),nn.ReLU())
        self.C = out_channels
        self.max = UnrollMax()
       
    def forward(self, x):
        C = self.C
        mid = self.mid_channels
        B,C_in,H,W,D = x.shape
        if(x.shape[1]==1):
            x.requires_grad = True
        
        centre = self.conv2(x)
        neighbour = self.conv1(x)
        #concat = torch.zeros_like(centre)
        #for i in range(6):#
        #    #concat += checkpoint(self.inrel,centre+torch.roll(neighbour,int((i%2-.5)*2),dims=int(i//2)+2))/6
        #    concat = torch.maximum(concat,checkpoint(self.inrel,centre+torch.roll(neighbour,int((i%2-.5)*2),dims=int(i//2)+2)))
        
        concat = checkpoint(self.max,centre,neighbour,self.inrel)
        output = self.conv3(concat)#/6
        if(self.stride[0]==2):
            output = F.avg_pool3d(output,kernel_size=(2,2,2),stride=(2,2,2))
        return output
    
class nnUNetTrainerV2_XEdgeConvMax(nnUNetTrainerV2):
    def initialize_network(self):
        self.base_num_features = 16  # otherwise we run out of VRAM
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        #self.net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        #self.net_pool_per_axis = [[4, 4, 4]]
        self.net_num_pool_op_kernel_sizes = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]])
        self.net_pool_per_axis = np.array([4, 4, 4])
        
        #self.pool_op_kernel_sizes = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        
     #   base_num_features = 16  # otherwise we run out of VRAM

    #net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

        count = 0; count2 = 0
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Conv3d):
                before = get_layer(self.network, name)
                if(before.kernel_size[0] == 3):
                    after = XEdgeConv3d(before.in_channels,before.out_channels,before.kernel_size,stride=before.stride)
                    set_layer(self.network, name, after); count += 1
                    

            if isinstance(module, nn.ConvTranspose3d):
                before = get_layer(self.network, name)
                after = nn.Sequential(nn.Conv3d(before.in_channels,before.out_channels,1,bias=False),nn.Upsample(scale_factor=before.kernel_size,mode='trilinear',align_corners=False))
                set_layer(self.network, name, after); count2 += 1
        print(count,'# Conv3d > XEdgeConv','and',count2,'#ConvTransp')
                    
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
