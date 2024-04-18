#    Copyright 2022 Institute of Medical Informatics, University of Luebeck, Germany

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

import functools

import torch
from torch.utils.checkpoint import checkpoint

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer



def set_module(module, keychain, replacee):
    """Replaces any module inside a pytorch module for a given keychain with "replacee".
       Use module.named_modules() to retrieve valid keychains for layers.
       e.g.
       first_keychain = list(module.keys())[0]
       new_first_replacee = torch.torch.nn.Conv1d(1,2,3)
       set_module(first_keychain, torch.torch.nn.Conv1d(1,2,3))
    """
    get_fn = lambda self, key: self[int(key)] if isinstance(self, torch.torch.nn.Sequential) \
        else getattr(self, key)

    key_list = keychain.split('.')
    root = functools.reduce(get_fn, key_list[:-1], module)
    leaf_id = key_list[-1]

    if isinstance(root, torch.torch.nn.Sequential) and leaf_id.isnumeric():
        root[int(leaf_id)] = replacee
    else:
        setattr(root, leaf_id, replacee)



class UnrollMax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, centre, neighbour, norm_relu):
        concat = torch.zeros_like(centre)
        for i in range(6):
            nb_sum = centre + torch.roll(neighbour, int((i%2-.5)*2), dims=int(i//2)+2)
            if not nb_sum.requires_grad:
                nb_sum = norm_relu(nb_sum)
            else:
                nb_sum = checkpoint(norm_relu, nb_sum)

            concat = torch.maximum(concat, nb_sum)
        return concat



class XEdgeConv3d(torch.nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mid_channels = max(8, (self.in_channels+self.out_channels)//2)

        self.conv1 = torch.nn.Conv3d(self.in_channels, self.mid_channels, 1, padding=(0,0,0))
        self.conv2 = torch.nn.Conv3d(self.in_channels, self.mid_channels, 1, padding=(0,0,0))
        self.conv3 = torch.nn.Conv3d(self.mid_channels, self.out_channels, 1, padding=(0,0,0))
        self.norm_relu = torch.nn.Sequential(torch.nn.InstanceNorm3d(self.mid_channels), torch.nn.ReLU())
        self.unroll_max = UnrollMax()

    def forward(self, x):
        neighbour = self.conv1(x)
        centre = self.conv2(x)

        if not x.requires_grad:
            concat = self.unroll_max(centre, neighbour, self.norm_relu)
        else:
            concat = checkpoint(self.unroll_max, centre, neighbour, self.norm_relu)
        output = self.conv3(concat)

        if(self.stride[0]==2):
            output = torch.nn.functional.avg_pool3d(output, kernel_size=(2,2,2), stride=(2,2,2))
        return output



class nnUNetTrainer_XEdgeConv(nnUNetTrainer):

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:

        network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        assert dim == 3, "This trainer only works with 3D networks."

        for name, module in network.named_modules(remove_duplicate=False):
            if isinstance(module, torch.nn.Conv3d):
                if(module.kernel_size[0] == 3):
                    xedgeconv = XEdgeConv3d(
                        module.in_channels,
                        module.out_channels,
                        module.kernel_size,
                        stride=module.stride
                    )
                    set_module(network, name, xedgeconv)

            if isinstance(module, torch.nn.ConvTranspose3d):
                upscale_conv = torch.nn.Sequential(
                    torch.nn.Conv3d(
                        module.in_channels,
                        module.out_channels,
                        1,
                        bias=False
                    ),
                    torch.nn.Upsample(
                        scale_factor=module.kernel_size,
                        mode='trilinear',
                        align_corners=False
                    )
                )
                set_module(network, name, upscale_conv)

        return network