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
       new_first_replacee = torch.nn.Conv1d(1,2,3)
       set_module(first_keychain, torch.nn.Conv1d(1,2,3))
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



class SymPermuteConv3d(torch.nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight = torch.nn.Conv3d(self.in_channels, self.out_channels, (4,1,1), groups=self.groups).weight
        self.weight_idxs = torch.tensor([
            [[3,0,3],[0,1,0],[3,0,3]],
            [[0,1,0],[1,2,1],[0,1,0]],
            [[3,0,3],[0,1,0],[3,0,3]]
        ])

    def forward(self, x):
        weight = self.weight[:,:,self.weight_idxs].squeeze(6).squeeze(5)
        x = torch.nn.functional.conv3d(x, weight, bias=self.bias,
                                       padding=self.padding, dilation=self.dilation,
                                       groups=self.groups,stride=self.stride
        )

        return x



class nnUNetTrainer_SymPermuteConv(nnUNetTrainer):

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
                    symconv = SymPermuteConv3d(
                        module.in_channels,
                        module.out_channels,
                        module.kernel_size,
                        stride=module.stride
                    )
                    set_module(network, name, symconv)

            # if isinstance(module, torch.nn.ConvTranspose3d):
            #     upscale_conv = torch.nn.Sequential(
            #         torch.nn.Conv3d(
            #             module.in_channels,
            #             module.out_channels,
            #             1,
            #             bias=False
            #         ),
            #         torch.nn.Upsample(
            #             scale_factor=module.kernel_size,
            #             mode='trilinear',
            #             align_corners=False
            #         )
            #     )
            # set_module(network, name, upscale_conv)

        return network