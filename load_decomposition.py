import torch
import torch.nn as nn
import numpy as np
from tensorly.decomposition import partial_tucker
import tensorly
import torchvision
from typing import Callable, Dict, List, NewType, Optional, Set, Union
from ResNet_imagenet import *
from ResNet_imageNet_tucker import *
from torchvision.models.vgg import *


def decompose_net(load_path, rate):
    model = torch.load(load_path)
    save_path = str(rate) + load_path
    for name in list(model.keys()):
        param = model[name]
        if len(param.size()) == 4 and param.size()[2] == 3 and param.size()[3] == 3:
            rank0 = int(param.size()[0] * rate)
            rank1 = int(param.size()[1] * rate)
            param = param.permute(0, 2, 3, 1)
            core, factors = partial_tucker(param.detach().numpy(), modes=[0, 1, 2, 3], rank=[rank0, 3, 3, rank1],
                                           init='svd')
            core = core.reshape(rank0 * 3, rank1 * 3)
            layer_name = name[:-6]
            model[layer_name + 'tucker_a'] = torch.from_numpy(factors[3].copy())
            model[layer_name + 'tucker_b'] = torch.from_numpy(factors[2].copy())
            model[layer_name + 'tucker_c'] = torch.from_numpy(factors[1].copy())
            model[layer_name + 'tucker_d'] = torch.from_numpy(factors[0].copy())
            model[layer_name + 'tucker_g'] = torch.from_numpy(core.copy())
            model[layer_name + 'stn.theta'] = torch.Tensor([[1., 0., 0.], [0., 1., 0.]]).expand(param.size()[0], 2, 3)
            del model[name]
    torch.save(model, save_path)


# m = resnet18(pretrained=True)
# torch.save(m.state_dict(), 'resnet18.pth')
# decompose_net('resnet18.pth', 0.25)
# decompose_net('resnet18.pth', 0.5)
# decompose_net('resnet18.pth', 0.75)


m = resnet50(pretrained=True)
torch.save(m.state_dict(), 'resnet50.pth')
decompose_net('resnet50.pth', 0.25)
decompose_net('resnet50.pth', 0.5)
decompose_net('resnet50.pth', 0.75)

