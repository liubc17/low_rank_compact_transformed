import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


class STN(nn.Module):
    def __init__(self, h, w, out_ch, affine=True):
        super().__init__()
        if affine:
            self.theta = nn.Parameter(torch.Tensor([[1., 0., 0.], [0., 1., 0.]]).expand(out_ch, 2, 3).cuda())
        else:
            theta = nn.Parameter(torch.zeros(out_ch, 1).cuda())
            sin = torch.sin(theta)
            cos = torch.cos(theta)
            self.theta = torch.cat((cos, sin, torch.zeros(out_ch, 1).cuda(), -sin, cos, torch.zeros(out_ch, 1).cuda()),
                                   1).reshape(out_ch, 2, 3)
        self.h = h
        self.w = w
        self.affine = affine

    def forward(self, x):
        affine_gird_points = F.affine_grid(self.theta, torch.Size((x.size(0), x.size(1), self.h, self.w)))
        # print(self.theta)
        output = F.grid_sample(x, affine_gird_points)
        return affine_gird_points, output


class TuckerLayer(nn.Module):
    def __init__(self, in_ch, out_ch, ch_com_rate=0.5, kernel_size=3, compress_size=3, stride=1,
                 bias=None, affine=True, group=True):
        super(TuckerLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = int((kernel_size-1)/2)
        self.bias = bias
        self.group = group

        if in_ch <= 3:
            in_ch_com = in_ch
        else:
            in_ch_com = int(in_ch * ch_com_rate)
        out_ch_com = int(out_ch * ch_com_rate)

        self.tucker_a = nn.Parameter(torch.zeros(in_ch, in_ch_com).cuda())
        self.tucker_b = nn.Parameter(torch.zeros(kernel_size, compress_size).cuda())
        self.tucker_c = nn.Parameter(torch.zeros(kernel_size, compress_size).cuda())
        self.tucker_d = nn.Parameter(torch.zeros(out_ch, out_ch_com).cuda())
        self.tucker_g = nn.Parameter(torch.zeros(out_ch_com * compress_size, compress_size * in_ch_com).cuda())
        init.kaiming_normal_(self.tucker_a)
        init.kaiming_normal_(self.tucker_b)
        init.kaiming_normal_(self.tucker_c)
        init.kaiming_normal_(self.tucker_d)
        init.kaiming_normal_(self.tucker_g)

        self.stn = STN(kernel_size, kernel_size, out_ch, affine=affine)
        if not group:
            self.stn_list = []
            for i in range(in_ch):
                self.stn_list.append(STN(kernel_size, kernel_size, out_ch, affine=affine))
            self.in_ch = in_ch

    def forward(self, x):
        kron1 = torch.kron(self.tucker_d, self.tucker_c)
        kron2 = torch.kron(self.tucker_b, self.tucker_a)
        mat1 = torch.matmul(self.tucker_g, kron2.transpose(0, 1))
        mat2 = torch.matmul(kron1, mat1).reshape([self.out_ch, self.kernel_size, self.kernel_size, self.in_ch])
        weight = mat2.permute(0, 3, 1, 2)
        if self.group:
            grid, kernel = self.stn(weight)
            out = F.conv2d(x, kernel, self.bias, stride=self.stride, padding=self.padding)
            return out
        else:
            kernel_list = []
            grid_list = []
            for i in range(self.in_ch):
                g, k = self.stn_list[i](weight[:, i, :, :].unsqueeze(1))
                grid_list.append(g)
                kernel_list.append(k)
            kernel = torch.cat(kernel_list, dim=1)
            out = F.conv2d(x, kernel, self.bias, stride=self.stride, padding=self.padding)
            return out
