import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm
from models.unet import unet
import pdb
import math
import pdb
import random

class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


import torchvision.transforms as transforms
rand_crop = transforms.RandomCrop(192, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
resize = transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias='warn')

class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim
        
        input_channels = 2*dim[0]
        # self.conv = nn.Conv2d(input_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.unet = unet.UNet(num_input_channels=18,
        #                  num_output_channels=1,
        #                  skip_type='sum',
        #                  activation='sigmoid',
        #                  num_encoders=4,
        #                  base_num_channels=32,
        #                  num_residual_blocks=2,
        #                  norm=None,
        #                  use_upsample_conv=True)
        
        self.unet = unet.UNet(num_input_channels=18,
                         num_output_channels=1,
                         skip_type='sum',
                         activation='sigmoid',
                         num_encoders=3,
                         base_num_channels=32,
                         num_residual_blocks=2,
                         norm=None,
                         use_upsample_conv=True)
        
        # self.unet = unet.UNet(num_input_channels=18,
        #                  num_output_channels=1,
        #                  skip_type='sum',
        #                  activation='sigmoid',
        #                  num_encoders=2,
        #                  base_num_channels=32,
        #                  num_residual_blocks=2,
        #                  norm=None,
        #                  use_upsample_conv=True)
        
        print("model parameter: ", sum(p.numel() for p in self.unet.parameters() if p.requires_grad))

    def crop_and_resize_to_resolution(self, x, output_resolution=(256, 256)):
        B, C, H, W = x.shape
        
        if H > W:
            w = (H-W) // 2
            ratio = output_resolution[0] / H
            pad = (w,w,0,0)
            x = F.pad(x, pad, "constant", 0.0)
            new_pad = (math.ceil(w*ratio), math.ceil(w*ratio), 0, 0)
        else:
            h = (W-H) // 2
            ratio = output_resolution[0] / W
            pad = (0,0,h,h)
            x = F.pad(x, pad, "constant", 0.0)
            new_pad = (0, 0, math.ceil(h*ratio), math.ceil(h*ratio))
        
        # if H > W:
        #     h = H // 2
        #     x = x[:, :, h - W // 2:h + W // 2, :]
        # else:
        #     h = W // 2
        #     x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=output_resolution)

        return x, new_pad
    
    def unpadding(self, x, pad):
        if pad[0] > pad[2]:
            x = x[:,:, :, pad[0]:-pad[1]]
        else:
            x = x[:,:, pad[2]:-pad[3], :]
        return x

    def forward(self, events, aug=False, aug_temp=False, for_cont=False):
        # if 82340 in b_idx:
        #     pdb.set_trace()
        # points is a list, since events can have any size            
        B = int((1+events[-1,-1]).item())
        num_voxels = int(2 * np.prod(self.dim) * B)
        
        mini_vox = events[0].new_full([num_voxels,], fill_value=0)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        
        
        x, y, t, p, b = events.t()
        # print(x.max(), y.max(), b)
        # print(b_idx)

        # normalizing timestamps
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()
        
        if aug_temp:
            # pdb.set_trace()
            tot_num = t.shape[0]
            # indice = random.sample(range(tot_num), tot_num * 95 // 100)
            indice = random.sample(range(tot_num), tot_num * 80 // 100)
            # indice = random.sample(range(tot_num), tot_num * 90 // 100)
            crop_index = torch.tensor(indice)
            
            # period = 0.9
            # start = random.uniform(0, 1 - period) 
            # crop_index = torch.where((t>start) & (t<(start + period)))
            crop_x = x[crop_index].long()
            crop_y = y[crop_index].long()
            crop_t = t[crop_index].long()
            crop_p = p[crop_index].long()
            crop_b = b[crop_index].long()

            crop_idx_before_bins = crop_x \
                          + W * crop_y \
                          + 0 \
                          + W * H * C * crop_p \
                          + W * H * C * 2 * crop_b
            for i_bin in range(C):
                values = crop_t * self.value_layer.forward(crop_t-i_bin/(C-1))
                # draw in voxel grid
                idx = crop_idx_before_bins + W * H * i_bin
                if idx.max() >= vox.shape[0]:
                    pdb.set_trace()
                mini_vox.put_(idx.long(), values, accumulate=True)
            mini_vox = mini_vox.view(-1, 2, C, H, W)
            mini_vox = torch.cat([mini_vox[:, 0, ...], mini_vox[:, 1, ...]], 1)
            # vox = self.conv(vox)
            mini_vox, mini_pad = self.crop_and_resize_to_resolution(mini_vox)
        
        # p = (p+1)/2  # maps polarity to 0, 1
        # import pdb; pdb.set_trace()
        x = x.long()
        y = y.long()
        p = p.long()
        b = b.long()
        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b
        # if 82340 in idx:
        #     pdb.set_trace()
        for i_bin in range(C):
            # if 82340 in b_idx:
            #     print(i_bin)
            #     # pdb.set_trace()
            #     if i_bin == 8:
            #         pdb.set_trace()
            values = t * self.value_layer.forward(t-i_bin/(C-1))
            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            if idx.max() >= vox.shape[0]:
                pdb.set_trace()
            vox.put_(idx.long(), values, accumulate=True)
        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)
        # vox = self.conv(vox)
        vox, pad = self.crop_and_resize_to_resolution(vox)
        
        # aug for ssl
        if aug:
            b, c, h, w = vox.shape
            th, tw = 128, 128
            p_w = pad[0]
            p_h = pad[2]
            x1 = random.randint(p_w, w - p_w - tw)
            y1 = random.randint(p_h, h - p_h - th)

            vox_crop = vox[:, :, y1:y1+th, x1:x1+tw]  
            # vox_crop = resize(vox_crop)
            # vox = torch.cat((vox, vox_crop), dim=0)
            vox_crop = self.unet(vox_crop)

        if aug_temp:
            vox_stack = torch.cat((vox, mini_vox), dim=0)
            vox_stack = self.unet(vox_stack)
            vox_stack = self.unpadding(vox_stack, pad)
        else:
            vox = self.unet(vox)
            vox = self.unpadding(vox, pad)
        
        if aug:
            if for_cont:
                return vox, vox_crop
            else:
                loss = (vox[:, :, y1-p_h:y1-p_h+th, x1-p_w:x1-p_w+tw] - vox_crop).abs().mean()
                return vox, vox_crop, loss
        elif aug_temp:
            if for_cont:
                return vox_stack
            else:
                vox, mini_vox = torch.chunk(vox_stack, chunks=2, dim=0)
                loss = (vox - mini_vox).abs().mean()
                # loss = (vox.detach() - mini_vox).abs().mean()
                # loss = (vox - mini_vox.detach()).abs().mean()
                return vox, mini_vox, loss
        else:
            return vox

import clip
from lightly.loss.ntx_ent_loss import NTXentLoss
from copy import deepcopy
# clip_model, _ = clip.load("ViT-B/32", device='cpu')

class Event_Reconst(nn.Module):
    def __init__(self,
                #  voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 voxel_dimension=(9,480,640),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True):
        super().__init__()
        
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        self.crop_dimension = crop_dimension
      
    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        # if H > W:
        #     h = H // 2
        #     x = x[:, :, h - W // 2:h + W // 2, :]
        # else:
        #     h = W // 2
        #     x = x[:, :, :, h - H // 2:h + H // 2]

        # x = F.interpolate(x, size=output_resolution, mode='bicubic')
        x = F.interpolate(x, size=output_resolution, mode='bilinear')

        return x

    def forward(self, x, aug=False, aug_temp=False, for_cont=False):
        if aug:
            if for_cont:
                vox, mini_vox = self.quantization_layer.forward(x, aug=aug, for_cont=for_cont)
                mini_vox_cropped = self.crop_and_resize_to_resolution(mini_vox, self.crop_dimension)
                mini_vox_cropped = mini_vox_cropped.repeat(1,3,1,1)
            else:
                vox, mini_vox, loss  = self.quantization_layer.forward(x, aug=aug)
                mini_vox = mini_vox.repeat(1,3,1,1)
        elif aug_temp:
            if for_cont:
                vox_stack = self.quantization_layer.forward(x, aug_temp=aug_temp, for_cont=for_cont)
                vox_stack = self.crop_and_resize_to_resolution(vox_stack, self.crop_dimension)
                vox_stack = vox_stack.repeat(1,3,1,1)
            else:
                vox, mini_vox, loss  = self.quantization_layer.forward(x, aug_temp=aug_temp)
                mini_vox = mini_vox.repeat(1,3,1,1)
        else:
            vox = self.quantization_layer.forward(x, aug=aug)
        
        if not (aug_temp and for_cont):
            vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
            vox_cropped = vox_cropped.repeat(1,3,1,1)
        
        if aug:
            if for_cont:
                return vox_cropped, mini_vox_cropped
            else:
                return vox_cropped, mini_vox, loss
        elif aug_temp:
            if for_cont:
                return vox_stack
            else:
                return vox_cropped, mini_vox, loss
        else:
            return vox_cropped
