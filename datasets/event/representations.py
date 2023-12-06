import torch


class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

    # def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, t: torch.Tensor):
        self.voxel_grid = self.voxel_grid.to(pol.device)
        voxel_grid = self.voxel_grid.clone()
        vol_size = self.voxel_grid.shape
        C, H, W = self.voxel_grid.shape
        
        t_min = t.min()
        t_max = t.max()
        t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min))

        mask = (x < W) & (x >= 0) & (y < H) & (y >= 0) & (t >= 0)

        ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled[mask].squeeze())
    
        
        inds_fl, vals_fl = create_update(x[mask], y[mask],
                                        ts_fl[0], ts_fl[1],
                                        pol[mask],
                                        vol_size)
        # import pdb; pdb.set_trace() 
        # voxel_grid.view(-1).put_(inds_fl, vals_fl, accumulate=True)
        voxel_grid.view(-1).put_(inds_fl, vals_fl.float(), accumulate=True)

        inds_ce, vals_ce = create_update(x[mask], y[mask],
                                        ts_ce[0], ts_ce[1],
                                        pol[mask],
                                        vol_size)
        # voxel_grid.view(-1).put_(inds_ce, vals_ce, accumulate=True)
        voxel_grid.view(-1).put_(inds_ce, vals_ce.float(), accumulate=True)
        return voxel_grid
    

class ReconVoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            value = 2*pol-1
            
            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:
                        xlim = xlim.float()
                        ylim = ylim.float()
                        tlim = tlim.float()
                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        
                        interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


import torch
import torch.nn as nn

import numpy as np

def none_safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if batch:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)
    else:
        return {}

def init_weights(m):
    """ Initialize weights according to the FlowNet2-pytorch from nvidia """
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.0001, b=0.0001)
        nn.init.xavier_uniform_(m.weight, gain=0.001)

    if isinstance(m, nn.Conv1d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

    if isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

def num_trainable_parameters(module):
    trainable_parameters = filter(lambda p: p.requires_grad,
                                  module.parameters())
    return sum([np.prod(p.size()) for p in trainable_parameters])


def num_parameters(network):
    n_params = 0
    modules = list(network.modules())

    for mod in modules:
        parameters = mod.parameters()
        n_params += sum([np.prod(p.size()) for p in parameters])
    return n_params

def calc_floor_ceil_delta(x): 
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]

def create_update(x, y, t, dt, p, vol_size):
    
    assert (x>=0).byte().all() and (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all() and (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() and (t<vol_size[0] // 2).byte().all()
 
    vol_mul = torch.where(p < 0,
                          torch.ones(p.shape, dtype=torch.long) * vol_size[0] // 2,
                          torch.zeros(p.shape, dtype=torch.long))

    inds = (vol_size[1]*vol_size[2]) * (t + vol_mul)\
         + (vol_size[2])*y\
         + x

    vals = dt

    return inds, vals

def gen_discretized_event_volume(events, vol_size):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size)

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min))

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    
    import pdb; pdb.set_trace()
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     events[:, 3],
                                     vol_size)
        
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],
                                     events[:, 3],
                                     vol_size)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)
    return volume
 
def create_batch_update(x, dx, y, dy, t, dt, p, vol_size):
    assert (x>=0).byte().all() and (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all() and (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() and (t<vol_size[0] // 2).byte().all()

    vol_mul = torch.where(p < 0,
                          torch.ones(p.shape, dtype=torch.long) * vol_size[0] // 2,
                          torch.zeros(p.shape, dtype=torch.long))

    batch_inds = torch.arange(x.shape[0], dtype=torch.long)[:, None]
    batch_inds = batch_inds.repeat((1, x.shape[1]))
    batch_inds = torch.reshape(batch_inds, (-1,))

    dx = torch.reshape(dx, (-1,))
    dy = torch.reshape(dy, (-1,))
    dt = torch.reshape(dt, (-1,))
    x = torch.reshape(x, (-1,))
    y = torch.reshape(y, (-1,))
    t = torch.reshape(t, (-1,))
    vol_mul = torch.reshape(vol_mul, (-1,))
    
    inds = vol_size[1]*vol_size[2]*vol_size[3] * batch_inds \
         + (vol_size[2]*vol_size[3]) * (t + vol_mul) \
         + (vol_size[3])*y \
         + x

    vals = dx * dy * dt

    return inds, vals


def gen_batch_discretized_event_volume(events, vol_size):
    # vol_size is [b, t, x, y]
    # events are BxNx4
    batch = events.shape[0]
    volume = events.new_zeros(vol_size)

    # Each is BxN
    x = events[..., 0].long()
    y = events[..., 1].long()
    t = events[..., 2]

    # Dim is now Bx1
    t_min = t.min(dim=1, keepdim=True)
    t_max = t.max(dim=1, keepdim=True)
    t_scaled = (t-t_min) * ((vol_size[1]//2 - 1) / (t_max-t_min))

    xs_fl, xs_ce = calc_floor_ceil_delta(x)
    ys_fl, ys_ce = calc_floor_ceil_delta(y)
    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled)
    
    inds_fl, vals_fl = create_batch_update(xs_fl[0], xs_fl[1],
                                           ys_fl[0], ys_fl[1],
                                           ts_fl[0], ts_fl[1],
                                           events[..., 3],
                                           vol_size)
    
    
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_batch_update(xs_ce[0], xs_ce[1],
                                           ys_ce[0], ys_ce[1],
                                           ts_ce[0], ts_ce[1],
                                           events[:, 3],
                                           vol_size)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)

    

    return volume

"""
 Network output is BxHxWxNx4, all between -1 and 1. Each 4-tuple is [x, y, t, p], 
 where [x, y] are relative to the center of the grid cell at that hxw pixel.
 This function scales this output to values in the range:
 [[0, volume_size[0]], [0, volume_size[1]], [0, volume_size[2]], [-1, 1]]
"""
def scale_events(events, volume_size, device='cuda'):
    # Compute the center of each grid cell.
    scale = volume_size[0] / events.shape[1]
    x_range = torch.arange(events.shape[2]).to(device) * scale + scale / 2
    y_range = torch.arange(events.shape[1]).to(device) * scale + scale / 2
    x_offset, y_offset = torch.meshgrid(x_range, y_range)
    
    t_scale = (volume_size[2] - 1) / 2.
    # Offset the timestamps from [-1, 1] to [0, 2].
    t_offset = torch.ones(x_offset.shape).to(device) * t_scale
    p_offset = torch.zeros(x_offset.shape).to(device)
    offset = torch.stack((x_offset.float(), y_offset.float(), t_offset, p_offset), dim=-1)
    offset = offset[None, ..., None, :]

    # Scale the [x, y] values to [-scale/2, scale/2] and
    # t to [-volume_size[2] / 2, volume_size[2] / 2].
    output_scale = torch.tensor((scale / 2, scale / 2, t_scale, 1))\
                        .to(device).reshape((1, 1, 1, 1, -1))

    # Scale the network output
    events *= output_scale

    # Offset the network output
    events += offset

    events = torch.reshape(events, (events.shape[0], -1, 4))

    return events
