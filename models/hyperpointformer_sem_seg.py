import torch
import torch.nn as nn
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from models.transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.wq = nn.Linear(head_dim, dim , bias=False)
        self.wk = nn.Linear(head_dim, dim , bias=False)
        self.wv = nn.Linear(head_dim, dim , bias=False)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True) 
        
    def forward(self, feat_1, feat_2):
        B, N, C = feat_1.size()
        pre = feat_1
        q = self.wq(feat_1.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3) # B x N x C -> B x N x H x (C/H)
        k = self.wk(feat_2.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3) # B x N x C -> B x N x H x (C/H)
        v = self.wv(feat_2.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3) # B x N x C -> B x N x H x (C/H)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
        x = x.reshape(B, N, C * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        res = self.gamma*x + pre
        return res

class Backbone(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, n_c, d_points, transformer_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.fc1_geom = nn.Sequential(
            nn.Linear(6, 32), # 6 is xyz and normxyz
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.fc1_spect = nn.Sequential(
            nn.Linear(12, 32), # 12 is RGB, I and 8-HSI
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1_geom = TransformerBlock(32, transformer_dim, nneighbor)
        self.transformer1_spect = TransformerBlock(32, transformer_dim, nneighbor)
        self.cross1_geom = CrossAttention(dim=32)
        self.cross1_spect = CrossAttention(dim=32)
        self.transition_downs_geom = nn.ModuleList()
        self.transformers_geom = nn.ModuleList()
        self.transition_downs_spect = nn.ModuleList()
        self.transformers_spect = nn.ModuleList()
        self.crosses_geom = nn.ModuleList()
        self.crosses_spect = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs_geom.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers_geom.append(TransformerBlock(channel, transformer_dim, nneighbor))
            self.transition_downs_spect.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers_spect.append(TransformerBlock(channel, transformer_dim, nneighbor))
            self.crosses_geom.append(CrossAttention(dim=channel))
            self.crosses_spect.append(CrossAttention(dim=channel))
        self.nblocks = nblocks
    
    def forward(self, x):
        xyz = x[..., :3]
        geometrical = x[..., [0,1,2,15,16,17]] # xyz and normxyz
        spectral = x[..., [3,4,5,6,7,8,9,10,11,12,13,14]] # RGB I 8-HSI
        points = self.transformer1_geom(xyz, self.fc1_geom(geometrical))[0]
        points_spectral = self.transformer1_spect(xyz, self.fc1_spect(spectral))[0]
        points_cr = self.cross1_geom(points, points_spectral)
        points_spectral_cr = self.cross1_spect(points_spectral, points)
        summed_points = points_cr + points_spectral_cr

        xyz_and_feats = [(xyz, points)]
        xyz_and_feats_spectral = [(xyz, points_spectral)]
        xyz_and_feats_summed = [(xyz, summed_points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs_geom[i](xyz, points)
            xyz, points_spectral = self.transition_downs_spect[i](xyz, points_spectral)
            points = self.transformers_geom[i](xyz, points)[0]
            points_spectral = self.transformers_spect[i](xyz, points_spectral)[0]
            points_cr = self.crosses_geom[i](points, points_spectral)
            points_spectral_cr = self.crosses_spect[i](points_spectral, points) 
            summed_points = points_cr + points_spectral_cr 
            xyz_and_feats.append((xyz, points))
            xyz_and_feats_spectral.append((xyz, points_spectral))
            xyz_and_feats_summed.append((xyz, summed_points))
        return summed_points, xyz_and_feats_summed


class get_model(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, n_c, d_points, transformer_dim):
        super(get_model, self).__init__()   
        self.backbone = Backbone(npoints, nblocks, nneighbor, n_c, d_points, transformer_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    
    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
        
        return self.fc3(points)


