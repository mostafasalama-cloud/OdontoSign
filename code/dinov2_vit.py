"""
Compact re-implementation of the DINOv2 ViT backbones (ViT-S/14, ViT-B/14,
no registers) with the same state_dict keys as facebookresearch/dinov2, so the
official weights load with torch only (no torchvision / xformers / hub code).
Output = layer-normed CLS token (the 'x_norm_clstoken' of the official model).
Verified numerically against the official implementation (tests/).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = {
    "dinov2_vits14": ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth", 384, 12, 6),
    "dinov2_vitb14": ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth", 768, 12, 12),
}


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 14, 14)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoViT(nn.Module):
    def __init__(self, dim, depth, heads, n_pos=37 * 37):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_pos + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.embed_dim = dim

    def interpolate_pos(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and h == w:
            return self.pos_embed
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        M = int(math.sqrt(N))
        w0, h0 = w // 14, h // 14
        # identical to the official code (interpolate_offset=0.1, no antialias)
        patch_pos = F.interpolate(
            patch_pos.reshape(1, M, M, -1).permute(0, 3, 1, 2),
            scale_factor=((w0 + 0.1) / M, (h0 + 0.1) / M), mode="bicubic", antialias=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)
        return torch.cat([cls_pos, patch_pos.to(cls_pos.dtype)], 1)

    def forward(self, x):
        B, _, h, w = x.shape
        t = self.patch_embed(x)
        t = torch.cat([self.cls_token.expand(B, -1, -1), t], 1)
        t = t + self.interpolate_pos(t, w, h)
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)[:, 0]


def dinov2(name="dinov2_vits14", pretrained=True):
    url, dim, depth, heads = WEIGHTS[name]
    m = DinoViT(dim, depth, heads)
    if pretrained:
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
        m.load_state_dict(sd, strict=True)
    return m
