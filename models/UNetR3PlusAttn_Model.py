import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from einops import rearrange
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class UNetSwin(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_channels=3, num_classes=9,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]):
        
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.name = f"SwimUnet3+Attn_model_{''.join(random.sample([str(x) for x in range(10)], 4))}"
        self.filters = [64, 128, 256, 512, 1024]
        self.hidden_dim = 1

        self.patch_embed = PatchEmbed(img_size = img_size, patch_size=patch_size, 
                                      in_chans = in_channels, embed_dim=embed_dim, 
                                      norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        #Encoder layers
        self.encoder_layers = []

        for i in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i),
                               input_resolution=(patches_resolution[0] // (2 ** i),
                                                 patches_resolution[1] // (2 ** i)),
                               depth=depths[i],
                               num_heads=num_heads[i],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i])
            
            self.encoder_layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        # for bly in self.encoder_layers:
        #     bly._init_respostnorm()

        """ CNN Decoder """
        ## Decoder 1
        self.decon_1 = DeconvBlock(self.hidden_dim*4, self.filters[3])
        self.skip_1_z9 = nn.Sequential(
            DeconvBlock(self.hidden_dim*4, self.filters[3]),
            ConvBlock(self.filters[3], self.filters[3])
        )

        self.skip_1_z6 = nn.Sequential(
            DeconvBlock(self.hidden_dim*4, self.filters[3]),
            ConvBlock(self.filters[3], self.filters[3]),
            # DeconvBlock(self.filters[3], self.filters[3]),
            ConvBlock(self.filters[3], self.filters[3])
        )

        self.skip_1_z3 = nn.Sequential(
            DeconvBlock(self.hidden_dim*4, self.filters[3]),
            ConvBlock(self.filters[3], self.filters[3]),
            # DeconvBlock(self.filters[3], self.filters[3]),
            ConvBlock(self.filters[3], self.filters[3]),
            # DeconvBlock(self.filters[3], self.filters[3]),
            ConvBlock(self.filters[3], self.filters[3])
        )

        self.conv_1 = SingleConvolution(in_channels = self.filters[3] * 2, out_channels = self.filters[3])

        ## Decoder 2
        self.decon_2 = DeconvBlock(self.filters[3], self.filters[2])
        self.skip_2_z6 = nn.Sequential(
            DeconvBlock(self.hidden_dim*4, self.filters[2]),
            ConvBlock(self.filters[2], self.filters[2]),
            DeconvBlock(self.filters[2], self.filters[2]),
            ConvBlock(self.filters[2], self.filters[2])
        )

        self.skip_2_z3 = nn.Sequential(
            DeconvBlock(self.hidden_dim*4, self.filters[2]),
            ConvBlock(self.filters[2], self.filters[2]),
            DeconvBlock(self.filters[2], self.filters[2]),
            ConvBlock(self.filters[2], self.filters[2]),
            # DeconvBlock(self.filters[2], self.filters[2]),
            ConvBlock(self.filters[2], self.filters[2])
        )

        self.conv_2 = SingleConvolution(in_channels = self.filters[2] * 2, out_channels=self.filters[2])

        ## Decoder 3
        self.decon_3 = DeconvBlock(256, 128)
        self.skip_3_z3 = nn.Sequential(
            DeconvBlock(self.hidden_dim*4, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128)
        )

        self.conv_3 = SingleConvolution(in_channels = self.filters[1] * 2, out_channels=self.filters[1])


        ## Decoder 4
        self.decon_4 = nn.Sequential(DeconvBlock(128, 64),
                                     ConvBlock(64, 64),
                                     DeconvBlock(64, 64),
                                     ConvBlock(64, 64)
        )
        self.skip_4_z0 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )

        self.conv_4 = SingleConvolution(in_channels = self.filters[0] * 2, out_channels=self.filters[0])

        """ Output """
        self.output = nn.Conv2d(self.filters[0], num_classes, kernel_size=1, padding=0)

        #Deep Supervision
        self.final_conv_1 = nn.Conv2d(in_channels = self.filters[0], out_channels = num_classes, kernel_size = 1)
        self.final_conv_2 = nn.Conv2d(in_channels = self.filters[1], out_channels = num_classes, kernel_size = 1)
        self.final_conv_3 = nn.Conv2d(in_channels = self.filters[2], out_channels = num_classes, kernel_size = 1)
        self.final_conv_4 = nn.Conv2d(in_channels = self.filters[3], out_channels = num_classes, kernel_size = 1)
        self.final_conv_5 = nn.Conv2d(in_channels = self.filters[4] * 4, out_channels = num_classes, kernel_size = 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
            
        x = self.pos_drop(x)

        x_downsample = []

        for layer in self.encoder_layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        print(x.shape)
        print(f"{x_downsample[0].shape}, {x_downsample[1].shape}, {x_downsample[2].shape}, {x_downsample[3].shape}")

        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        
        if self.norm is not None:
            x = self.norm(x)

        return x

class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False).to(torch.device('cuda'))
        self.norm = norm_layer(2 * dim).to(torch.device('cuda'))

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = x.to(torch.device('cuda'))

        x = self.reduction(x)
        x = self.norm(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = []

        for i in range(self.depth):

            block = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            
            self.blocks.append(block)

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            
            x = x.to(torch.device('cuda'))
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim).to(torch.device('cuda'))
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim).to(torch.device('cuda'))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = self.window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C).to(torch.device('cuda'))  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x
    
    def window_partition(self, x, window_size):
        
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

        return windows.to(torch.device('cuda'))

    def window_reverse(self, windows, window_size, H, W):
        
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

        return x.to(torch.device('cuda'))

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim).to(torch.device('cuda'))
        self.proj_drop = nn.Dropout(proj_drop).to(torch.device('cuda'))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)).to(torch.device('cuda'))
        qkv = F.linear(input=x, weight=self.qkv.weight.to(torch.device('cuda')), bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0].to(torch.device('cuda')), qkv[1].to(torch.device('cuda')), qkv[2].to(torch.device('cuda'))  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)).to(torch.device('cuda'))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp().to(torch.device('cuda'))
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0).to(torch.device('cuda'))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        attn = attn.to(torch.device('cuda'))

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).to(torch.device('cuda'))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features).to(torch.device('cuda'))
        self.act = act_layer().to(torch.device('cuda'))
        self.fc2 = nn.Linear(hidden_features, out_features).to(torch.device('cuda'))
        self.drop = nn.Dropout(drop).to(torch.device('cuda'))

    def forward(self, x):
        
        x = x.to(torch.device('cuda'))
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class SingleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv_layers(x)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)

# class PatchEmbed1(nn.Module):
#     """
#     Image to Patch Embedding
#     """
#     def __init__(self, img_shape, patch_size, embed_dim):
#         super().__init__()

#         c, h, w = img_shape
#         patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
#         assert h % patch_size[0] == 0 and w % patch_size[1] == 0,\
#             'image dimensions must be divisible by the patch size'

#         self.patch_size = patch_size
#         self.num_patches = (h // patch_size[0]) * (w // patch_size[1])
#         patch_dim = c * patch_size[0] * patch_size[1]
#         assert self.num_patches > 16,\
#             f'your number of patches ({self.num_patches}) is too small for ' \
#             f'attention to be effective. try decreasing your patch size'

#         self.projection = nn.Linear(patch_dim, embed_dim)

#     def forward(self, x):
#         p1, p2 = self.patch_size

#         b, c, h, w = x.shape

#         x = torch.reshape(x, (b, ((h//p1) * (w//p2)), (c * p1 * p2)))

#         # x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2)

#         print(x.shape)

#         x = self.projection(x)
#         return x

# class PatchifyV2(nn.Module):
#     def __init__(self, patch_size=56):
#         super().__init__()
#         self.p = patch_size
#         self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         # x -> B c h w
#         bs, c, h, w = x.shape
        
#         x = self.unfold(x)
#         # x -> B (c*p*p) L
        
#         # Reshaping into the shape we want
#         a = x.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
#         # a -> ( B no.of patches c p p )
#         return a

# class PatchEmbedding1(nn.Module):
#     """Create 1D Sequence lernable embedding vector from a 2D input image

#     Args:
#         in_channels (int): Nunber of Color Channels. Defaults to 3
#         patch_size (int): Target size for each patch. Defaults to 8
#         embedding_dim (int): Size of image embedding. Defaults to 768 (ViT base) 
#     """

#     def __init__(self,
#                  in_channels:int = 3,
#                  patch_size:int = 8,
#                  embedding_dim:int = 768
#                  ):
        
#         super().__init__()
        
#         self.patch_size = patch_size 

#         # Layer to create patch embeddings
#         self.patcher = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=embedding_dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#             padding=0
#         )

#         # Layer to flatten the flatten the feature map dim. to a single vector
#         self.flatten = nn.Flatten(
#             start_dim=2, end_dim=3
#         )
    
#     def forward(self, x):
#         image_size = x.shape[-1]
#         assert image_size % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_size}, patch size: {self.patch_size}"

#         x_patched = self.patcher(x)
#         x_flattened = self.flatten(x_patched)

#         return x_flattened.permute(0, 2, 1)

# class UNetR3PlusAttnModel(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
        
#         self.in_channels = in_channels
#         self.num_blocks = 5 - 1

#         self.name = f"unet3R+Attn_model_{''.join(random.sample([str(x) for x in range(10)], 4))}"
#         self.filters = [64, 128, 256, 512, 1024]

#         self.num_layers = 12
#         self.hidden_dim = 66
#         self.mlp_dim = 128 #3072
#         self.num_heads = 6
#         self.dropout_rate = 0.1
#         self.num_patches = 1024 #H * W / patch_size * patch_size
#         self.patch_size = 16

#         """ Patch + Position Embeddings """
#         self.patch_embed_1 = nn.Linear(
#             self.patch_size * self.patch_size * in_channels, self.hidden_dim
#         )

#         # self.patch_embed = PatchEmbed(img_shape = (3, 512, 512), patch_size = 16, embed_dim = self.hidden_dim)
#         self.patch_embed = PatchEmbedding(in_channels = 3, patch_size = 16, embedding_dim = self.hidden_dim)

#         self.positions = torch.arange(start = 0, end = self.num_patches, step = 1, dtype=torch.int32)
#         self.pos_embed = nn.Embedding(self.num_patches, self.hidden_dim)

#         """ Transformer Encoder """
#         self.trans_encoder_layers = []

#         for i in range(self.num_layers):
#             layer = nn.TransformerEncoderLayer(
#                 d_model = self.hidden_dim,
#                 nhead = self.num_heads,
#                 dim_feedforward = self.mlp_dim,
#                 dropout = self.dropout_rate,
#                 activation = nn.GELU(),
#                 batch_first = True
#             ).to(torch.device('cuda'))
#             self.trans_encoder_layers.append(layer)

#         """ CNN Decoder """
#         ## Decoder 1
#         self.decon_1 = DeconvBlock(self.hidden_dim*4, self.filters[3])
#         self.skip_1_z9 = nn.Sequential(
#             DeconvBlock(self.hidden_dim*4, self.filters[3]),
#             ConvBlock(self.filters[3], self.filters[3])
#         )

#         self.skip_1_z6 = nn.Sequential(
#             DeconvBlock(self.hidden_dim*4, self.filters[3]),
#             ConvBlock(self.filters[3], self.filters[3]),
#             # DeconvBlock(self.filters[3], self.filters[3]),
#             ConvBlock(self.filters[3], self.filters[3])
#         )

#         self.skip_1_z3 = nn.Sequential(
#             DeconvBlock(self.hidden_dim*4, self.filters[3]),
#             ConvBlock(self.filters[3], self.filters[3]),
#             # DeconvBlock(self.filters[3], self.filters[3]),
#             ConvBlock(self.filters[3], self.filters[3]),
#             # DeconvBlock(self.filters[3], self.filters[3]),
#             ConvBlock(self.filters[3], self.filters[3])
#         )

#         self.conv_1 = SingleConvolution(in_channels = self.filters[3] * 2, out_channels = self.filters[3])

#         ## Decoder 2
#         self.decon_2 = DeconvBlock(self.filters[3], self.filters[2])
#         self.skip_2_z6 = nn.Sequential(
#             DeconvBlock(self.hidden_dim*4, self.filters[2]),
#             ConvBlock(self.filters[2], self.filters[2]),
#             DeconvBlock(self.filters[2], self.filters[2]),
#             ConvBlock(self.filters[2], self.filters[2])
#         )

#         self.skip_2_z3 = nn.Sequential(
#             DeconvBlock(self.hidden_dim*4, self.filters[2]),
#             ConvBlock(self.filters[2], self.filters[2]),
#             DeconvBlock(self.filters[2], self.filters[2]),
#             ConvBlock(self.filters[2], self.filters[2]),
#             # DeconvBlock(self.filters[2], self.filters[2]),
#             ConvBlock(self.filters[2], self.filters[2])
#         )

#         self.conv_2 = SingleConvolution(in_channels = self.filters[2] * 2, out_channels=self.filters[2])

#         ## Decoder 3
#         self.decon_3 = DeconvBlock(256, 128)
#         self.skip_3_z3 = nn.Sequential(
#             DeconvBlock(self.hidden_dim*4, 128),
#             ConvBlock(128, 128),
#             DeconvBlock(128, 128),
#             ConvBlock(128, 128),
#             DeconvBlock(128, 128),
#             ConvBlock(128, 128)
#         )

#         self.conv_3 = SingleConvolution(in_channels = self.filters[1] * 2, out_channels=self.filters[1])


#         ## Decoder 4
#         self.decon_4 = nn.Sequential(DeconvBlock(128, 64),
#                                      ConvBlock(64, 64),
#                                      DeconvBlock(64, 64),
#                                      ConvBlock(64, 64)
#         )
#         self.skip_4_z0 = nn.Sequential(
#             ConvBlock(3, 64),
#             ConvBlock(64, 64)
#         )

#         self.conv_4 = SingleConvolution(in_channels = self.filters[0] * 2, out_channels=self.filters[0])

#         """ Output """
#         self.output = nn.Conv2d(self.filters[0], num_classes, kernel_size=1, padding=0)

#         #Deep Supervision
#         self.final_conv_1 = nn.Conv2d(in_channels = self.filters[0], out_channels = num_classes, kernel_size = 1)
#         self.final_conv_2 = nn.Conv2d(in_channels = self.filters[1], out_channels = num_classes, kernel_size = 1)
#         self.final_conv_3 = nn.Conv2d(in_channels = self.filters[2], out_channels = num_classes, kernel_size = 1)
#         self.final_conv_4 = nn.Conv2d(in_channels = self.filters[3], out_channels = num_classes, kernel_size = 1)
#         self.final_conv_5 = nn.Conv2d(in_channels = self.filters[4] * 4, out_channels = num_classes, kernel_size = 1)

#         #Bilinear Upsampling
#         self.upscore6 = nn.Upsample(scale_factor = 32, mode = 'bilinear')###
#         self.upscore5 = nn.Upsample(scale_factor = 16, mode = 'bilinear')
#         self.upscore4 = nn.Upsample(scale_factor = 8, mode = 'bilinear')
#         self.upscore3 = nn.Upsample(scale_factor = 4, mode = 'bilinear')
#         self.upscore2 = nn.Upsample(scale_factor = 2, mode = 'bilinear')

#     def patchify(self, images, n_patches):
#         n, c, h, w = images.shape

#         # assert h == w, "Patchify method is implemented for square images only"

#         patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
#         patch_size = h // n_patches

#         for idx, image in enumerate(images):
#             for i in range(n_patches):
#                 for j in range(n_patches):
#                     patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
#                     patches[idx, i * n_patches + j] = patch.flatten()
#         return patches

#     def forward(self, inputs):
#         """ Patch + Position Embeddings """

#         patch_embed = self.patch_embed(inputs) #self.patchify(inputs, n_patches = 1024)

#         # patch_embed = self.patch_embed(inputs)   ## [8, 256, 768]

#         print(patch_embed.shape)

#         positions = self.positions.to(torch.device('cuda'))
#         pos_embed = self.pos_embed(positions)   ## [256, 768]

#         # print(pos_embed.shape)

#         x = patch_embed + pos_embed ## [8, 256, 768]

#         # print(x.shape)

#         """ Transformer Encoder """
#         skip_connection_index = [3, 6, 9, 12]
#         skip_connections = []

#         x = x.to(torch.device('cuda'))

#         for i in range(self.num_layers):
#             layer = self.trans_encoder_layers[i]
#             x = layer(x)

#             if (i+1) in skip_connection_index:
#                 skip_connections.append(x)

#         """ CNN Decoder """
#         z3, z6, z9, z12 = skip_connections

#         # print(z6.shape)

#         ## Reshaping
#         batch = inputs.shape[0]
#         z0 = inputs.view((batch, self.in_channels, 512, 512))

#         shape = (batch, self.hidden_dim*4, self.patch_size, self.patch_size)
#         z3 = z3.view(shape)
#         z6 = z6.view(shape)
#         z9 = z9.view(shape)
#         z12 = z12.view(shape)


#         ## Decoder 1
#         x1_z12 = self.decon_1(z12)

#         s1_z9 = self.skip_1_z9(z9)
#         # s1_z6 = self.skip_1_z6(z6)
#         # s1_z3 = self.skip_1_z3(z3)

#         # print(f"{x1_z12.shape}, {s1_z9.shape}, {s1_z6.shape}, {s1_z3.shape}")

#         hd4 = self.conv_1(torch.cat((x1_z12, s1_z9), 1)) # s1_z6, s1_z3), 1))

#         ## Decoder 2
#         x2_hd4 = self.decon_2(hd4)

#         s2_z6 = self.skip_2_z6(z6)
#         # s2_z3 = self.skip_2_z3(z3)
        
#         hd3 = self.conv_2(torch.cat((x2_hd4, s2_z6), 1)) #s2_z3), 1))

#         ## Decoder 3
#         x3_hd3 = self.decon_3(hd3)

#         s3_z3 = self.skip_3_z3(z3)
        
#         hd2 = self.conv_3(torch.cat((x3_hd3, s3_z3), 1))

#         ## Decoder 4
#         x4_hd2 = self.decon_4(hd2)
        
#         s4_z0 = self.skip_4_z0(z0)

#         # print(f"{x4_hd2.shape}, {s4_z0.shape}")
        
#         hd1 = self.conv_4(torch.cat((x4_hd2, s4_z0), 1))

#         """ Output """
#         # output = self.output(hd1)

#         d5 = self.final_conv_5(z12)
#         d5 = self.upscore5(d5)

#         d4 = self.final_conv_4(hd4)
#         d4 = self.upscore4(d4)

#         d3 = self.final_conv_3(hd3)
#         d3 = self.upscore3(d3)

#         d2 = self.final_conv_2(hd2)
#         d2 = self.upscore2(d2)

#         #Output segmentation map
#         d1 = self.final_conv_1(hd1)

#         # print(f"{d1.shape}, {d2.shape}, {d3.shape}, {d4.shape}, {d5.shape}")

#         return d1, d2, d3, d4, d5 #output
        

# class PatchMerging(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(4 * dim)
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
#         x = self.norm(x)
#         x = self.reduction(x)
#         return x

# class WindowAttention(nn.Module):
#     def __init__(
#             self,
#             dim,
#             window_size,
#     ):
#         super().__init__()
#         self.window_size = window_size
#         self.window_area = self.window_size[0]*self.window_size[1]
#         self.num_heads = 4
#         head_dim =  dim // self.num_heads
#         # attn_dim = head_dim * self.num_heads
#         self.scale = head_dim ** -0.5

#         self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) **2, self.num_heads))

#         # get pair-wise relative position index for each token inside the window
#         self.register_buffer("relative_position_index", self.get_relative_position_index(self.window_size[0], self.window_size[1]), persistent=False)

#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)

#         torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def _get_rel_pos_bias(self):
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         return relative_position_bias.unsqueeze(0)

#     def get_relative_position_index(self, win_h: int, win_w: int):
#         # get pair-wise relative position index for each token inside the window
#         coords = torch.stack(torch.meshgrid(torch.arange(win_h), torch.arange(win_w),indexing='ij'))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
#         relative_coords[:, :, 1] += win_w - 1
#         relative_coords[:, :, 0] *= 2 * win_w - 1
#         return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

#     def forward(self, x, mask = None):
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)


#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)
#         attn = attn + self._get_rel_pos_bias()
#         if mask is not None:
#             num_win = mask.shape[0]
#             attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#         attn = self.softmax(attn)
#         x = attn @ v

#         x = x.transpose(1, 2).reshape(B_, N, -1)
#         x = self.proj(x)
#         return x

# class SwinTransformerBlock1(nn.Module):
#     def __init__(
#             self,  dim, input_resolution, window_size = 7, shift_size = 0):

#         super().__init__()
#         self.input_resolution = input_resolution
#         window_size = (window_size, window_size)
#         shift_size = (shift_size, shift_size)
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.window_area = self.window_size[0] * self.window_size[1]

#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = WindowAttention(
#             dim,
#             window_size=self.window_size,
#         )

#         self.norm2 = nn.LayerNorm(dim)

#         self.mlp = nn.Sequential(
#             nn.Linear(dim, 4 * dim),
#             nn.GELU(),
#             nn.LayerNorm(4 * dim),
#             nn.Linear( 4 * dim, dim)
#         )

#         if self.shift_size:
#             # calculate attention mask for SW-MSA
#             H, W = self.input_resolution
#             H = math.ceil(H / self.window_size[0]) * self.window_size[0]
#             W = math.ceil(W / self.window_size[1]) * self.window_size[1]
#             img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
#             cnt = 0
#             for h in (
#                     slice(0, -self.window_size[0]),
#                     slice(-self.window_size[0], -self.shift_size[0]),
#                     slice(-self.shift_size[0], None)):
#                 for w in (
#                         slice(0, -self.window_size[1]),
#                         slice(-self.window_size[1], -self.shift_size[1]),
#                         slice(-self.shift_size[1], None)):
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1
#             mask_windows = self.window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#             mask_windows = mask_windows.view(-1, self.window_area)
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         else:
#             attn_mask = None

#         self.register_buffer("attn_mask", attn_mask, persistent=False)

#     def window_partition(self, x, window_size):
#         B, H, W, C = x.shape
#         x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
#         windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
#         return windows
    
#     def window_reverse(self, windows, window_size, H, W):
#         C = windows.shape[-1]
#         x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
#         x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
#         return x

#     def _attn(self, x):
#         B, H, W, C = x.shape

#         # cyclic shift
#         if self.shift_size:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
#         else:
#             shifted_x = x

#         # partition windows
#         x_windows = self.window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
#         shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
#         shifted_x = shifted_x[:, :H, :W, :].contiguous()

#         # reverse cyclic shift
#         if self.shift_size:
#             x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
#         else:
#             x = shifted_x
#         return x

#     def forward(self, x):
#         B, H, W, C = x.shape
#         B, H, W, C = x.shape
#         x = x + self._attn(self.norm1(x))
#         x = x.reshape(B, -1, C)
#         x = x + self.mlp(self.norm2(x))
#         x = x.reshape(B, H,W, C)
#         return x

# class SwinBlock(nn.Module):
#     def __init__(self, dims, ip_res, ss_size = 3):
#         super().__init__()
#         self.swtb1 = SwinTransformerBlock1(dim=dims, input_resolution=ip_res)
#         self.swtb2 = SwinTransformerBlock1(dim=dims, input_resolution=ip_res, shift_size=ss_size)

#     def forward(self, x):
#         return self.swtb2(self.swtb1(x))
    
# class Encoder(nn.Module):
#     def __init__(self, C, partioned_ip_res, num_blocks=3):
#         super().__init__()
#         H, W = partioned_ip_res[0], partioned_ip_res[1]
#         self.enc_swin_blocks = nn.ModuleList([
#             SwinBlock(C, (H, W)),
#             SwinBlock(2*C, (H//2, W//2)),
#             SwinBlock(4*C, (H//4, W//4))
#         ])
#         self.enc_patch_merge_blocks = nn.ModuleList([
#             PatchMerging(C),
#             PatchMerging(2*C),
#             PatchMerging(4*C)
#         ])

#     def forward(self, x):
#         skip_conn_ftrs = []
#         for swin_block,patch_merger in zip(self.enc_swin_blocks, self.enc_patch_merge_blocks):
#             x = swin_block(x)
#             skip_conn_ftrs.append(x)
#             x = patch_merger(x)
#         return x, skip_conn_ftrs
    
# class SwinUNet(nn.Module):
#     def __init__(self, H, W, ch, C, num_class, num_blocks=3, patch_size = 4):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(ch, C, patch_size)
#         self.encoder = Encoder(C, (H//patch_size, W//patch_size), num_blocks)
#         self.bottleneck = SwinBlock(C*(2**num_blocks), (H//(patch_size* (2**num_blocks)), W//(patch_size* (2**num_blocks))))

#     def forward(self, x):
#         x = self.patch_embed(x)

#         x, skip_connections  = self.encoder(x)

#         x = self.bottleneck(x)

#         return x