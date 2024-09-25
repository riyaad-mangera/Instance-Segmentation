import torch
import torch.nn as nn
import random

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

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv_layers(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layer = DoubleConvolution(in_channels, out_channels)
        self.pooling_layer = nn.MaxPool2d(kernel_size = 2) #, stride = 2)

    def forward(self, x):
        down_sampling = self.conv_layer(x)
        pooling = self.pooling_layer(down_sampling)

        return down_sampling, pooling

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsampling_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2)
        self.conv_layer = DoubleConvolution(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.upsampling_layer(x1)
        x = torch.cat([x1, x2], 1)

        return self.conv_layer(x)

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 1, padding = 0, stride = 1),
            nn.BatchNorm2d(in_channels)
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 1, padding = 0, stride = 1),
            nn.BatchNorm2d(in_channels)
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels, 1, kernel_size = 1, padding = 0, stride = 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()            
        )

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)

        psi = self.psi(g1 + x1)

        attn = x * psi

        return attn
    
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

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_shape, patch_size, embed_dim):
        super().__init__()

        c, h, w = img_shape
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        assert h % patch_size[0] == 0 and w % patch_size[1] == 0,\
            'image dimensions must be divisible by the patch size'

        self.patch_size = patch_size
        self.num_patches = (h // patch_size[0]) * (w // patch_size[1])
        patch_dim = c * patch_size[0] * patch_size[1]
        assert self.num_patches > 16,\
            f'your number of patches ({self.num_patches}) is too small for ' \
            f'attention to be effective. try decreasing your patch size'

        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        p1, p2 = self.patch_size

        b, c, h, w = x.shape

        x = torch.reshape(x, (b, ((h//p1) * (w//p2)), (c * p1 * p2)))

        # x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2)

        print(x.shape)

        x = self.projection(x)
        return x

class PatchifyV2(nn.Module):
    def __init__(self, patch_size=56):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x -> B c h w
        bs, c, h, w = x.shape
        
        x = self.unfold(x)
        # x -> B (c*p*p) L
        
        # Reshaping into the shape we want
        a = x.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
        # a -> ( B no.of patches c p p )
        return a

class PatchEmbedding(nn.Module):
    """Create 1D Sequence lernable embedding vector from a 2D input image

    Args:
        in_channels (int): Nunber of Color Channels. Defaults to 3
        patch_size (int): Target size for each patch. Defaults to 8
        embedding_dim (int): Size of image embedding. Defaults to 768 (ViT base) 
    """

    def __init__(self,
                 in_channels:int = 3,
                 patch_size:int = 8,
                 embedding_dim:int = 768
                 ):
        
        super().__init__()
        
        self.patch_size = patch_size 

        # Layer to create patch embeddings
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        # Layer to flatten the flatten the feature map dim. to a single vector
        self.flatten = nn.Flatten(
            start_dim=2, end_dim=3
        )
    
    def forward(self, x):
        image_size = x.shape[-1]
        assert image_size % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_size}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1)

class UNetR3PlusAttnModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_blocks = 5 - 1

        self.name = f"unet3R+Attn_model_{''.join(random.sample([str(x) for x in range(10)], 4))}"
        self.filters = [64, 128, 256, 512, 1024]

        self.num_layers = 12
        self.hidden_dim = 66
        self.mlp_dim = 128 #3072
        self.num_heads = 6
        self.dropout_rate = 0.1
        self.num_patches = 1024 #H * W / patch_size * patch_size
        self.patch_size = 16

        """ Patch + Position Embeddings """
        self.patch_embed_1 = nn.Linear(
            self.patch_size * self.patch_size * in_channels, self.hidden_dim
        )

        # self.patch_embed = PatchEmbed(img_shape = (3, 512, 512), patch_size = 16, embed_dim = self.hidden_dim)
        self.patch_embed = PatchEmbedding(in_channels = 3, patch_size = 16, embedding_dim = self.hidden_dim)

        self.positions = torch.arange(start = 0, end = self.num_patches, step = 1, dtype=torch.int32)
        self.pos_embed = nn.Embedding(self.num_patches, self.hidden_dim)

        """ Transformer Encoder """
        self.trans_encoder_layers = []

        for i in range(self.num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model = self.hidden_dim,
                nhead = self.num_heads,
                dim_feedforward = self.mlp_dim,
                dropout = self.dropout_rate,
                activation = nn.GELU(),
                batch_first = True
            ).to(torch.device('cuda'))
            self.trans_encoder_layers.append(layer)

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

        #Bilinear Upsampling
        self.upscore6 = nn.Upsample(scale_factor = 32, mode = 'bilinear')###
        self.upscore5 = nn.Upsample(scale_factor = 16, mode = 'bilinear')
        self.upscore4 = nn.Upsample(scale_factor = 8, mode = 'bilinear')
        self.upscore3 = nn.Upsample(scale_factor = 4, mode = 'bilinear')
        self.upscore2 = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        # assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

    def forward(self, inputs):
        """ Patch + Position Embeddings """

        patch_embed = self.patch_embed(inputs) #self.patchify(inputs, n_patches = 1024)

        # patch_embed = self.patch_embed(inputs)   ## [8, 256, 768]

        print(patch_embed.shape)

        positions = self.positions.to(torch.device('cuda'))
        pos_embed = self.pos_embed(positions)   ## [256, 768]

        # print(pos_embed.shape)

        x = patch_embed + pos_embed ## [8, 256, 768]

        # print(x.shape)

        """ Transformer Encoder """
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        x = x.to(torch.device('cuda'))

        for i in range(self.num_layers):
            layer = self.trans_encoder_layers[i]
            x = layer(x)

            if (i+1) in skip_connection_index:
                skip_connections.append(x)

        """ CNN Decoder """
        z3, z6, z9, z12 = skip_connections

        # print(z6.shape)

        ## Reshaping
        batch = inputs.shape[0]
        z0 = inputs.view((batch, self.in_channels, 512, 512))

        shape = (batch, self.hidden_dim*4, self.patch_size, self.patch_size)
        z3 = z3.view(shape)
        z6 = z6.view(shape)
        z9 = z9.view(shape)
        z12 = z12.view(shape)


        ## Decoder 1
        x1_z12 = self.decon_1(z12)

        s1_z9 = self.skip_1_z9(z9)
        # s1_z6 = self.skip_1_z6(z6)
        # s1_z3 = self.skip_1_z3(z3)

        # print(f"{x1_z12.shape}, {s1_z9.shape}, {s1_z6.shape}, {s1_z3.shape}")

        hd4 = self.conv_1(torch.cat((x1_z12, s1_z9), 1)) # s1_z6, s1_z3), 1))

        ## Decoder 2
        x2_hd4 = self.decon_2(hd4)

        s2_z6 = self.skip_2_z6(z6)
        # s2_z3 = self.skip_2_z3(z3)
        
        hd3 = self.conv_2(torch.cat((x2_hd4, s2_z6), 1)) #s2_z3), 1))

        ## Decoder 3
        x3_hd3 = self.decon_3(hd3)

        s3_z3 = self.skip_3_z3(z3)
        
        hd2 = self.conv_3(torch.cat((x3_hd3, s3_z3), 1))

        ## Decoder 4
        x4_hd2 = self.decon_4(hd2)
        
        s4_z0 = self.skip_4_z0(z0)

        # print(f"{x4_hd2.shape}, {s4_z0.shape}")
        
        hd1 = self.conv_4(torch.cat((x4_hd2, s4_z0), 1))

        """ Output """
        # output = self.output(hd1)

        d5 = self.final_conv_5(z12)
        d5 = self.upscore5(d5)

        d4 = self.final_conv_4(hd4)
        d4 = self.upscore4(d4)

        d3 = self.final_conv_3(hd3)
        d3 = self.upscore3(d3)

        d2 = self.final_conv_2(hd2)
        d2 = self.upscore2(d2)

        #Output segmentation map
        d1 = self.final_conv_1(hd1)

        # print(f"{d1.shape}, {d2.shape}, {d3.shape}, {d4.shape}, {d5.shape}")

        return d1, d2, d3, d4, d5 #output
    
