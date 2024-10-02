import torch
import torch.nn as nn
import random

class UNet3PlusAttnModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.name = f"unet3+Attn_model_{''.join(random.sample([str(x) for x in range(10)], 4))}"
        self.filters = [64, 128, 256, 512, 1024]
        self.num_blocks = 5

        #Encoder Layers
        self.down_conv_1 = DownSample(in_channels = in_channels, out_channels = self.filters[0])
        self.down_conv_2 = DownSample(in_channels = self.filters[0], out_channels = self.filters[1])
        self.down_conv_3 = DownSample(in_channels = self.filters[1], out_channels = self.filters[2])
        self.down_conv_4 = DownSample(in_channels = self.filters[2], out_channels = self.filters[3])

        #Bottleneck
        self.bottleneck = DoubleConvolution(self.filters[3], self.filters[4])

        #Attention Layers
        self.attn_1 = AttentionGate(self.filters[0])
        self.attn_2 = AttentionGate(self.filters[0])
        self.attn_3 = AttentionGate(self.filters[0])
        self.attn_4 = AttentionGate(self.filters[0])

        #Dropout
        self.dropout = nn.Dropout2d(0.1)

        #Decoder Layers
        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = SingleConvolution(self.filters[0], self.filters[0])

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = SingleConvolution(self.filters[1], self.filters[0])

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = SingleConvolution(self.filters[2], self.filters[0])

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = SingleConvolution(self.filters[3], self.filters[0])

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = SingleConvolution(self.filters[4], self.filters[0])

        self.conv4d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = SingleConvolution(self.filters[0], self.filters[0])

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = SingleConvolution(self.filters[1], self.filters[0])

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = SingleConvolution(self.filters[2], self.filters[0])

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = SingleConvolution(self.filters[4], self.filters[0])

        self.conv3d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = SingleConvolution(self.filters[0], self.filters[0])

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = SingleConvolution(self.filters[1], self.filters[0])

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = SingleConvolution(self.filters[4], self.filters[0])

        self.conv2d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = SingleConvolution(self.filters[0], self.filters[0])

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = SingleConvolution(self.filters[4], self.filters[0])

        self.conv1d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)

        # Final Convolution
        self.final_conv = nn.Conv2d(in_channels = self.filters[0] * self.num_blocks, out_channels = num_classes, kernel_size = 1)

        # Deep Supervision
        self.final_conv_1 = nn.Conv2d(in_channels = self.filters[0] * self.num_blocks, out_channels = num_classes, kernel_size = 1)
        self.final_conv_2 = nn.Conv2d(in_channels = self.filters[0] * self.num_blocks, out_channels = num_classes, kernel_size = 1)
        self.final_conv_3 = nn.Conv2d(in_channels = self.filters[0] * self.num_blocks, out_channels = num_classes, kernel_size = 1)
        self.final_conv_4 = nn.Conv2d(in_channels = self.filters[0] * self.num_blocks, out_channels = num_classes, kernel_size = 1)
        self.final_conv_5 = nn.Conv2d(in_channels = self.filters[4], out_channels = num_classes, kernel_size = 1)

        # Bilinear Upsampling
        self.upscore6 = nn.Upsample(scale_factor = 32, mode = 'bilinear')
        self.upscore5 = nn.Upsample(scale_factor = 16, mode = 'bilinear')
        self.upscore4 = nn.Upsample(scale_factor = 8, mode = 'bilinear')
        self.upscore3 = nn.Upsample(scale_factor = 4, mode = 'bilinear')
        self.upscore2 = nn.Upsample(scale_factor = 2, mode = 'bilinear')

    def forward(self, x):
        # Encoder
        down_1, pool_1 = self.down_conv_1(x)
        down_2, pool_2 = self.down_conv_2(pool_1)
        down_3, pool_3 = self.down_conv_3(pool_2)
        down_3 = self.dropout(down_3)
        pool_3 = self.dropout(pool_3)

        down_4, pool_4 = self.down_conv_4(pool_3)
        down_4 = self.dropout(down_4)
        pool_4 = self.dropout(pool_4)

        # Bottleneck
        bottleneck = self.bottleneck(pool_4)
        bottleneck = self.dropout(bottleneck)

        # Decoder
        '''stage 4d'''
        #Inter-skip Connections
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(down_1))
        h1_PT_hd4 = self.dropout(h1_PT_hd4)

        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(down_2))
        h2_PT_hd4 = self.dropout(h2_PT_hd4)

        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(down_3))
        h3_PT_hd4 = self.dropout(h3_PT_hd4)

        h4_Cat_hd4 = self.h4_Cat_hd4_conv(down_4)
        h4_Cat_hd4 = self.dropout(h4_Cat_hd4)

        hd5_UT_hd4 = self.hd5_UT_hd4_conv(self.hd5_UT_hd4(bottleneck))

        # Attention Gate
        h1_PT_hd4 = self.attn_4(hd5_UT_hd4, h1_PT_hd4)
        h2_PT_hd4 = self.attn_4(hd5_UT_hd4, h2_PT_hd4)
        h3_PT_hd4 = self.attn_4(hd5_UT_hd4, h3_PT_hd4)
        h4_Cat_hd4 = self.attn_4(hd5_UT_hd4, h4_Cat_hd4)

        hd4 = self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))

        '''stage 3d'''
        # Inter-skip Connections
        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(down_1))
        h1_PT_hd3 = self.dropout(h1_PT_hd3)

        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(down_2))
        h2_PT_hd3 = self.dropout(h2_PT_hd3)

        h3_Cat_hd3 = self.h3_Cat_hd3_conv(down_3)
        h3_Cat_hd3 = self.dropout(h3_Cat_hd3)

        hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))

        # Intra-skip Connection
        hd5_UT_hd3 = self.hd5_UT_hd3_conv(self.hd5_UT_hd3(bottleneck))

        # Attention Gate
        h1_PT_hd3 = self.attn_3(hd4_UT_hd3, h1_PT_hd3)
        h2_PT_hd3 = self.attn_3(hd4_UT_hd3, h2_PT_hd3)
        h3_Cat_hd3 = self.attn_3(hd4_UT_hd3, h3_Cat_hd3)

        hd5_UT_hd3 = self.attn_3(hd4_UT_hd3, hd5_UT_hd3)

        hd3 = self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd5_UT_hd3, hd4_UT_hd3), 1))

        '''stage 2d'''
        # Inter-skip Connections
        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(down_1))
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(down_2)

        hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))

        # Intra-skip Connections
        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
        hd5_UT_hd2 = self.hd5_UT_hd2_conv(self.hd5_UT_hd2(bottleneck))

        # Attention Gate
        h1_PT_hd2 = self.attn_2(hd3_UT_hd2, h1_PT_hd2)
        h2_Cat_hd2 = self.attn_2(hd3_UT_hd2, h2_Cat_hd2)

        hd4_UT_hd2 = self.attn_2(hd3_UT_hd2, hd4_UT_hd2)
        hd5_UT_hd2 = self.attn_2(hd3_UT_hd2, hd5_UT_hd2)

        hd2 = self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd4_UT_hd2, hd5_UT_hd2, hd3_UT_hd2), 1))

        '''stage 1d'''
        # Inter-skip Connection
        h1_Cat_hd1 = self.h1_Cat_hd1_conv(down_1)

        hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))

        # Intra-skip Connections
        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))
        hd5_UT_hd1 = self.hd5_UT_hd1_conv(self.hd5_UT_hd1(bottleneck))

        # Attention Gate
        h1_Cat_hd1 = self.attn_1(hd2_UT_hd1, h1_Cat_hd1)

        hd3_UT_hd1 = self.attn_1(hd2_UT_hd1, hd3_UT_hd1)
        hd4_UT_hd1 = self.attn_1(hd2_UT_hd1, hd4_UT_hd1)
        hd5_UT_hd1 = self.attn_1(hd2_UT_hd1, hd5_UT_hd1)

        hd1 = self.conv1d_1(torch.cat((h1_Cat_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1, hd2_UT_hd1), 1))

        d5 = self.final_conv_5(bottleneck)
        d5 = self.upscore5(d5)

        d4 = self.final_conv_4(hd4)
        d4 = self.upscore4(d4)

        d3 = self.final_conv_3(hd3)
        d3 = self.upscore3(d3)

        d2 = self.final_conv_2(hd2)
        d2 = self.upscore2(d2)

        # Output segmentation map
        d1 = self.final_conv_1(hd1)

        return d1, d2, d3, d4, d5 #output

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