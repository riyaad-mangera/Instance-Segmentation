import torch
import torch.nn as nn
import random

class UNet3PlusAttnModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.name = f"unet3+Attn_model_{''.join(random.sample([str(x) for x in range(10)], 4))}"

        #Down Sampling
        self.down_conv_1 = DownSample(in_channels = in_channels, out_channels = 64)
        self.down_conv_2 = DownSample(in_channels = 64, out_channels = 128)
        self.down_conv_3 = DownSample(in_channels = 128, out_channels = 256)
        self.down_conv_4 = DownSample(in_channels = 256, out_channels = 512)

        #Attention Layers
        # self.attn_1 = SelfAttention(64)
        # self.attn_2 = SelfAttention(128)
        # self.attn_3 = SelfAttention(256)
        # self.attn_4 = SelfAttention(512)

        #Bottleneck
        self.bottleneck = DoubleConvolution(512, 1024)

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = SingleConvolution(64, 64)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = SingleConvolution(128, 64)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = SingleConvolution(256, 64)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = SingleConvolution(512, 64)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = SingleConvolution(1024, 64)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = SingleConvolution(64 * 5, 64 * 5) #64 in_channels; 5 blocks

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = SingleConvolution(64, 64)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = SingleConvolution(128, 64)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = SingleConvolution(256, 64)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = SingleConvolution(64 * 5, 64)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14

        self.hd5_UT_hd3_conv = SingleConvolution(1024, 64)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = SingleConvolution(64 * 5, 64 * 5)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = SingleConvolution(64, 64)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = SingleConvolution(128, 64)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = SingleConvolution(64 * 5, 64)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = SingleConvolution(64 * 5, 64)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = SingleConvolution(1024, 64)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = SingleConvolution(64 * 5, 64 * 5)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = SingleConvolution(64, 64)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = SingleConvolution(64 * 5, 64)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = SingleConvolution(64 * 5, 64)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = SingleConvolution(64 * 5, 64)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = SingleConvolution(1024, 64)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = SingleConvolution(64 * 5, 64 * 5)

        #Final Convolution
        self.final_conv = nn.Conv2d(in_channels = 64 * 5, out_channels = num_classes, kernel_size = 3)

    def forward(self, x):
        #Encoder
        down_1, pool_1 = self.down_conv_1(x)
        down_2, pool_2 = self.down_conv_2(pool_1)
        down_3, pool_3 = self.down_conv_3(pool_2)
        down_4, pool_4 = self.down_conv_4(pool_3)

        bottleneck = self.bottleneck(pool_4)

        #Decoder
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(down_1))
        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(down_2))
        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(down_3))
        h4_Cat_hd4 = self.h4_Cat_hd4_conv(down_4)
        hd5_UT_hd4 = self.hd5_UT_hd4_conv(self.hd5_UT_hd4(bottleneck))
        hd4 = self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(down_1))
        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(down_2))
        h3_Cat_hd3 = self.h3_Cat_hd3_conv(down_3)
        hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))
        hd5_UT_hd3 = self.hd5_UT_hd3_conv(self.hd5_UT_hd3(bottleneck))
        hd3 = self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(down_1))
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(down_2)
        hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))
        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
        hd5_UT_hd2 = self.hd5_UT_hd2_conv(self.hd5_UT_hd2(bottleneck))
        hd2 = self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_conv(down_1)
        hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))
        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))
        hd5_UT_hd1 = self.hd5_UT_hd1_conv(self.hd5_UT_hd1(bottleneck))
        hd1 = self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)) # hd1->320*320*UpChannels

        output = self.final_conv(hd1)  # d1->320*320*n_classes

        return torch.nn.functional.sigmoid(output)

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
        self.pooling_layer = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        down_sampling = self.conv_layer(x)
        pooling = self.pooling_layer(down_sampling)

        return down_sampling, pooling

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsampling_layer = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size = 2, stride = 2)
        self.conv_layer = DoubleConvolution(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.upsampling_layer(x1)
        x = torch.cat([x1, x2], 1)

        return self.conv_layer(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.query = nn.Conv2d(in_channels = in_channels, out_channels = in_channels/8, kernel_size = 1)
        self.key = nn.Conv2d(in_channels = in_channels, out_channels = in_channels/8, kernel_size = 1)
        self.value = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        
        B, C, W, H = x.shape

        q = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        k = self.key(x).view(B, -1, W * H)

        attention = torch.softmax(torch.bmm(q, k), dim = -1)

        v = self.value(x).view(B, -1, W * H)

        output = torch.bmm(v, attention.permute(0, 2, 1))
        output = output.view(B, C, W, H)

        output = self.gamma * output + x

        return output, attention