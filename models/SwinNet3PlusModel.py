import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import timm
import os

class SwinNet3PlusModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.name = f"SwinNet3+_model_{''.join(random.sample([str(x) for x in range(10)], 4))}"

        # self.filters = [64, 128, 256, 512, 1024]
        self.filters = [128, 256, 512, 1024, 2048]

        #Swin Encoder as backbone
        self.encoder = SwinEncoder(image_size = (512, 1024)) #(256, 512))

        self.bottleneck = DoubleConvolution(self.filters[3], self.filters[4])

        self.num_blocks = 5

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
        self.hd5_UT_hd4 = UpSample(self.filters[4], self.filters[4], scale_factor = 2) #nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = SingleConvolution(self.filters[4], self.filters[0])

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        # self.conv4d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks) #64 in_channels; 5 blocks
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
        self.hd4_UT_hd3 = UpSample(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks, scale_factor = 2) #nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = UpSample(self.filters[4], self.filters[4], scale_factor = 2) #nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = SingleConvolution(self.filters[4], self.filters[0])

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        # self.conv3d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)
        self.conv3d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = SingleConvolution(self.filters[0], self.filters[0])

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = SingleConvolution(self.filters[1], self.filters[0])

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = UpSample(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks, scale_factor = 2) #nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = UpSample(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks, scale_factor = 4) #nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = UpSample(self.filters[4], self.filters[4], scale_factor = 4) #nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = SingleConvolution(self.filters[4], self.filters[0])

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        # self.conv2d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)
        self.conv2d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = SingleConvolution(self.filters[0], self.filters[0])

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = UpSample(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks, scale_factor = 2) #nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = UpSample(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks, scale_factor = 4) #nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = UpSample(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks, scale_factor = 8) #nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0])

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = UpSample(self.filters[4], self.filters[4], scale_factor = 8) #nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = SingleConvolution(self.filters[4], self.filters[0])

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        # self.conv1d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)
        self.conv1d_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)

        #Final Convolution
        self.final_upsample = UpSample(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks, scale_factor = 2) #nn.Upsample(scale_factor=2, mode='bilinear')
        self.final_conv_1 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)
        self.final_conv_2 = SingleConvolution(self.filters[0] * self.num_blocks, self.filters[0] * self.num_blocks)

        self.final_conv_3 = nn.Conv2d(in_channels = self.filters[0] * self.num_blocks, out_channels = num_classes, kernel_size = 1)

    def forward(self, x):
        
        down_1, down_2, down_3, down_4 = self.encoder(x)

        bottleneck = self.bottleneck(down_4)

        # print(f"D1: {down_1.shape}")
        # print(f"D2: {down_2.shape}")
        # print(f"D3: {down_3.shape}")
        # print(f"D4: {down_4.shape}")
        # print(f"Bott: {bottleneck.shape}")

        #Decoder
        '''stage 4d'''
        #Inter-skip Connections
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(down_1))
        # h1_PT_hd4 = self.dropout(h1_PT_hd4)

        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(down_2))
        # h2_PT_hd4 = self.dropout(h2_PT_hd4)

        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(down_3))
        # h3_PT_hd4 = self.dropout(h3_PT_hd4)

        h4_Cat_hd4 = self.h4_Cat_hd4_conv(down_4)
        # h4_Cat_hd4 = self.dropout(h4_Cat_hd4)

        hd5_UT_hd4 = self.hd5_UT_hd4_conv(bottleneck)

        # print(f"Tar: {h2_Cat_hd2.shape}")
        # print(self.h1_PT_hd2(down_1).shape)
        # print(f"hd5_UT_hd1: {h1_PT_hd2.shape}")

        hd4 = self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))

        '''stage 3d'''
        #Inter-skip Connections
        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(down_1))
        # h1_PT_hd3 = self.dropout(h1_PT_hd3)

        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(down_2))
        # h2_PT_hd3 = self.dropout(h2_PT_hd3)

        h3_Cat_hd3 = self.h3_Cat_hd3_conv(down_3)
        # h3_Cat_hd3 = self.dropout(h3_Cat_hd3)

        hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))

        #Intra-skip Connection
        hd5_UT_hd3 = self.hd5_UT_hd3_conv(self.hd5_UT_hd3(bottleneck))

        hd3 = self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))

        '''stage 2d'''
        #Inter-skip Connections
        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(down_1))
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(down_2)

        hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))

        #Intra-skip Connections
        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
        hd5_UT_hd2 = self.hd5_UT_hd2_conv(self.hd5_UT_hd2(bottleneck))

        hd2 = self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))


        '''stage 1d'''
        #Inter-skip Connection
        h1_Cat_hd1 = self.h1_Cat_hd1_conv(down_1)

        hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))

        #Intra-skip Connections
        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))
        hd5_UT_hd1 = self.hd5_UT_hd1_conv(self.hd5_UT_hd1(bottleneck))

        hd1 = self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))

        # print(f"H1: {hd1.shape}")
        # print(f"H2: {hd2.shape}")
        # print(f"H3: {hd3.shape}")
        # print(f"H4: {hd4.shape}")

        output = self.final_upsample(hd1)
        output = self.final_conv_1(output)
        # print(output.shape)
        output = self.final_upsample(output)
        output = self.final_conv_2(output)
        # print(output.shape)

        output = self.final_conv_3(output)  # d1->320*320*n_classes

        # print(output.shape)

        return output    

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.conv_layers(x)

class SingleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.conv_layers(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        
        self.upsampling_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = scale_factor, stride = scale_factor)
        # self.conv_layer = SingleConvolution(in_channels, out_channels)
    
    def forward(self, x):

        return self.upsampling_layer(x)

class SwinEncoder(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.swin_model = "swinv2_base_window8_256.ms_in1k"
        self.model_encoder = timm.create_model(self.swin_model, pretrained = False, features_only = True, img_size = image_size)

    def forward(self, x):

        skip_connections = self.model_encoder(x)
        skip_1, skip_2, skip_3, skip_4 = [skip_con.permute(0, 3, 1, 2) for skip_con in skip_connections]

        return skip_1, skip_2, skip_3, skip_4

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