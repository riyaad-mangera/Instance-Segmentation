import torch
import torch.nn as nn
import random
import timm

"""The Decoder Layers of this model were based off of the 
    Unet3+ model developed by ZJUGiveLab (2020). As such,
    the structure of the Decoder, as well as the names of some
    variable components, are left as is for clarity.
"""
class SwinNet3PlusModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.name = f"SwinNet3+_model_{''.join(random.sample([str(x) for x in range(10)], 4))}"

        self.filters = [128, 256, 512, 1024, 2048, 128] # Last value determines size of Decoder filters

        # Swin Encoder as backbone
        self.encoder = SwinEncoder(image_size = (512, 1024)) #(256, 512) Use if encountering memory errors

        self.bottleneck = DoubleConvolution(self.filters[3], self.filters[4])

        self.num_blocks = 5

        # Decoder Layers
        '''stage 4d'''
        # Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = SingleConvolution(self.filters[0], self.filters[5])

        # Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = SingleConvolution(self.filters[1], self.filters[5])

        # Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = SingleConvolution(self.filters[2], self.filters[5])

        self.h4_Cat_hd4_conv = SingleConvolution(self.filters[3], self.filters[5])

        # Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd5_UT_hd4_conv = SingleConvolution(self.filters[4], self.filters[5])

        self.conv4d_1 = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5] * self.num_blocks)

        '''stage 3d'''
        # Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = SingleConvolution(self.filters[0], self.filters[5])

        # Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = SingleConvolution(self.filters[1], self.filters[5])

        self.h3_Cat_hd3_conv = SingleConvolution(self.filters[2], self.filters[5])

        # Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd4_UT_hd3_conv = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5])

        # Upsample 2 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd5_UT_hd3_conv = SingleConvolution(self.filters[4], self.filters[5])

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5] * self.num_blocks)

        '''stage 2d '''
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = SingleConvolution(self.filters[0], self.filters[5])

        self.h2_Cat_hd2_conv = SingleConvolution(self.filters[1], self.filters[5])

        # Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5])

        # Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd4_UT_hd2_conv = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5])

        # Upsample 4 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd5_UT_hd2_conv = SingleConvolution(self.filters[4], self.filters[5])

        self.conv2d_1 = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5] * self.num_blocks)

        '''stage 1d'''
        self.h1_Cat_hd1_conv = SingleConvolution(self.filters[0], self.filters[5])

        # Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd2_UT_hd1_conv = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5])

        # Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd3_UT_hd1_conv = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5])

        # Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd4_UT_hd1_conv = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5])

        # Upsample 8 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd1_conv = SingleConvolution(self.filters[4], self.filters[5])

        self.conv1d_1 = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5] * self.num_blocks)

        #Final Convolution
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.final_conv_1 = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5] * self.num_blocks)
        self.final_conv_2 = SingleConvolution(self.filters[5] * self.num_blocks, self.filters[5] * self.num_blocks)

        self.final_conv_3 = nn.Conv2d(in_channels = self.filters[5] * self.num_blocks, out_channels = num_classes, kernel_size = 1)

    def forward(self, x):
        
        down_1, down_2, down_3, down_4 = self.encoder(x)

        bottleneck = self.bottleneck(down_4)

        #Decoder
        '''stage 4d'''
        #Inter-skip Connections
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(down_1))

        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(down_2))
        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(down_3))

        h4_Cat_hd4 = self.h4_Cat_hd4_conv(down_4)

        hd5_UT_hd4 = self.hd5_UT_hd4_conv(bottleneck)

        hd4 = self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))

        '''stage 3d'''
        #Inter-skip Connections
        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(down_1))

        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(down_2))

        h3_Cat_hd3 = self.h3_Cat_hd3_conv(down_3)

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

        output = self.final_upsample(hd1)
        output = self.final_conv_1(output)
        
        output = self.final_upsample(output)
        output = self.final_conv_2(output)

        output = self.final_conv_3(output)

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

# Retrieve a pre-trained SwinV2 backbone and extract only the outputs of each of its 4 blocks
class SwinEncoder(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.swin_model = "swinv2_base_window8_256.ms_in1k"
        self.model_encoder = timm.create_model(self.swin_model, pretrained = True, features_only = True, img_size = image_size, window_size = 8)

    def forward(self, x):

        skip_connections = self.model_encoder(x)
        skip_1, skip_2, skip_3, skip_4 = [skip_con.permute(0, 3, 1, 2) for skip_con in skip_connections]

        return skip_1, skip_2, skip_3, skip_4