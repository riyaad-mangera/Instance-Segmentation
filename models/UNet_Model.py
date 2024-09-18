import torch
import torch.nn as nn
import random

class UNetModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.name = f"unet_model_{''.join(random.sample([str(x) for x in range(10)], 4))}"

        self.filters = [64, 128, 256, 512, 1024]
        # self.filters = [128, 256, 512, 1024, 2048]
        # self.filters = [256, 512, 1024, 2048, 2046]

        #Contracting Path
        self.down_conv_1 = DownSample(in_channels = in_channels, out_channels = self.filters[0])
        self.down_conv_2 = DownSample(in_channels = self.filters[0], out_channels = self.filters[1])
        self.down_conv_3 = DownSample(in_channels = self.filters[1], out_channels = self.filters[2])
        self.down_conv_4 = DownSample(in_channels = self.filters[2], out_channels = self.filters[3])

        #Mid Point
        self.bottleneck = DoubleConvolution(self.filters[3], self.filters[4])

        #Dropout
        self.dropout = nn.Dropout2d(0.5)

        #Expanding Path
        self.up_conv_1 = UpSample(in_channels = self.filters[4], out_channels = self.filters[3])
        self.up_conv_2 = UpSample(in_channels = self.filters[3], out_channels = self.filters[2])
        self.up_conv_3 = UpSample(in_channels = self.filters[2], out_channels = self.filters[1])
        self.up_conv_4 = UpSample(in_channels = self.filters[1], out_channels = self.filters[0])

        #Final Convolution
        self.final_conv = nn.Conv2d(in_channels = self.filters[0], out_channels = num_classes, kernel_size = 1)

    def forward(self, x):
        #Encoder
        down_1, pool_1 = self.down_conv_1(x)
        down_2, pool_2 = self.down_conv_2(pool_1)
        down_3, pool_3 = self.down_conv_3(pool_2)
        down_3 = self.dropout(down_3)
        pool_3 = self.dropout(pool_3)

        down_4, pool_4 = self.down_conv_4(pool_3)
        down_4 = self.dropout(down_4)
        pool_4 = self.dropout(pool_4)

        #Bottleneck
        bottleneck = self.bottleneck(pool_4)
        bottleneck = self.dropout(bottleneck)

        # print(bottleneck.shape)
        # print(down_4.shape)

        #Decoder
        up_1 = self.up_conv_1(bottleneck, down_4)
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)

        #Final Convolution
        output = self.final_conv(up_4)

        # print(output.shape)

        return output

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
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
        
        self.upsampling_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
        self.conv_layer = DoubleConvolution(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.upsampling_layer(x1)
        x = torch.cat([x1, x2], 1)

        return self.conv_layer(x)