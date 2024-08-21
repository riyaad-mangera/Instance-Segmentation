import torch
import torch.nn as nn

class UNetModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        #Contracting Path
        self.down_conv_1 = DownSample(in_channels = in_channels, out_channels = 64)
        self.down_conv_2 = DownSample(in_channels = 64, out_channels = 128)
        self.down_conv_3 = DownSample(in_channels = 128, out_channels = 256)
        self.down_conv_4 = DownSample(in_channels = 256, out_channels = 512)

        #Mid Point
        self.centre_conv = DoubleConvolution(512, 1024)

        #Expanding Path
        self.up_conv_1 = UpSample(in_channels = 1024, out_channels = 512)
        self.up_conv_2 = UpSample(in_channels = 512, out_channels = 256)
        self.up_conv_3 = UpSample(in_channels = 256, out_channels = 128)
        self.up_conv_4 = UpSample(in_channels = 128, out_channels = 64)

        #Final Convolution
        self.final_conv = nn.Conv2d(in_channels = 64, out_channels = num_classes, kernel_size = 1)

    def forward(self, x):
        #Contracting Path
        down_1, pool_1 = self.down_conv_1(x)
        down_2, pool_2 = self.down_conv_2(pool_1)
        down_3, pool_3 = self.down_conv_3(pool_2)
        down_4, pool_4 = self.down_conv_4(pool_3)

        #Mid point
        centre = self.centre_conv(pool_4)

        #Expanding Path
        up_1 = self.up_conv_1(centre, down_4)
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)

        #Final Convolution
        output = self.final_conv(up_4)

        return output

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
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