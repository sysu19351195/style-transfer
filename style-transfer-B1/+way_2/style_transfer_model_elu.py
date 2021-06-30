import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
    def forward(self, x):
        out_style = {}
        out_content = {}
        x = self.features[0](x)
        x = self.features[1](x)
        # out_style['r11'] = x
        x = self.features[2](x)
        x = self.features[3](x)
        out_style['r12'] = x
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        # out_style['r21'] = x
        x = self.features[7](x)
        x = self.features[8](x)
        out_style['r22'] = x
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        # out_style['r31'] = x
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        out_style['r33'] = x
        out_content['r33'] = x
        x = self.features[16](x)
        x = self.features[17](x)
        x = self.features[18](x)
        # out_style['r41'] = x
        x = self.features[19](x)
        x = self.features[20](x)
        # out_content['r42'] = x
        x = self.features[21](x)
        x = self.features[22](x)
        out_style['r43'] = x
        x = self.features[23](x)
        x = self.features[24](x)
        x = self.features[25](x)
        # out_style['r51'] = x
        x = self.features[26](x)
        x = self.features[27](x)
        x = self.features[28](x)
        x = self.features[29](x)
        x = self.features[30](x)
        return out_content, out_style

class Resnet_block(nn.Module):
    def __init__(self, input_dim, norm_layer = nn.InstanceNorm2d, use_bias = True):
        super(Resnet_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=use_bias),
            # nn.GroupNorm(8, 512),
            # norm_layer(input_dim),
            # nn.ReLU(True),
            nn.ELU(alpha=0.1, inplace=True),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=use_bias),
            # nn.GroupNorm(8, 512),
            norm_layer(input_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x

class Reset_Generator(nn.Module):
    def __init__(self,num_blocks ,norm_layer = nn.InstanceNorm2d, use_bias = True):
        super(Reset_Generator, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=use_bias),
            # nn.GroupNorm(1, 64),
            norm_layer(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            # nn.GroupNorm(2, 128),
            norm_layer(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            # nn.GroupNorm(4, 256),
            norm_layer(256),
            nn.ReLU(True),
            # nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),
            # nn.ReLU(True),
            # # nn.GroupNorm(8, 512),
            # norm_layer(512),
        )
        block = []
        for i in range(num_blocks):
            block += [Resnet_block(input_dim=256)]
        self.block = nn.Sequential(*block)
        self.conv_transpose = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=use_bias),
            # nn.ReLU(True),
            # # nn.GroupNorm(4, 256),
            # norm_layer(256),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=use_bias),
            # nn.GroupNorm(2, 128),
            norm_layer(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=use_bias),
            # nn.GroupNorm(1, 64),
            norm_layer(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = self.conv_transpose(x)
        return x





