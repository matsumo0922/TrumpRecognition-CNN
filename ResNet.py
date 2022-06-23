import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, activation, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x))) if self.relu is not None else self.norm(self.conv(x))


class DiffBlock(nn.Module):
    def __init__(self, conv_input_ch, conv_output_ch, identity_conv=None, stride=1):
        super(DiffBlock, self).__init__()

        self.conv1 = Conv(conv_input_ch, conv_output_ch, nn.ReLU(), kernel_size=1, stride=1)
        self.conv2 = Conv(conv_output_ch, conv_output_ch, nn.ReLU(), kernel_size=3, stride=stride, padding=1)
        self.conv3 = Conv(conv_output_ch, conv_output_ch * 4, None, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.identity_conv = identity_conv

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.identity_conv is not None:
            identity = self.identity_conv(identity)

        x += identity

        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, diff_block, n_classes):
        super(ResNet, self).__init__()

        self.input_channels = 64

        self.conv1 = Conv(3, 64, nn.ReLU(), kernel_size=7, stride=2, padding=3)
        self.conv2_x = self.get_layer(diff_block, n_blocks=3, first_conv_output_ch=64, stride=1)
        self.conv3_x = self.get_layer(diff_block, n_blocks=4, first_conv_output_ch=128, stride=2)
        self.conv4_x = self.get_layer(diff_block, n_blocks=6, first_conv_output_ch=256, stride=2)
        self.conv5_x = self.get_layer(diff_block, n_blocks=3, first_conv_output_ch=512, stride=2)

        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(512 * 4, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pooling(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avg_pooling(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x

    def get_layer(self, diff_block, n_blocks, first_conv_output_ch, stride):
        layers = list()

        identity_conv = nn.Conv2d(self.input_channels, first_conv_output_ch * 4, kernel_size=1, stride=stride)
        layers.append(diff_block(self.input_channels, first_conv_output_ch, identity_conv=identity_conv, stride=stride))

        self.input_channels = first_conv_output_ch * 4

        for i in range(n_blocks - 1):
            layers.append(diff_block(self.input_channels, first_conv_output_ch))

        return nn.Sequential(*layers)
