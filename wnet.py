import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate=6, kernel_size=3, stride=1):
        super(MBConv, self).__init__()
        hidden_channels = in_channels * expansion_rate
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, kernel_size // 2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        if self.use_residual:
            return x + self.se(self.conv(x))
        return self.se(self.conv(x))

class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, 5, padding=4, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.relu(self.bn(x1 + x2 + x3))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)
        self.mbconv1 = MBConv(32, 64, expansion_rate=1, kernel_size=3, stride=2)
        self.mbconv2 = MBConv(64, 128, expansion_rate=6, kernel_size=3)
        self.mbconv3 = MBConv(128, 256, expansion_rate=6, kernel_size=5)
        self.mbconv4 = MBConv(256, 512, expansion_rate=6, kernel_size=3)
        self.mbconv5 = MBConv(512, 512, expansion_rate=6, kernel_size=5)
        self.mbconv6 = MBConv(512, 1024, expansion_rate=6, kernel_size=3)
        self.conv2 = nn.Conv2d(1024, 1024, 1, bias=False)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.multiscale1 = MultiScaleBlock(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.multiscale2 = MultiScaleBlock(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 5, padding=2, bias=False)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)

    def forward(self, x, skip1, skip2):
        x = self.upsample1(x)
        x = x + skip1
        x = self.multiscale1(x)
        x = self.conv1(x)

        x = self.upsample2(x)
        x = x + skip2
        x = self.multiscale2(x)
        x = self.conv2(x)

        x = self.conv3(x)
        return x

class WNet(nn.Module):
    def __init__(self, num_grades=8):
        super(WNet, self).__init__()
        self.encoder = Encoder()

        self.decoder1 = Decoder(1024, 2)  # Foreground map
        self.decoder2 = Decoder(1024, 2 * num_grades)  # Ordinal grade map

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder branches
        foreground_map = self.decoder1(x, skip1=x, skip2=x)
        ordinal_map = self.decoder2(x, skip1=x, skip2=x)

        return foreground_map, ordinal_map