from lib import *
from config import *

class VGG(nn.Module):
    def __init__(self, conv_arch, num_classes=10, in_channels=1):
        super(VGG, self).__init__()

        self.conv_blks = self._make_conv_layers(conv_arch, in_channels)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_arch[-1][1]*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def _vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _make_conv_layers(self, conv_arch, in_channels):
        layers = []
        for num_convs, out_channels in conv_arch:
            layers.append(self._vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_blks(x)
        x = self.fc(x)
        return x


# Residual Net
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(ResNet, self).__init__()

        # Block 1
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block 2 to Block 5
        self.b2 = nn.Sequential(*self._residual_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*self._residual_block(64, 128, 2))
        self.b4 = nn.Sequential(*self._residual_block(128, 256, 2))
        self.b5 = nn.Sequential(*self._residual_block(256, 512, 2))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def _residual_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    x = torch.rand(1, 1, 224, 224)

    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    vggnet = VGG(conv_arch)
    resnet = ResNet()
    
    output = vggnet(x)
    print(output.shape)

