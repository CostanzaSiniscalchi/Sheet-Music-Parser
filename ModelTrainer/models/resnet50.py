import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

class ResNet1Configuration(nn.Module):
    """ResNet-1 Configuration in PyTorch"""

    def __init__(self, optimizer, width: int, height: int, training_minibatch_size,number_of_classes: int, weight_decay: float = 1e-4):
        super(ResNet1Configuration, self).__init__()
        self.weight_decay = weight_decay
        self.input_shape = (3, height, width)
        self.name = 'resnet'
        self.optimizer = optimizer
        self.training_minibatch_size = training_minibatch_size

        # Initial Convolution and Pooling
        self.initial_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Blocks
        self.layer1 = self._make_resnet_layer(64, 3, stride=1, layer_number=2)
        self.layer2 = self._make_resnet_layer(128, 3, stride=2, layer_number=3)
        self.layer3 = self._make_resnet_layer(256, 3, stride=2, layer_number=4)
        self.layer4 = self._make_resnet_layer(512, 3, stride=2, layer_number=5)

        # Global Average Pooling and Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, number_of_classes)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten
        x = self.fc(x)
        return F.softmax(x, dim=1)

    def _make_resnet_layer(self, filters, blocks, stride, layer_number):
        """Create a ResNet layer consisting of multiple blocks."""
        layers = []
        # First block with stride
        layers.append(ResNetBlock(self.input_shape[0], filters, stride, is_first=True, weight_decay=self.weight_decay))
        self.input_shape = (filters, self.input_shape[1] // stride, self.input_shape[2] // stride)
        # Remaining blocks with stride 1
        for _ in range(1, blocks):
            layers.append(ResNetBlock(filters, filters, stride=1, is_first=False, weight_decay=self.weight_decay))
        return nn.Sequential(*layers)


class ResNetBlock(nn.Module):
    """Residual Block for ResNet"""

    def __init__(self, in_channels, out_channels, stride=1, is_first=False, weight_decay=1e-4):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if is_first or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return F.relu(x)


if __name__ == "__main__":
    classifier = ResNet1Configuration(width=96, height=96, number_of_classes=32)
    classifier = classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # print(summary(classifier))