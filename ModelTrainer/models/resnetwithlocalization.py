import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResNetlWithLocalization(nn.Module):
    def __init__(self, input_shape, num_classes, weight_decay=1e-4):
        super(ResNetlWithLocalization, self).__init__()
        self.weight_decay = weight_decay
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_bounding_box_outputs = 4  # origin-x, origin-y, width, height

        # Initial convolution layer
        self.conv1 = self._add_convolution(3, 16, kernel_size=3)

        # Residual blocks
        self.block1 = self._add_res_net_block(16, 16, kernel_size=3, shortcut_is_conv=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.block2a = self._add_res_net_block(16, 32, kernel_size=3, shortcut_is_conv=True)
        self.block2b = self._add_res_net_block(32, 32, kernel_size=3, shortcut_is_conv=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.block3a = self._add_res_net_block(32, 64, kernel_size=3, shortcut_is_conv=True)
        self.block3b = self._add_res_net_block(64, 64, kernel_size=3, shortcut_is_conv=False)
        self.block3c = self._add_res_net_block(64, 64, kernel_size=3, shortcut_is_conv=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.block4a = self._add_res_net_block(64, 128, kernel_size=3, shortcut_is_conv=True)
        self.block4b = self._add_res_net_block(128, 128, kernel_size=3, shortcut_is_conv=False)
        self.block4c = self._add_res_net_block(128, 128, kernel_size=3, shortcut_is_conv=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.block5a = self._add_res_net_block(128, 256, kernel_size=3, shortcut_is_conv=True)
        self.block5b = self._add_res_net_block(256, 256, kernel_size=3, shortcut_is_conv=False)
        self.block5c = self._add_res_net_block(256, 256, kernel_size=3, shortcut_is_conv=False)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers for classification and regression
        self.flatten = nn.Flatten()
        self.classification_head = nn.Linear(256, num_classes)
        self.regression_head = nn.Linear(256, self.num_bounding_box_outputs)

    def _add_convolution(self, in_channels, out_channels, kernel_size):
        """ Add a convolutional layer with BatchNorm and ReLU """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _add_res_net_block(self, in_channels, out_channels, kernel_size, shortcut_is_conv):
        """ Add a residual block """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        shortcut = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False) if shortcut_is_conv else nn.Identity()
        return nn.Sequential(*layers), shortcut

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1[0](x) + self.block1[1](x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.block2a[0](x) + self.block2a[1](x)
        x = F.relu(x)
        x = self.block2b[0](x) + self.block2b[1](x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.block3a[0](x) + self.block3a[1](x)
        x = F.relu(x)
        x = self.block3b[0](x) + self.block3b[1](x)
        x = F.relu(x)
        x = self.block3c[0](x) + self.block3c[1](x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.block4a[0](x) + self.block4a[1](x)
        x = F.relu(x)
        x = self.block4b[0](x) + self.block4b[1](x)
        x = F.relu(x)
        x = self.block4c[0](x) + self.block4c[1](x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.block5a[0](x) + self.block5a[1](x)
        x = F.relu(x)
        x = self.block5b[0](x) + self.block5b[1](x)
        x = F.relu(x)
        x = self.block5c[0](x) + self.block5c[1](x)
        x = F.relu(x)
        
        x = self.pool5(x)

        x = self.flatten(x)
        classification_output = self.classification_head(x)
        bounding_box_output = self.regression_head(x)
        return classification_output, bounding_box_output


if __name__ == "__main__":
    # Example usage
    model = ResNetlWithLocalization(input_shape=(3, 112, 112), num_classes=32)
    summary(model, input_size=(3, 112, 112))
