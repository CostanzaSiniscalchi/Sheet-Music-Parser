import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta
from models.TrainingConfiguration import TrainingConfiguration


class ResNet1Configuration(TrainingConfiguration):
    """A network with residual modules implemented in PyTorch."""

    def __init__(self, optimizer: str, width: int, height: int, training_minibatch_size: int, number_of_classes: int):
        super().__init__(optimizer=optimizer, data_shape=(height, width, 3),
                         training_minibatch_size=training_minibatch_size, number_of_classes=number_of_classes)
        self.weight_decay = 1e-4 

    def classifier(self) -> nn.Sequential:
        """Returns the PyTorch implementation of the ResNet model."""
        layers = nn.Sequential()

        # Initial Convolution Block using `add_convolution_block_with_batch_normalization`
        self.add_convolution_block_with_batch_normalization(layers, 3, 64, 7, stride=2, layer_number=1)
        layers.add_module("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Residual Blocks
        in_channels = 64
        for layer_number, num_blocks in enumerate([3, 4, 6, 3], start=2):
            for block_number in range(1, num_blocks + 1):
                is_first_convolution = block_number == 1
                out_channels = in_channels * (2 if is_first_convolution and layer_number > 2 else 1)
                layers.add_module(
                    f"res_block_{layer_number}_{block_number}",
                    self.add_res_net_block(in_channels, out_channels, 3, is_first_convolution)
                )
                in_channels = out_channels

        # Global Average Pooling
        layers.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))

        # Fully Connected Layer
        layers.add_module("flatten", nn.Flatten())
        layers.add_module("fc", nn.Linear(in_channels, self.number_of_classes))

        return layers

    def add_convolution_block_with_batch_normalization(self, model, input_size, output_size, kernel_size,
                                                       stride, layer_number: int) -> nn.Sequential:
        model.add_module(f"Conv2d_{layer_number}", nn.Conv2d(input_size, output_size, kernel_size, stride, 1, bias=False))
        model.add_module(f"BatchNorm_{layer_number}", nn.BatchNorm2d(output_size))
        model.add_module(f"ReLU_{layer_number}", nn.ReLU(inplace=True))


    def add_res_net_block(self, in_channels, out_channels, kernel_size, is_first_convolution):
        stride = 2 if is_first_convolution else 1


        # Residual Path
        residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Shortcut Path
        shortcut = nn.Sequential()
        if is_first_convolution or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # Combine Residual and Shortcut Paths
        class ResidualBlock(nn.Module):
            def __init__(self, residual, shortcut):
                super(ResidualBlock, self).__init__()
                self.residual = residual
                self.shortcut = shortcut

            def forward(self, x):
                return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))

        return ResidualBlock(residual, shortcut)
        

    def get_optimizer(self, parameters):
        """Returns the optimizer based on the configuration."""
        if self.optimizer == "SGD":
            return SGD(parameters, lr=self.learning_rate, momentum=self.nesterov_momentum, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            return Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "Adadelta":
            return Adadelta(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def name(self) -> str:
        """Returns the name of this configuration."""
        return "resnet"

    def performs_localization(self) -> bool:
        """Indicates if this configuration performs localization."""
        return False

if __name__ == "__main__":
    classifier = ResNet1Configuration(width=96, height=96, number_of_classes=32)
    classifier = classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Print model summary
    print(classifier)