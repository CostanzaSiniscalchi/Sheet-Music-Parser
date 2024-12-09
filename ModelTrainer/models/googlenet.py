import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta
from models.TrainingConfiguration import TrainingConfiguration


class GoogleNetConfiguration(TrainingConfiguration):
    """A network with Inception modules implemented in PyTorch."""

    def __init__(self, optimizer: str, width: int, height: int, training_minibatch_size: int, number_of_classes: int):
        super().__init__(optimizer=optimizer, data_shape=(height, width, 3),
                         training_minibatch_size=training_minibatch_size, number_of_classes=number_of_classes)
        self.weight_decay = 1e-4

    def classifier(self) -> nn.Sequential:
        """Returns the PyTorch implementation of the GoogleNet model."""
        layers = nn.Sequential()

        # Initial Convolution Layers
        self.add_convolution_block_with_batch_normalization(layers, 3, 64, 7, stride=2, layer_number=1)
        layers.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.add_convolution_block_with_batch_normalization(layers, 64, 64, 1, stride=1, layer_number=2)
        self.add_convolution_block_with_batch_normalization(layers, 64, 192, 3, stride=1, layer_number=3)
        layers.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Inception Modules
        layers.add_module("inception3a", self.create_inception_module(192, 64, 96, 128, 16, 32, 32))
        layers.add_module("inception3b", self.create_inception_module(256, 128, 128, 192, 32, 96, 64))
        layers.add_module("maxpool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.add_module("inception4a", self.create_inception_module(480, 192, 96, 208, 16, 48, 64))
        layers.add_module("inception4b", self.create_inception_module(512, 160, 112, 224, 24, 64, 64))
        layers.add_module("inception4c", self.create_inception_module(512, 128, 128, 256, 24, 64, 64))
        layers.add_module("inception4d", self.create_inception_module(512, 112, 144, 288, 32, 64, 64))
        layers.add_module("inception4e", self.create_inception_module(528, 256, 160, 320, 32, 128, 128))
        layers.add_module("maxpool4", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.add_module("inception5a", self.create_inception_module(832, 256, 160, 320, 32, 128, 128))
        layers.add_module("inception5b", self.create_inception_module(832, 384, 192, 384, 48, 128, 128))

        # Global Average Pooling
        layers.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))

        # Dropout and Fully Connected Layer
        layers.add_module("dropout", nn.Dropout(0.4))
        layers.add_module("flatten", nn.Flatten())
        layers.add_module("fc", nn.Linear(1024, self.number_of_classes))

        return layers

    def add_convolution_block_with_batch_normalization(self, model, input_size, output_size, kernel_size,
                                                       stride, layer_number: int) -> nn.Sequential:
        model.add_module(f"Conv2d_{layer_number}", nn.Conv2d(input_size, output_size, kernel_size, stride, 1, bias=False))
        model.add_module(f"BatchNorm_{layer_number}", nn.BatchNorm2d(output_size))
        model.add_module(f"ReLU_{layer_number}", nn.ReLU(inplace=True))

    def create_inception_module(self, in_channels, c1x1, c3x3_reduce, c3x3, c5x5_reduce, c5x5, pool_proj):
        """
        Create an Inception module with multiple parallel convolution paths
        
        Args:
        - in_channels: Number of input channels
        - c1x1: Number of 1x1 convolution filters
        - c3x3_reduce: Number of 1x1 convolution filters before 3x3 conv
        - c3x3: Number of 3x3 convolution filters
        - c5x5_reduce: Number of 1x1 convolution filters before 5x5 conv
        - c5x5: Number of 5x5 convolution filters
        - pool_proj: Number of 1x1 convolution filters after max pooling
        """
        class InceptionModule(nn.Module):
            def __init__(self, in_channels, c1x1, c3x3_reduce, c3x3, c5x5_reduce, c5x5, pool_proj):
                super(InceptionModule, self).__init__()
                
                # 1x1 convolution path
                self.branch1 = nn.Sequential(
                    nn.Conv2d(in_channels, c1x1, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(c1x1),
                    nn.ReLU(inplace=True)
                )
                
                # 3x3 convolution path
                self.branch2 = nn.Sequential(
                    nn.Conv2d(in_channels, c3x3_reduce, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(c3x3_reduce),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c3x3_reduce, c3x3, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(c3x3),
                    nn.ReLU(inplace=True)
                )
                
                # 5x5 convolution path
                self.branch3 = nn.Sequential(
                    nn.Conv2d(in_channels, c5x5_reduce, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(c5x5_reduce),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c5x5_reduce, c5x5, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.BatchNorm2d(c5x5),
                    nn.ReLU(inplace=True)
                )
                
                # Pooling path
                self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_channels, pool_proj, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(pool_proj),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                branch1 = self.branch1(x)
                branch2 = self.branch2(x)
                branch3 = self.branch3(x)
                branch4 = self.branch4(x)
                return torch.cat([branch1, branch2, branch3, branch4], 1)

        return InceptionModule(in_channels, c1x1, c3x3_reduce, c3x3, c5x5_reduce, c5x5, pool_proj)

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
        return "googlenet"

    def performs_localization(self) -> bool:
        """Indicates if this configuration performs localization."""
        return False

if __name__ == "__main__":
    classifier = GoogleNetConfiguration(width=96, height=96, number_of_classes=32, optimizer="Adam", training_minibatch_size=64)
    classifier = classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Print model summary
    print(classifier)