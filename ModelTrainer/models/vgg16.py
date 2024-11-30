import torch
import torch.nn as nn
# from torchsummary import summary

class VggConfiguration(nn.Module):
    """VGG Configuration in PyTorch"""
    def __init__(self, optimizer, width: int, height: int, training_minibatch_size,number_of_classes: int, weight_decay: float = 1e-4):
        super(VggConfiguration, self).__init__()
        
        self.weight_decay = weight_decay
        self.input_shape = (3, height, width)
        self.name = 'vgg'
        self.optimizer = optimizer
        self.training_minibatch_size = training_minibatch_size

        self.features = nn.Sequential(
            # Block 1
            self.add_convolution(3, 16, 3),
            self.add_convolution(16, 16, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            self.add_convolution(16, 32, 3),
            self.add_convolution(32, 32, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            self.add_convolution(32, 64, 3),
            self.add_convolution(64, 64, 3),
            self.add_convolution(64, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            self.add_convolution(64, 128, 3),
            self.add_convolution(128, 128, 3),
            self.add_convolution(128, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            self.add_convolution(128, 192, 3),
            self.add_convolution(192, 192, 3),
            self.add_convolution(192, 192, 3),
            self.add_convolution(192, 192, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * (width // 32) * (height // 32), number_of_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def add_convolution(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        """Helper function to create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def name(self) -> str:
        """ Returns the name of this configuration """
        return "vgg"

    def performs_localization(self) -> bool:
        return False
    
if __name__ == "__main__":
    configuration = VggConfiguration("Adadelta", 96, 96, 16, 32)
    classifier = configuration.classifier()
    classifier = classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # print(summary(classifier))