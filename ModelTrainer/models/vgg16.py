import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adadelta
from models.TrainingConfiguration import TrainingConfiguration


class VggConfiguration(TrainingConfiguration):
    """A VGG-like configuration implemented in PyTorch."""

    def __init__(self, optimizer: str, width: int, height: int, training_minibatch_size: int, number_of_classes: int):
        super().__init__(optimizer=optimizer, data_shape=(height, width, 3),
                         training_minibatch_size=training_minibatch_size, number_of_classes=number_of_classes)
        self.weight_decay = 1e-4  # Corresponds to l2 regularization

    def classifier(self) -> nn.Sequential:
        """ Returns the model of this configuration """
        model = nn.Sequential()
        # Block 1
        self.add_convolution(model, 3, 16, 3)
        self.add_convolution(model, 16, 16, 3)
        model.add_module("MaxPool1", nn.MaxPool2d(kernel_size=2, stride=2))

        # Block 2
        self.add_convolution(model, 16, 32, 3)
        self.add_convolution(model, 32, 32, 3)
        model.add_module("MaxPool2", nn.MaxPool2d(kernel_size=2, stride=2)),

        # Block 3
        self.add_convolution(model, 32, 64, 3)
        self.add_convolution(model, 64, 64, 3)
        self.add_convolution(model, 64, 64, 3)
        model.add_module("MaxPool3", nn.MaxPool2d(kernel_size=2, stride=2)),

        # Block 4
        self.add_convolution(model, 64, 128, 3)
        self.add_convolution(model, 128, 128, 3)
        self.add_convolution(model, 128, 128, 3)
        model.add_module("MaxPool4", nn.MaxPool2d(kernel_size=2, stride=2)),

        # Block 5
        self.add_convolution(model, 128, 192, 3)
        self.add_convolution(model, 192, 192, 3)
        self.add_convolution(model, 192, 192, 3)
        self.add_convolution(model, 192, 192, 3)
        model.add_module("MaxPool5", nn.MaxPool2d(kernel_size=2, stride=2)),
    
        model.add_module("Flatten", nn.Flatten()) 
        model.add_module(
        "Dense", 
        nn.Linear(192 * (self.data_shape[0] // 32) * (self.data_shape[1] // 32), self.number_of_classes),
        )
        return model


    def add_convolution(self, model, input_size, output_size, kernel_size, stride=1, padding=1, input_shape=None):
        """Helper function to add a convolutional block."""
        model.add_module(f"Conv2d_{input_size}_{output_size}", nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=False))
        model.add_module(f"BatchNorm_{output_size}", nn.BatchNorm2d(output_size))
        model.add_module(f"ReLU_{output_size}", nn.ReLU(inplace=True))

    
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
        return "vgg"

    def performs_localization(self) -> bool:
        return False


if __name__ == "__main__":
    configuration = VggConfiguration("Adadelta", 96, 96, 16, 32)
    classifier = configuration.classifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)

    # Print model summary
    print(classifier)