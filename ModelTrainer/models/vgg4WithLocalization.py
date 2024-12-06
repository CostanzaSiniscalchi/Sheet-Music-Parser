import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Adadelta
from torchsummary import summary
from models.TrainingConfiguration import TrainingConfiguration

class Vgg4WithLocalizationConfiguration(TrainingConfiguration):
    """ The winning VGG-Net 4 configuration from Deep Learning course """

    def __init__(self, optimizer, width, height, training_minibatch_size, number_of_classes):
        super().__init__(optimizer=optimizer, data_shape=(3, height, width),
                         training_minibatch_size=training_minibatch_size, number_of_classes=number_of_classes,
                         number_of_epochs_before_early_stopping=30,
                         number_of_epochs_before_reducing_learning_rate=14)
        self.weight_decay = self.weight_decay

    def classifier(self) -> nn.Module:
        """ Returns the model of this configuration """
        return Vgg4Model(self.data_shape, self.number_of_classes, self.weight_decay)

    def name(self) -> str:
        """ Returns the name of this configuration """
        return "vgg4_with_localization"

    def performs_localization(self) -> bool:
        return True


class Vgg4Model(nn.Module):
    def __init__(self, input_shape, num_classes, weight_decay):
        super(Vgg4Model, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        # Convolutional blocks
        self.conv_block1 = self._create_conv_block(3, 32)
        self.conv_block2 = self._create_conv_block(32, 64)
        self.conv_block3 = self._create_conv_block(64, 128, layers=3)
        self.conv_block4 = self._create_conv_block(128, 256, layers=3)
        self.conv_block5 = self._create_conv_block(256, 512, layers=3, pooling="avg")

        # Dynamically calculate the size of the flattened feature vector
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            flattened_size = self._get_flattened_size(dummy_input)

        self.flatten = nn.Flatten()
        self.classification_head = nn.Linear(flattened_size, num_classes)
        self.regression_head = nn.Linear(flattened_size, 4)

    def _create_conv_block(self, in_channels, out_channels, kernel_size=3, layers=2, pooling="max"):
        modules = []
        for _ in range(layers):
            modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same"))
            modules.append(nn.BatchNorm2d(out_channels))
            modules.append(nn.ReLU())
            in_channels = out_channels
        if pooling == "max":
            modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif pooling == "avg":
            modules.append(nn.AvgPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*modules)

    def _get_flattened_size(self, x):
        """Pass a dummy tensor through the convolutional layers to calculate the flattened size."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x.view(1, -1).size(1)  # Flatten and get the size

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        features = self.flatten(x)
        classification_output = F.softmax(self.classification_head(features), dim=1)
        regression_output = self.regression_head(features)  # No activation for regression
        return classification_output, regression_output



if __name__ == "__main__":
    # Example instantiation
    configuration = Vgg4WithLocalizationConfiguration("Adadelta", 112, 112, 16, 32)
    model = configuration.classifier()

    # Display model summary
    summary(model, input_size=(3, 112, 112))

    # Print configuration summary
    print(configuration.summary())
