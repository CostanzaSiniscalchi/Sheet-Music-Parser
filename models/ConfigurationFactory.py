from typing import List
from models.resnet50 import ResNet1Configuration
from models.vgg16 import VggConfiguration
from models.TrainingConfiguration import TrainingConfiguration


class ConfigurationFactory:
    @staticmethod
    def get_configuration_by_name(name: str,
                                  optimizer: str,
                                  width: int,
                                  height: int,
                                  training_minibatch_size: int,
                                  number_of_classes: int) -> TrainingConfiguration:

        configurations = ConfigurationFactory.get_all_configurations(optimizer, width, height, training_minibatch_size,
                                                                     number_of_classes)

        for i in range(len(configurations)):
            if configurations[i].name() == name:
                return configurations[i]

        raise Exception("No configuration found by name {0}".format(name))

    @staticmethod
    def get_all_configurations(optimizer, width, height, training_minibatch_size, number_of_classes) -> List[
        TrainingConfiguration]:
        configurations = [ResNet1Configuration(optimizer, width, height, training_minibatch_size, number_of_classes),
                          VggConfiguration(optimizer, width, height, training_minibatch_size, number_of_classes),]
        return configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations("SGD", 1, 1, 1, 1)
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())