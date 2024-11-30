from abc import ABC, abstractmethod
from torch import nn, optim


class TrainingConfiguration(ABC):
    """Base class for a configuration that specifies the hyperparameters of training."""

    def __init__(self,
                 data_shape: tuple = (224, 128, 3),  # Rows = Height, columns = Width, channels = typically 3 (RGB)
                 number_of_classes: int = 32,
                 number_of_epochs: int = 2,
                 number_of_epochs_before_early_stopping: int = 20,
                 number_of_epochs_before_reducing_learning_rate: int = 8,
                 training_minibatch_size: int = 64,
                 initialization: str = "xavier_uniform",
                 learning_rate: float = 0.01,
                 learning_rate_reduction_factor: float = 0.5,
                 minimum_learning_rate: float = 0.00001,
                 weight_decay: float = 0.0001,
                 nesterov_momentum: float = 0.9,
                 zoom_range: float = 0.2,
                 rotation_range: int = 10,
                 optimizer: str = "SGD"):
        """
        :param data_shape: Tuple with order (rows, columns, channels).
        :param zoom_range: Percentage that the input will dynamically be zoomed during training (0-1).
        :param rotation_range: Random rotation of the input image during training in degrees.
        :param optimizer: The optimizer for training, currently supported are 'SGD', 'Adam', or 'Adadelta'.
        """
        self.optimizer = optimizer
        self.rotation_range = rotation_range
        self.data_shape = data_shape
        self.number_of_classes = number_of_classes
        self.input_image_rows, self.input_image_columns, self.input_image_channels = data_shape
        self.number_of_epochs = number_of_epochs
        self.zoom_range = zoom_range
        self.number_of_epochs_before_early_stopping = number_of_epochs_before_early_stopping
        self.number_of_epochs_before_reducing_learning_rate = number_of_epochs_before_reducing_learning_rate
        self.training_minibatch_size = training_minibatch_size
        self.initialization = initialization
        self.learning_rate = learning_rate
        self.learning_rate_reduction_factor = learning_rate_reduction_factor
        self.minimum_learning_rate = minimum_learning_rate
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum

    @abstractmethod
    def classifier(self) -> nn.Module:
        """Returns the classifier of this configuration."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Returns the name of this configuration."""
        pass

    @abstractmethod
    def performs_localization(self) -> bool:
        """Returns whether this configuration has a regression head that performs object localization."""
        pass

    def get_optimizer(self, model_parameters):
        """
        Returns the configured optimizer for this configuration.

        :param model_parameters: Parameters of the model to optimize.
        :return: Configured optimizer.
        """
        if self.optimizer == "SGD":
            return optim.SGD(model_parameters, lr=self.learning_rate, momentum=self.nesterov_momentum, nesterov=True,
                             weight_decay=self.weight_decay)
        if self.optimizer == "Adam":
            return optim.Adam(model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer == "Adadelta":
            return optim.Adadelta(model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)

        raise ValueError(f"Invalid optimizer '{self.optimizer}' requested.")

    def get_initial_learning_rate(self) -> float:
        """
        Returns the initial learning rate.

        :return: Initial learning rate.
        """
        return self.learning_rate

    def summary(self) -> str:
        """Returns the string that summarizes this configuration."""
        summary = f"Training for {self.number_of_epochs} epochs ...\n"
        summary += f"Additional parameters: Initialization: {self.initialization}, Weight-decay of {self.weight_decay}, " \
                   f"Minibatch-size: {self.training_minibatch_size}, Early stopping after " \
                   f"{self.number_of_epochs_before_early_stopping} epochs without improvement.\n"
        summary += f"Data-Shape: {self.data_shape}, Reducing learning rate by factor {self.learning_rate_reduction_factor} " \
                   f"if validation accuracy does not improve after " \
                   f"{self.number_of_epochs_before_reducing_learning_rate} epochs.\n"
        summary += f"Data-augmentation: Zooming {self.zoom_range * 100}%, rotating {self.rotation_range}° randomly.\n"
        summary += f"Optimizer: {self.optimizer}, Learning Rate: {self.learning_rate}.\n"
        summary += f"Performing object localization: {self.performs_localization()}."
        return summary