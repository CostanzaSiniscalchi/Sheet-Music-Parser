import numpy as np
import matplotlib.pyplot as plt

class TrainingHistoryPlotter:
    @staticmethod
    def plot_history(history: dict, file_name: str, show_plot: bool = False):
        """
        Plot training and validation metrics.

        Args:
            history (dict): Dictionary containing training and validation metrics.
                            Expected keys: 'epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy', etc.
            file_name (str): File path to save the plot.
            show_plot (bool): Whether to display the plot.
        """
        # Extract epoch list
        epoch_list = np.arange(1, len(history['epoch']) + 1)

        fig = plt.figure(1)

        # Regular plot for classification tasks
        if "val_accuracy" in history:
            TrainingHistoryPlotter.add_subplot(
                epoch_list, fig, history, 211, "Loss", "loss", "Training loss",
                "val_loss", "Validation loss", "upper right"
            )
            TrainingHistoryPlotter.add_subplot(
                epoch_list, fig, history, 212, "Accuracy", "accuracy", "Training accuracy",
                "val_accuracy", "Validation accuracy", "lower right"
            )
        else:
            # Example for additional metrics (e.g., bounding-box loss/accuracy)
            TrainingHistoryPlotter.add_subplot(
                epoch_list, fig, history, 221, "Classification Loss", "output_class_loss",
                "Training loss", "val_output_class_loss", "Validation loss", "upper right"
            )
            TrainingHistoryPlotter.add_subplot(
                epoch_list, fig, history, 222, "Classification Accuracy", "output_class_acc",
                "Training accuracy", "val_output_class_acc", "Validation accuracy", "lower right"
            )
            TrainingHistoryPlotter.add_subplot(
                epoch_list, fig, history, 223, "Bounding-Box Loss", "output_bounding_box_loss",
                "Training loss", "val_output_bounding_box_loss", "Validation loss", "upper right"
            )
            TrainingHistoryPlotter.add_subplot(
                epoch_list, fig, history, 224, "Bounding-Box Accuracy", "output_bounding_box_acc",
                "Training accuracy", "val_output_bounding_box_acc", "Validation accuracy", "lower right"
            )

        plt.tight_layout()
        plt.savefig(file_name)

        if show_plot:
            plt.show()

    @staticmethod
    def add_subplot(epoch_list, fig, history, subplot_region, y_axis_label,
                    history_parameter1, parameter1_label,
                    history_parameter2, parameter2_label, legend_position):
        """
        Add a subplot to the figure for specific metrics.

        Args:
            epoch_list (array): Array of epoch numbers.
            fig (matplotlib.figure.Figure): Figure object to add the subplot.
            history (dict): Dictionary containing training metrics.
            subplot_region (int): Subplot region identifier (e.g., 211, 212).
            y_axis_label (str): Label for the Y-axis.
            history_parameter1 (str): Key for the first metric (e.g., 'loss').
            parameter1_label (str): Label for the first metric line.
            history_parameter2 (str): Key for the second metric (e.g., 'val_loss').
            parameter2_label (str): Label for the second metric line.
            legend_position (str): Legend position (e.g., 'upper right').
        """
        ax = fig.add_subplot(subplot_region)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_axis_label)
        ax.plot(epoch_list, history[history_parameter1], '--', linewidth=2, label=parameter1_label)
        ax.plot(epoch_list, history[history_parameter2], label=parameter2_label)
        ax.legend(loc=legend_position)

