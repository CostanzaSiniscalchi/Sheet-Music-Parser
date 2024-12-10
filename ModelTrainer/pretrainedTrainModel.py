import os
import argparse
import datetime
import pickle
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from models.ConfigurationFactory import ConfigurationFactory
from datasets.DirectoryIteratorWithBoundingBoxes import DirectoryIteratorWithBoundingBoxes
from ClassWeightCalculator import ClassWeightCalculator
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS, default_loader

class FilteredImageFolder(DatasetFolder):
    """Custom ImageFolder to handle classes with empty folders."""
    def __init__(self, root, transform=None, extensions=None):
        self.root = root
        self.extensions = extensions or IMG_EXTENSIONS
        self.loader = default_loader
        classes, class_to_idx = self.find_classes(root)

        samples = self.make_dataset(root, class_to_idx, self.extensions, is_valid_file=None)
        valid_classes = {c for c, idx in class_to_idx.items() if any(s[1] == idx for s in samples)}

        if len(valid_classes) == 0:
            print(f"Warning: No valid files found in {root}. Skipping this dataset.")
            super().__init__(root, loader=self.loader, extensions=self.extensions, transform=transform)
            return

        self.classes = sorted(valid_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [(s[0], self.class_to_idx[classes[s[1]]]) for s in samples if classes[s[1]] in valid_classes]
        super().__init__(root, loader=self.loader, extensions=self.extensions, transform=transform)

    def make_dataset(self, directory, class_to_idx, extensions, is_valid_file=None):
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file is not None:
                        if is_valid_file(path):
                            instances.append((path, class_to_idx[target_class]))
                    elif path.lower().endswith(tuple(extensions)):
                        instances.append((path, class_to_idx[target_class]))
        return instances

def load_dataset(dataset_directory, model_name, optimizer_name, height, width, training_minibatch_size,
                 class_weights_balancing_method):
    image_dataset_directory = os.path.join(dataset_directory, "images")
    bounding_boxes_cache = os.path.join(dataset_directory, "audiveris_omr_raw/bounding_boxes.pkl")
    bounding_boxes = None
    print("Loading configuration and data-readers...")
    number_of_classes = len(os.listdir(os.path.join(image_dataset_directory, "training")))
    training_configuration = ConfigurationFactory.get_configuration_by_name(
        model_name, optimizer_name, width, height, training_minibatch_size, number_of_classes
    )

    if os.path.exists(bounding_boxes_cache):
        with open(bounding_boxes_cache, "rb") as cache:
            bounding_boxes = pickle.load(cache)
    else:
        print("Bounding boxes file not found. Ensure it exists if localization is required.")
        bounding_boxes = None

    train_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((height, width), scale=(1.0 - 0.2, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if bounding_boxes:
        train_dataset = DirectoryIteratorWithBoundingBoxes(
            directory=os.path.join(image_dataset_directory, "training"),
            bounding_boxes=bounding_boxes,
            target_size=(height, width),
            transform=train_transform,
        )
        val_dataset = DirectoryIteratorWithBoundingBoxes(
            directory=os.path.join(image_dataset_directory, "validation"),
            bounding_boxes=bounding_boxes,
            target_size=(height, width),
            transform=val_test_transform,
        )
        test_dataset = DirectoryIteratorWithBoundingBoxes(
            directory=os.path.join(image_dataset_directory, "test"),
            bounding_boxes=bounding_boxes,
            target_size=(height, width),
            transform=val_test_transform,
        )
    else:
        train_dataset = ImageFolder(os.path.join(image_dataset_directory, "training"), transform=train_transform)
        val_dataset = FilteredImageFolder(os.path.join(image_dataset_directory, "validation"), transform=val_test_transform)
        test_dataset = FilteredImageFolder(os.path.join(image_dataset_directory, "test"), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=training_minibatch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_minibatch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=training_minibatch_size, shuffle=False) if test_dataset else None

    if class_weights_balancing_method:
        class_weight_calculator = ClassWeightCalculator()
        class_weights = class_weight_calculator.calculate_class_weights(image_dataset_directory, method=class_weights_balancing_method)
        class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float).to(device)
    else:
        class_weights_tensor = None

    return training_configuration, bounding_boxes, train_loader, val_loader, test_loader, class_weights_tensor

def train_loop(training_configuration, model, train_loader, val_loader, device, optimizer, criterion,
               localization=False, localization_criterion=None, bounding_boxes=None):
    if localization and bounding_boxes is None:
        raise ValueError("Bounding boxes are required for localization training.")

    best_val_accuracy = 0.0
    writer = SummaryWriter(log_dir=f"./logs/{datetime.date.today()}_{training_configuration.name()}")

    for epoch in range(3):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        localization_loss_sum = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = torch.tensor(labels).to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}], Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        if val_loader:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_inputs)
                    _, val_predicted = val_outputs.max(1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()

            val_accuracy = 100.0 * val_correct / val_total
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch model.")
    parser.add_argument("--dataset_directory", required=True, type=str)
    parser.add_argument("--model_name", required=True, choices=["resnet", "vgg", "vgg4_with_localization", "googlenet", "resnet_with_localization"], type=str) #"resnet_with_localization",
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--class_weights_balancing_method", type=str, default=None)
    parser.add_argument("--localization", action="store_true", help="Enable localization training")  
    args = parser.parse_args()

    training_configuration, bounding_boxes, train_loader, val_loader, test_loader, class_weights_tensor = load_dataset(
        args.dataset_directory, args.model_name, args.optimizer, 192, 96, args.minibatch_size,
        args.class_weights_balancing_method
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = training_configuration.classifier().to(device)

    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # print(f"Bounding boxes: {bounding_boxes}")
    train_loop(training_configuration, model, train_loader, val_loader, device, optimizer, 
            criterion, localization=True, 
            localization_criterion=nn.SmoothL1Loss(), 
            bounding_boxes=bounding_boxes)

    if hasattr(training_configuration, "performs_localization") and training_configuration.performs_localization():
        train_loop(training_configuration, model, train_loader, val_loader, device, optimizer, criterion,
                   localization=True, localization_criterion=nn.SmoothL1Loss(), bounding_boxes=bounding_boxes)
    else:
        train_loop(training_configuration, model, train_loader, val_loader, device, optimizer, criterion)