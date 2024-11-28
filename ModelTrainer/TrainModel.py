
import os
import argparse
import datetime
import pickle
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from models.ConfigurationFactory import ConfigurationFactory
from datasets.DirectoryIteratorWithBoundingBoxes import DirectoryIteratorWithBoundingBoxes
from ClassWeightCalculator import ClassWeightCalculator

def train_model(dataset_directory, model_name, width, height,
                training_minibatch_size, optimizer_name, dynamic_learning_rate_reduction,
                class_weights_balancing_method, save_after_every_epoch, resume_from_checkpoint):
    
    # Ensure device compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_of_training = datetime.date.today()

    # Dataset paths
    image_dataset_directory = os.path.join(dataset_directory, "images")
    bounding_boxes_cache = os.path.join(dataset_directory, "bounding_boxes.txt")
    bounding_boxes = None

    print("Loading configuration and data-readers...")
    number_of_classes = len(os.listdir(os.path.join(image_dataset_directory, "training")))
    training_configuration = ConfigurationFactory.get_configuration_by_name(
        model_name, optimizer_name, width, height, training_minibatch_size, number_of_classes
    )

    # Load bounding boxes if localization is required
    if os.path.exists(bounding_boxes_cache):
        with open(bounding_boxes_cache, "rb") as cache:
            bounding_boxes = pickle.load(cache)

    # Data augmentation and preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((height, width), scale=(1.0 - 0.2, 1.0)),  # Placeholder for zoom range
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    if bounding_boxes is not None:
        train_dataset = DirectoryIteratorWithBoundingBoxes(
            directory=os.path.join(image_dataset_directory, "training"),
            bounding_boxes=bounding_boxes,
            target_size=(height, width),
            transform=train_transform
        )
        val_dataset = DirectoryIteratorWithBoundingBoxes(
            directory=os.path.join(image_dataset_directory, "validation"),
            bounding_boxes=bounding_boxes,
            target_size=(height, width),
            transform=val_test_transform
        )
        test_dataset = DirectoryIteratorWithBoundingBoxes(
            directory=os.path.join(image_dataset_directory, "test"),
            bounding_boxes=bounding_boxes,
            target_size=(height, width),
            transform=val_test_transform
        )
    else:
        train_dataset = ImageFolder(os.path.join(image_dataset_directory, "training"), transform=train_transform)
        val_dataset = ImageFolder(os.path.join(image_dataset_directory, "validation"), transform=val_test_transform)
        test_dataset = ImageFolder(os.path.join(image_dataset_directory, "test"), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=training_minibatch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_minibatch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=training_minibatch_size, shuffle=False)

    print(f"Number of classes: {len(train_dataset.classes)}")

    # Initialize model
    print("Loading model...")
    num_classes = len(train_dataset.classes)
    model = training_configuration.classifier()
    model = model.to(device)

    # Loss and optimizer
    if class_weights_balancing_method:
        class_weight_calculator = ClassWeightCalculator()
        class_weights = class_weight_calculator.calculate_class_weights(dataset_directory, method=class_weights_balancing_method)
        class_weights_tensor = torch.tensor(list(class_weights.values())).to(device)
    else:
        class_weights_tensor = None


    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=0.001)

    # Learning Rate Scheduler
    if dynamic_learning_rate_reduction:
        scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                                         patience=training_configuration.number_of_epochs_before_reducing_learning_rate,
                                                         factor=training_configuration.learning_rate_reduction_factor,
                                                         min_lr=training_configuration.minimum_learning_rate)
    else:
        scheduler = None

    writer = SummaryWriter(log_dir=f"./logs/{start_of_training}_{model_name}")

    # Resume from checkpoint
    initial_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from {resume_from_checkpoint}, starting from epoch {initial_epoch}")

    # Training loop
    best_val_accuracy = 0.0
    for epoch in range(initial_epoch, training_configuration.number_of_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        writer.add_scalar("Loss/Train", running_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        print(f"Epoch {epoch + 1}, Train Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_accuracy = 100.0 * val_correct / val_total
        writer.add_scalar("Loss/Validation", val_loss / len(val_loader), epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        if scheduler:
            scheduler.step(val_accuracy)

        # Save best model
        if save_after_every_epoch or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
            }, f"{model_name}_epoch_{epoch + 1}.pth")
            print(f"Best model saved with accuracy: {val_accuracy:.2f}%")

    print("Training completed. Testing...")
    model.eval()
    test_correct, test_total = 0, 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            test_total += labels.size(0)
            test_correct += preds.eq(labels).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=train_dataset.classes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch model for symbol classification.")
    parser.add_argument("--dataset_directory", required=True, type=str)
    parser.add_argument("--model_name", required=True, choices=["resnet", "vgg"], type=str)
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    train_model(
        dataset_directory=args.dataset_directory,
        model_name=args.model_name,
        width=192,
        height=96,
        training_minibatch_size=args.minibatch_size,
        optimizer_name=args.optimizer,
        dynamic_learning_rate_reduction=True,
        class_weights_balancing_method=None,
        save_after_every_epoch=False,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
