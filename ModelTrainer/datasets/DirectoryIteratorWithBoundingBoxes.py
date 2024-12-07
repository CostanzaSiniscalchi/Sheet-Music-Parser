import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DirectoryIteratorWithBoundingBoxes(Dataset):
    def __init__(self, directory, bounding_boxes=None, target_size=(256, 256),
                 transform=None, class_mode='categorical', classes=None):
        """
        Args:
            directory (str): Path to the image directory.
            bounding_boxes (dict): Dictionary with filenames as keys and bounding box coordinates as values.
            target_size (tuple): Image size (width, height).
            transform (callable): Transformations to apply to the images.
            class_mode (str): 'categorical', 'binary', or 'sparse'.
            classes (list): List of class names.
        """
        self.directory = directory
        self.bounding_boxes = bounding_boxes
        self.target_size = target_size
        self.transform = transform
        self.class_mode = class_mode
        self.classes = classes
        
        self.filenames = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)} if classes else None

        # Load filenames and labels
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.filenames.append(os.path.join(root, file))
                    if self.classes:
                        class_label = os.path.basename(root)
                        self.labels.append(self.class_to_idx[class_label])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert("RGB").resize(self.target_size)

        if self.transform:
            img = self.transform(img)

        # Load bounding box
        file_name = os.path.basename(img_path)
        bounding_box = torch.zeros((4,), dtype=torch.float32)  # Default bounding box (zeroed)
        if self.bounding_boxes and file_name in self.bounding_boxes:
            bbox = self.bounding_boxes[file_name]
            bounding_box = torch.tensor([bbox['origin']['x'], bbox['origin']['y'], 
                                         bbox['width'], bbox['height']], dtype=torch.float32)

        # Load label
        if self.classes:
            label = self.labels[idx]
            if self.class_mode == 'categorical':
                label = torch.eye(len(self.classes))[label]
            elif self.class_mode == 'binary':
                label = torch.tensor(label, dtype=torch.float32)
            elif self.class_mode == 'sparse':
                label = torch.tensor(label, dtype=torch.long)
            return img, (label, bounding_box)
        else:
            return img, bounding_box

# Example usage
if __name__ == "__main__":
    # Example bounding boxes
    bounding_boxes = {
        "image1.jpg": {"origin": {"x": 10, "y": 20}, "width": 50, "height": 30},
        "image2.jpg": {"origin": {"x": 15, "y": 25}, "width": 40, "height": 35},
    }

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((192, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and data loader
    dataset = DirectoryIteratorWithBoundingBoxes(
        directory="../data/images/test",
        bounding_boxes=bounding_boxes,
        target_size=(192, 96),
        transform=transform,
        class_mode="categorical",
        classes=["class1", "class2"]
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Iterate through batches
    for batch_x, batch_y in dataloader:
        print("Batch of images shape:", batch_x.shape)
        if isinstance(batch_y, tuple):
            labels, bboxes = batch_y
            print("Batch of labels shape:", labels.shape)
            print("Batch of bounding boxes shape:", bboxes.shape)
        else:
            print("Batch of bounding boxes shape:", batch_y.shape)
        break
