import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DirectoryDatasetWithBoundingBoxes(Dataset):
    def __init__(self, directory, bounding_boxes: dict = None, target_size=(256, 256),
                 transform=None, class_mode: str = 'categorical', classes=None):
        """
        Args:
            directory (str): Path to the image directory.
            bounding_boxes (dict): A dictionary with filenames as keys and bounding box coordinates as values.
            target_size (tuple): Tuple of target image size (width, height).
            transform (callable, optional): A function/transform to apply to the images.
            class_mode (str): One of "categorical", "binary", or "sparse".
            classes (list, optional): List of class labels.
        """
        self.directory = directory
        self.bounding_boxes = bounding_boxes
        self.target_size = target_size
        self.transform = transform
        self.class_mode = class_mode
        self.classes = classes
        
        self.filenames = []
        self.labels = []
        self.class_to_index = {cls: idx for idx, cls in enumerate(classes)} if classes else None
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.filenames.append(os.path.join(root, file))
                    if self.classes:
                        class_label = os.path.basename(root)
                        self.labels.append(self.class_to_index[class_label])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert("RGB")
        if self.target_size:
            img = img.resize(self.target_size)
        
        if self.transform:
            img = self.transform(img)
        
        bounding_box = None
        if self.bounding_boxes:
            file_name = os.path.basename(img_path)
            bounding_box = self.bounding_boxes[file_name]
            bounding_box = torch.tensor([bounding_box.origin.x, bounding_box.origin.y,
                                         bounding_box.width, bounding_box.height], dtype=torch.float32)
        
        label = None
        if self.classes:
            label = self.labels[idx]
            if self.class_mode == "categorical":
                label = torch.eye(len(self.classes))[label]
            elif self.class_mode == "binary":
                label = torch.tensor(label, dtype=torch.float32)
            elif self.class_mode == "sparse":
                label = torch.tensor(label, dtype=torch.long)
        
        if self.bounding_boxes:
            return img, (label, bounding_box)
        return img, label


if __name__ == "__main__":
    # Example 
    bounding_boxes = {
        "image1.jpg": {"origin": {"x": 10, "y": 20}, "width": 50, "height": 30},
        "image2.jpg": {"origin": {"x": 15, "y": 25}, "width": 40, "height": 35},
    }
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = DirectoryDatasetWithBoundingBoxes(
        directory="../data/images/test",
        bounding_boxes=bounding_boxes,
        target_size=(192, 96),
        transform=transform,
        class_mode="categorical",
        classes=["class1", "class2"]
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch_x, batch_y in dataloader:
        print(batch_x.shape)  
        if isinstance(batch_y, tuple): 
            print(batch_y[0].shape)  # Shape of labels
            print(batch_y[1].shape)  # Shape of bounding boxes
        else:
            print(batch_y.shape)  # Shape of labels (if no bounding boxes)
