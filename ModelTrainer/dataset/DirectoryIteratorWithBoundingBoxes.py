import os
import torch
from PIL import Image

class DirectoryIteratorWithBoundingBoxes(torch.utils.data.Dataset):
    def __init__(self, directory, bounding_boxes=None, target_size=(256, 256), transform=None):
        """
        PyTorch Dataset for images with bounding boxes.

        Args:
            directory (str): Path to the directory containing image folders.
            bounding_boxes (dict): Dictionary with bounding box coordinates for images.
                                   Format: {'image_name.jpg': [x, y, width, height]}.
            target_size (tuple): Desired size of the images (height, width).
            transform (callable, optional): Transform to apply to the images.
        """
        self.directory = directory
        self.bounding_boxes = bounding_boxes
        self.target_size = target_size
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # Load image paths and labels
        for class_name in sorted(os.listdir(directory)):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                class_idx = len(self.classes)
                self.classes.append(class_name)
                self.class_to_idx[class_name] = class_idx

                for file_name in sorted(os.listdir(class_path)):
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        self.samples.append((file_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding label and bounding box.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label, bounding_box) where bounding_box is a tensor [x, y, width, height].
        """
        image_path, label = self.samples[idx]
        image_name = os.path.basename(image_path)

        # Load and resize image
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # Ensure RGB format
            if self.target_size:
                img = img.resize(self.target_size)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Get bounding box
        bounding_box = torch.tensor(self.bounding_boxes[image_name], dtype=torch.float32) \
            if self.bounding_boxes and image_name in self.bounding_boxes else None

        return img, label, bounding_box
    