import os
import shutil
import random
from pathlib import Path

# Source and destination paths
source_dir = "data/data/muscima_images"
destination_dir = "data/data/images"

# Total number of images in the sample
total_sample_size = 10000

# Create the destination directory
Path(destination_dir).mkdir(parents=True, exist_ok=True)

# Get a list of all classes and their respective image files
classes = [cls for cls in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, cls))]

# Calculate the total number of images in the dataset
class_image_counts = {cls: len(os.listdir(os.path.join(source_dir, cls))) for cls in classes}
total_images = sum(class_image_counts.values())

# Calculate the proportional number of images to sample for each class
sample_counts = {cls: int((count / total_images) * total_sample_size) for cls, count in class_image_counts.items()}

random.seed(0)

min_class_size = 300 

dropped_classes = []
# Sample images for each class and copy to the destination folder
for cls in classes:
    source_class_dir = os.path.join(source_dir, cls)
    destination_class_dir = os.path.join(destination_dir, cls)
    
    # Get all images in the class directory
    images = os.listdir(source_class_dir)
    print(f'Number of images for class {cls}: {len(images)}')

    # Skip class if number of images below minimum 
    if len(images) < min_class_size : 
        dropped_classes.append(cls)
        continue

    random.shuffle(images)
    
    # Select the required number of samples
    sampled_images = images[:sample_counts[cls]]
    
    # Create destination directory for the sample 
    Path(destination_class_dir).mkdir(parents=True, exist_ok=True)

    # Copy sampled images to the destination directory
    for image in sampled_images:
        src = os.path.join(source_class_dir, image)
        dest = os.path.join(destination_class_dir, image)
        shutil.copy(src, dest)

print(f"The following classes were dropped due to number of images beign < {min_class_size}:\n {dropped_classes}")
print(f"Sample dataset saved successfully under", destination_dir)