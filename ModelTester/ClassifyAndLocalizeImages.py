import os
import sys
from argparse import ArgumentParser
from typing import List

import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import resnet50, vgg16
from torch.nn.functional import softmax

# List of class names
class_names = []


def load_model(model_name: str, model_path: str, num_classes: int):
    """
    Load a pre-trained PyTorch model.

    Args:
        model_name (str): Name of the model to load ('resnet' or 'vgg').
        model_path (str): Path to the saved model file.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The loaded model.
    """
    print(f"Loading {model_name} model from: {model_path}")

    # Initialize model based on the model_name
    if model_name == 'resnet':
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg':
        model = vgg16(pretrained=False)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Choose one of the following models: ['resnet', 'vgg']")

    # Load weights and set to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image_path: str, target_size=(192, 96)):
    """
    Preprocess the input image for the model.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size (height, width) for resizing.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    print(f"Loading and preprocessing image: {image_path}")

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # Add batch dimension


def test_model(model_name: str, model_path: str, image_paths: List[str]):
    """
    Test a pre-trained model on a list of images.

    Args:
        model_name (str): Name of the model being used.
        model_path (str): Path to the pre-trained model.
        image_paths (List[str]): List of image paths to classify.
    """
    global class_names

    # Number of classes is determined from the class names
    num_classes = len(class_names)
    model = load_model(model_name, model_path, num_classes)

    for image_path in image_paths:
        # Preprocess the image
        input_tensor = preprocess_image(image_path)

        # Perform inference
        print("Classifying image ...")
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = softmax(outputs[0], dim=0).numpy()

        # Get predicted class
        class_with_highest_probability = np.argmax(probabilities)
        most_likely_class = class_names[class_with_highest_probability]

        # Display classification results
        print(f"\nImage: {image_path}")
        print("Class scores:")
        for i, score in enumerate(probabilities):
            print(f"{class_names[i]:<18s} {score:.5f}")

        print(f"\nImage is most likely: {most_likely_class} "
              f"(certainty: {probabilities[class_with_highest_probability]:.2f})")

        # Draw bounding box (if applicable; currently simulated)
        bounding_box = [20, 30, 100, 80]  # Replace with actual bounding box if available
        red = (255, 0, 0)

        # Draw bounding box on the image
        original_image = Image.open(image_path).convert('RGB')
        image_with_bounding_box = original_image.copy()
        draw = ImageDraw.Draw(image_with_bounding_box)

        if bounding_box:
            rectangle = (bounding_box[0], bounding_box[1],
                         bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
            draw.rectangle(rectangle, fill=None, outline=red)

        # Save the result
        path = os.path.dirname(image_path)
        file_name, extension = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(path, f"{file_name}_{most_likely_class}_localization{extension}")
        image_with_bounding_box.save(output_path)
        print(f"Saved output image to: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Classify an RGB image with a pre-trained PyTorch model")
    parser.add_argument("-m", "--model_name", dest="model_name",
                        help="Model name (e.g., 'resnet', 'vgg')",
                        required=True)
    parser.add_argument("-c", "--classifier", dest="model_path",
                        help="Path to the classifier that contains the weights (*.pth)",
                        required=True)
    parser.add_argument("-i", "--images", dest="image_paths", nargs="+",
                        help="Path(s) to the RGB image(s) to classify", default=[])
    parser.add_argument("-d", "--image_directory", dest="image_directory",
                        help="Path to a folder containing images to classify", default="")

    args = parser.parse_args()

    # Validate input
    if len(args.image_paths) == 0 and len(args.image_directory) == 0:
        print("No data for classification provided. Aborting.")
        parser.print_help()
        sys.exit(-1)

    # Determine class names
    if args.image_directory:
        class_names = sorted(os.listdir(args.image_directory))
    else:
        class_names = ['12-8-Time', '2-2-Time', '2-4-Time', '3-4-Time', '3-8-Time', '4-4-Time', '6-8-Time',
                       '9-8-Time', 'Barline', 'C-Clef', 'Common-Time', 'Cut-Time', 'Dot', 'Double-Sharp',
                       'Eighth-Note', 'Eighth-Rest', 'F-Clef', 'Flat', 'G-Clef', 'Half-Note', 'Natural',
                       'Quarter-Note', 'Quarter-Rest', 'Sharp', 'Sixteenth-Note', 'Sixteenth-Rest',
                       'Sixty-Four-Note', 'Sixty-Four-Rest', 'Thirty-Two-Note', 'Thirty-Two-Rest',
                       'Whole-Half-Rest', 'Whole-Note']

    # Collect image files
    files = []
    if args.image_paths:
        files.extend(args.image_paths)
    if args.image_directory:
        files_in_directory = os.listdir(args.image_directory)
        images_in_directory = [os.path.join(args.image_directory, f) for f in files_in_directory
                                if f.endswith(("png", "jpg", "jpeg"))]
        files.extend(images_in_directory)

    # Run the test
    test_model(args.model_name, args.model_path, files)
