import os
import sys
from argparse import ArgumentParser
from typing import List

import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50, vgg16
from PIL import Image, ImageDraw

# List of class names
class_names = [
    '12-8-Time', '2-2-Time', '2-4-Time', '3-4-Time', '3-8-Time', '4-4-Time', '6-8-Time',
    '9-8-Time', 'Barline', 'C-Clef', 'Common-Time', 'Cut-Time', 'Dot', 'Double-Sharp',
    'Eighth-Note', 'Eighth-Rest', 'F-Clef', 'Flat', 'G-Clef', 'Half-Note', 'Natural',
    'Quarter-Note', 'Quarter-Rest', 'Sharp', 'Sixteenth-Note', 'Sixteenth-Rest',
    'Sixty-Four-Note', 'Sixty-Four-Rest', 'Thirty-Two-Note', 'Thirty-Two-Rest',
    'Whole-Half-Rest', 'Whole-Note'
]


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
    print("Loading model from:", model_path)

    if model_name == 'resnet':
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg':
        model = vgg16(pretrained=False)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Choose one of the following models: ['resnet', 'vgg']")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image_path: str):
    """
    Preprocess the image for model inference.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    print(f"Preprocessing image: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((192, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # Add batch dimension


def test_model(model_path: str, model_name: str, image_paths: List[str]):
    """
    Perform inference using a pre-trained model on a set of images.

    Args:
        model_path (str): Path to the saved model file.
        model_name (str): Name of the model being used.
        image_paths (List[str]): List of paths to images to classify.
    """
    global class_names

    num_classes = len(class_names)
    classifier = load_model(model_name, model_path, num_classes)

    for image_path in image_paths:
        # Preprocess image
        input_tensor = preprocess_image(image_path)

        # Perform inference
        print("Classifying image ...")
        with torch.no_grad():
            output = classifier(input_tensor)
            scores = torch.softmax(output[0], dim=0).numpy()

        # Get predicted class
        class_with_highest_probability = np.argmax(scores)
        most_likely_class = class_names[class_with_highest_probability]

        # Display results
        print("Class scores:")
        for i, score in enumerate(scores):
            print(f"{class_names[i]:<18s} {score:.5f}")

        print(f"Image is most likely: {most_likely_class} (certainty: {scores[class_with_highest_probability]:.2f})")

        # Simulate bounding box (if applicable)
        bounding_box = [20, 30, 100, 80]  # Example bounding box; replace with real data if available
        red = (255, 0, 0)

        # Draw bounding box on the image
        original_image = Image.open(image_path).convert('RGB')
        image_with_bounding_box = original_image.copy()
        draw = ImageDraw.Draw(image_with_bounding_box)
        rectangle = (bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
        draw.rectangle(rectangle, fill=None, outline=red)

        # Save the result
        path = os.path.dirname(image_path)
        file_name, extension = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(path, f"{file_name}_{most_likely_class}_localization{extension}")
        image_with_bounding_box.save(output_path)
        print(f"Saved image with bounding box to: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Classify an RGB image with a pre-trained PyTorch model")
    parser.add_argument("-c", "--classifier", dest="model_path",
                        help="Path to the classifier that contains the weights (*.pth)",
                        required=True)
    parser.add_argument("-m", "--model", dest="model_name",
                        help="Name of the model being used (e.g., 'resnet', 'vgg')",
                        required=True)
    parser.add_argument("-i", "--images", dest="image_paths", nargs="+",
                        help="Path(s) to the RGB image(s) to classify",
                        required=True)

    args = parser.parse_args()

    if not args.model_path or not args.model_name or not args.image_paths:
        parser.print_help()
        sys.exit(-1)

    test_model(args.model_path, args.model_name, args.image_paths)