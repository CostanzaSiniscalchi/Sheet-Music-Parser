import os
import pickle as pkl
import xml.etree.ElementTree as ET
import torch

def parse_bounding_boxes_from_xml(xml_directory):

    bounding_boxes = {}

    for xml_file in os.listdir(xml_directory):
        print("outside")
        if xml_file.endswith(".xml"):
            print("inside")
            xml_path = os.path.join(xml_directory, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract the filename from the document attribute
            # document_name = root.attrib['document']
            # image_filename = f"{document_name}.png"  # Adjust extension if needed
            image_filename = xml_file.replace('.xml', '.png')

            # Find all bounding boxes for this image
            boxes = []
            
            # Adjust this path based on your XML structure
            for symbol in root.findall('.//Symbol'):  # Modify if needed
                try:
                    bounds = symbol.find('Bounds')
                    left = float(bounds.get('x')) if bounds is not None else 0.0
                    top = float(bounds.get('y')) if bounds is not None else 0.0
                    width = float(bounds.get('w')) if bounds is not None else 0.0
                    height = float(bounds.get('h')) if bounds is not None else 0.0
                    
                    # You can print or process these values as needed
                    print(f"Left: {left}, Top: {top}, Width: {width}, Height: {height}")

                    # Normalize bounding box coordinates
                    normalized_box = [
                        left / 1000.0,    # x normalized
                        top / 1000.0,     # y normalized
                        width / 1000.0,   # width normalized
                        height / 1000.0   # height normalized
                    ]
                    boxes.append(normalized_box)
                except Exception as e:
                    print(f"Error processing symbol: {e}")

            if boxes:
                bounding_boxes[image_filename] = torch.tensor(boxes, dtype=torch.float32)

    return bounding_boxes

def prepare_bounding_boxes_for_training(xml_directory, output_directory):

    os.makedirs(output_directory, exist_ok=True)
    
    bounding_boxes = parse_bounding_boxes_from_xml(xml_directory)
    
    output_path = os.path.join(output_directory, "bounding_boxes.pkl")
    with open(output_path, 'wb') as file:
        pkl.dump(bounding_boxes, file)
    
    print(f"Bounding boxes saved to: {output_path}")
    
    print(f"Total images with bounding boxes: {len(bounding_boxes)}")
    for img, boxes in list(bounding_boxes.items())[:5]: 
        print(f"{img}: {len(boxes)} bounding boxes")

if __name__ == "__main__":
    prepare_bounding_boxes_for_training(
        xml_directory=".",  
        output_directory="."  
    )
