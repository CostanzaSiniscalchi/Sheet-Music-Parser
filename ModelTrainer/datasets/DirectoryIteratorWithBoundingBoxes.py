import os
import json
import xml.etree.ElementTree as ET

def parse_bounding_boxes_from_xml(xml_directory):
    """
    Parses XML files to extract bounding box information.

    Args:
        xml_directory (str): Path to the directory containing XML files.

    Returns:
        dict: A dictionary with image filenames as keys and bounding box data as values.
    """
    bounding_boxes = {}

    # Iterate through XML files in the directory
    for xml_file in os.listdir(xml_directory):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_directory, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract the document name as the image filename
            document_name = root.attrib['document']
            image_filename = f"{document_name}.jpg"  # Assuming images are .jpg files

            for node in root.findall('Node'):
                left = int(node.find('Left').text)
                top = int(node.find('Top').text)
                width = int(node.find('Width').text)
                height = int(node.find('Height').text)

                # Store bounding box in the required format
                bounding_boxes[image_filename] = {
                    "origin": {"x": left, "y": top},
                    "width": width,
                    "height": height
                }

    return bounding_boxes

# Example usage
xml_directory = "/Users/costanzasiniscalchi/Documents/MS/DLCV/Sheet-Music-Parser/ModelTrainer/datasets/data/data/muscima_pp_raw/v2.0/data/annotations"  # Path to directory with XML files
output_json_path = "/Users/costanzasiniscalchi/Documents/MS/DLCV/Sheet-Music-Parser/ModelTrainer/datasets/data/data/muscima_pp_raw/v2.0/data/bounding_boxes.json"
bounding_boxes = parse_bounding_boxes_from_xml(xml_directory)
print(bounding_boxes)
# Write the dictionary to the JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(bounding_boxes, json_file, indent=4)

print(f"Bounding boxes saved to: {output_json_path}")
