import os
import xml.etree.ElementTree as ET

def parse_bounding_boxes_from_xml(xml_directory):
    """
    Parses XML files to extract bounding box information.

    Args:
        xml_directory (str): Path to the directory containing XML files.

    Returns:
        dict: A dictionary with image filenames as keys and bounding box coordinates as values.
    """
    bounding_boxes = {}

    # Iterate through XML files in the directory
    for xml_file in os.listdir(xml_directory):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_directory, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract the document name as the image filename (adjust if necessary)
            document_name = root.attrib['document']
            image_filename = f"{document_name}.jpg"  # Assuming image files are .jpg

            # Extract nodes for bounding boxes
            bounding_boxes[image_filename] = []  # Each image may have multiple bounding boxes
            for node in root.findall('Node'):
                left = int(node.find('Left').text)
                top = int(node.find('Top').text)
                width = int(node.find('Width').text)
                height = int(node.find('Height').text)

                # Append the bounding box coordinates as a dictionary
                bounding_boxes[image_filename].append({
                    "x": left,
                    "y": top,
                    "width": width,
                    "height": height
                })

    return bounding_boxes

# Example usage
xml_directory = "./annotations"  # Path to XML directory
bounding_boxes = parse_bounding_boxes_from_xml(xml_directory)
print(bounding_boxes)
