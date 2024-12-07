import os
import pickle as pkl
import xml.etree.ElementTree as ET

def parse_bounding_boxes_from_xml(xml_directory):
    """
    Parses XML files to extract multiple bounding box information per image.

    Args:
        xml_directory (str): Path to the directory containing XML files.

    Returns:
        dict: A dictionary with image filenames as keys and lists of bounding box data as values.
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
            image_filename = f"{document_name}.png"  # Adjust extension if needed

            # Add an empty list to store bounding boxes
            if image_filename not in bounding_boxes:
                bounding_boxes[image_filename] = []

            # Extract nodes for bounding boxes
            for node in root.findall('Node'):
                left = int(node.find('Left').text)
                top = int(node.find('Top').text)
                width = int(node.find('Width').text)
                height = int(node.find('Height').text)

                # Append the bounding box data
                bounding_boxes[image_filename].append({
                    "origin": {"x": left, "y": top},
                    "width": width,
                    "height": height
                })

    return bounding_boxes

# Example usage
xml_directory = "./data/data/muscima_pp_raw/v2.0/data/annotations"  # Path to directory with XML files
output_path = "./data/data/muscima_pp_raw/v2.0/data/bounding_boxes.pkl"
bounding_boxes = parse_bounding_boxes_from_xml(xml_directory)
print(bounding_boxes['CVC-MUSCIMA_W-28_N-05_D-ideal.png'])
# Write the dictionary to the pkl file
with open(output_path, 'wb') as file:
    pkl.dump(bounding_boxes, file)

print(f"Bounding boxes saved to: {output_path}")
