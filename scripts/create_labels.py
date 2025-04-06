import os
import xml.etree.ElementTree as ET

def extract_annotations_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    depth = int(root.find('size/depth').text)
    image_shape = width, height, depth

    labels_and_bboxes = []


    for obj in root.findall('object'):

        label = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)


        labels_and_bboxes.append((label, (xmin, ymin, xmax, ymax)))

    return image_name, image_shape, labels_and_bboxes

def create_labels(xml_dir, labels_dir):

    os.makedirs(labels_dir, exist_ok=True)
    print(os.listdir(xml_dir))
    annotations = [file for file in os.listdir(xml_dir) if file.lower().endswith('.xml')]
    count = 0
    ignored = 0

    for xml_file in annotations:
        image_name, image_shape, labels_and_bboxes = extract_annotations_from_xml(os.path.join(xml_dir, xml_file))
        txt_file = os.path.join(labels_dir, xml_file.replace('.xml', '.txt'))
        file_corrupt = False

        with open(txt_file, 'w') as f:
            for label, bbox in labels_and_bboxes:
                label = 1 if label == 'With Helmet' else 0
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                x_center /= image_shape[0]
                y_center /= image_shape[1]
                width /= image_shape[0]
                height /= image_shape[1]

                if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                    file_corrupt = True
                    break

                f.write(f"{label} {x_center} {y_center} {width} {height}\n")

        if file_corrupt:
            ignored += 1
            f.close()
            os.remove(txt_file)
            continue

        print(f"\rImage: {image_name}     ", end='', flush=True)
        count += 1

    print(f"\n>> {count} labels created | {ignored} images ignored")


 
