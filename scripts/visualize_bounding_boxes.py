import os
import cv2
from matplotlib import pyplot as plt


def visualize_bounding_boxes(img_file, labels_and_bboxes):
    image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

    for label, bbox in labels_and_bboxes:
        xmin, ymin, xmax, ymax = bbox
        rgb_color = (0, 255, 0) if label == 'With Helmet' else (255, 0, 0)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), rgb_color, 2)
        cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, rgb_color, 1)

    plt.axis("off")
    plt.title(os.path.split(img_file)[-1], y=-0.1)
    plt.imshow(image)