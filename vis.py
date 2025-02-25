
import os
import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat

image_dir = ""
label_dir = ""
output_dir = ""

os.makedirs(output_dir, exist_ok=True)

for image_name in os.listdir(image_dir):

    if not image_name.endswith(".png"):
        continue

    label_name = image_name.replace(".png", ".mat")
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, label_name)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        continue

    label_data = loadmat(label_path)
    label = label_data['inst_map']


    unique_values = np.unique(label)
    unique_values = unique_values[unique_values != 0]

    for value in unique_values:
        instance_mask = np.uint8(label == value) * 255
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(image, [contour], -1, (0, 255, 255), thickness=2)

    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, image)
