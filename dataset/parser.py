import cv2
import numpy as np


def get_data(dataset_path, cat=None):
    found_bg = False

    total_images = {}
    classes_count = {}
    class_mapping = {}

    with open(dataset_path, 'r') as f:

        print('Parsing annotation files from a text file')

        for line in f:
            stripped_line = line.strip().split(',')
            (image_path, xmin, ymin, xmax, ymax, label) = stripped_line

            if label not in classes_count:
                classes_count[label] = 1
            else:
                classes_count[label] += 1

            if label not in class_mapping:
                if label == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region')
                    found_bg = True
                class_mapping[label] = len(class_mapping)

            if image_path not in total_images:
                total_images[image_path] = {}

                image = cv2.imread(image_path)
                (width, height) = image.shape[:2]
                total_images[image_path]['filepath'] = image_path
                total_images[image_path]['width'] = height
                total_images[image_path]['height'] = width
                total_images[image_path]['bboxes'] = []

                if np.random.randint(0, 6) > 0:
                    total_images[image_path]['imageset'] = 'trainval'
                else:
                    total_images[image_path]['imageset'] = 'test'

            total_images[image_path]['bboxes'].append({'class': label, 'x1': int(xmin), 'x2': int(xmax), 'y1': int(ymin), 'y2': int(ymax)})

        all_data = []
        for key in total_images:
            all_data.append(total_images[key])

        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping
