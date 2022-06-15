import cv2
import os
import glob

splits = ["train", "valid"]
images_dir = "images"


def transform_images(image_shape=256):
    image_dims = (image_shape, image_shape)

    for each_split in splits:
        split_image_paths = os.path.join(images_dir, each_split)
        images = glob.glob(split_image_paths + "/*")

        for each in images:
            filename = each.split(".")[0].split("/")[-1]
            img = cv2.imread(each, 1)
            resized_image = cv2.resize(img, image_dims, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(f'{filename}.jpg', resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


transform_images()
