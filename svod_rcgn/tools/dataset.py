import os

import cv2
import numpy as np
from PIL import Image

from svod_rcgn.tools import images

META_FILENAME = 'meta.json'


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = []
    for path in os.listdir(path_exp):
        # Exclude hidden directories
        if (os.path.isdir(os.path.join(path_exp, path))
                and not path.startswith('.')):
            classes.append(path)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir, limit=None):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images if not img.startswith('.')]
    if limit:
        return image_paths[:limit]
    return image_paths


def split_to_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        for image_path in dataset[i].image_paths:
            if os.path.basename(image_path) != META_FILENAME:
                image_paths_flat.append(image_path)
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def load_data(image_paths, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    imgs = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.ndim == 2:
            img = images.to_rgb(img)
        if len(img.shape) >= 3 and img.shape[2] > 3:
            # RGBA, convert to RGB
            img = np.array(Image.fromarray(img).convert('RGB'))
        if do_prewhiten:
            img = images.prewhiten(img)
        imgs[i, :, :, :] = img
    return imgs


class ImageClass:
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)
