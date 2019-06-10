import json
import os

import cv2
import numpy as np
from PIL import Image

from app.tools import images

META_FILENAME = 'meta.json'
PREWHITEN = True

def get_dataset(path, min_img_count=None, limit=None):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = []
    for path in os.listdir(path_exp):
        # Exclude hidden directories
        if os.path.isdir(os.path.join(path_exp, path)) and not path.startswith('.'):
            classes.append(path)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths, meta = get_image_paths(facedir)
        if min_img_count is None or len(image_paths) >= min_img_count:
            dataset.append(ImageClass(class_name, image_paths, meta=meta))
        if limit is not None and len(dataset) >= limit:
            break

    return dataset


def get_image_paths(facedir, limit=None):
    image_paths = []
    meta = None
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [
            os.path.join(facedir, img)
            for img in images
            if not img.startswith('.') and os.path.basename(img) != META_FILENAME
        ]
        m_file = os.path.join(os.path.expanduser(facedir), META_FILENAME)
        if os.path.isfile(m_file):
            with open(m_file) as mf:
                meta = json.load(mf)
    if limit:
        return image_paths[:limit]
    return image_paths, meta


def split_to_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        images_count = 0
        for image_path in dataset[i].image_paths:
            image_paths_flat.append(image_path)
            images_count += 1
        labels_flat += [i] * images_count
    return image_paths_flat, labels_flat


def get_meta(dataset):
    meta = {}
    for i in range(len(dataset)):
        if dataset[i].meta is not None:
            meta[dataset[i].name] = dataset[i].meta
        # for image_path in dataset[i].image_paths:
        #     if os.path.basename(image_path) == META_FILENAME:
        #         with open(image_path) as mf:
        #             meta[dataset[i].name] = json.load(mf)
    return meta


def load_data(image_paths, image_size, fixed_normalization=False):
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
        if fixed_normalization:
            img = images.fixed_normalize(img)
        else:
            img = images.prewhiten(img)
        imgs[i, :, :, :] = img
    return imgs


class ImageClass:
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths, meta=None):
        self.name = name
        self.image_paths = image_paths
        self.meta = meta

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)
