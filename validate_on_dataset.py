"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import sys
from ml_serving.drivers import driver
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

from app.tools import dataset
from app.tools import helpers


def main(args):
    # Get the paths for the corresponding images
    image_size = args.image_size
    driver_name = 'openvino'
    if os.path.isdir(args.model) and os.path.exists(os.path.join(args.model, 'saved_model.pb')):
        driver_name = 'tensorflow'
        image_size = 112

    img_paths, actual_issame = load_dataset(args.lfw_dir)
    drv = driver.load_driver(driver_name)
    serving = drv()
    serving.load_model(
        args.model,
        inputs='input:0,phase_train:0',
        outputs='embeddings:0',
        device='CPU',
        flexible_batch_size=True,
    )

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on dataset images')

    # Enqueue one epoch of image paths and labels
    nrof_images = len(img_paths)

    embedding_size = list(serving.outputs.values())[0][-1]
    nrof_batches = int(np.ceil(float(nrof_images) / args.lfw_batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))

    # TODO(nmakhotkin): cache embeddings by image paths (because image pairs
    #  are duplicated and no need to do inference on them)
    for i in range(nrof_batches):
        start_index = i * args.lfw_batch_size
        end_index = min((i + 1) * args.lfw_batch_size, nrof_images)
        paths_batch = img_paths[start_index:end_index]
        probe_imgs = dataset.load_data(paths_batch, image_size, fixed_normalization=True)
        emb = _predict(serving, probe_imgs)
        emb_array[start_index:end_index, :] = emb
        if i % 5 == 4:
            print('{}/{}'.format(i + 1, nrof_batches))
            sys.stdout.flush()
    print('')
    embeddings = emb_array

    tpr, fpr, accuracy, val, val_std, far = helpers.evaluate(
        embeddings, actual_issame, nrof_folds=args.lfw_nrof_folds,
        distance_metric=args.distance_metric, subtract_mean=args.subtract_mean
    )

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)


def load_dataset(dataset_dir):
    ds = dataset.get_dataset(dataset_dir)
    size = 0
    for cls in ds:
        size += len(cls.image_paths)

    imgs_original = [''] * size
    imgs = [''] * (size * 4)
    imgs_cls = [''] * size
    issame = np.zeros([size * 2], dtype=np.bool)

    i = 0
    for cls in ds:
        for j, path in enumerate(cls.image_paths):
            img_path = path
            imgs_original[i] = img_path
            imgs[i + i] = img_path
            imgs[i + i + size * 2] = img_path + ':flip'
            imgs_cls[i] = cls.name
            i += 1

    # Expand dataset with random pairs
    print('Generating pairs for dataset...')
    for i in range(len(imgs_original) * 2):
        if i % 2 == 0:
            target = lambda x: not x
        else:
            target = lambda x: x
        j = np.random.randint(0, len(imgs_original) * 2)
        if target(imgs_cls[i % size] != imgs_cls[j % size]):
            shift = 1
            j1 = j - shift if j > 0 else 0
            j2 = j + shift if j < len(imgs_original) * 2 else len(imgs_original) - 1
            while target(imgs_cls[i % size] != imgs_cls[j1 % size]) and target(imgs_cls[i % size] != imgs_cls[j2 % size]):
                j1 = j - shift if j1 > 0 else 0
                j2 = j + shift if j2 < len(imgs_original) * 2 else len(imgs_original) - 1
                shift += 1
            if target(imgs_cls[i % size] == imgs_cls[j1 % size]):
                j = j1
            else:
                j = j2

        if j >= size:
            imgs[i + i + 1] = imgs_original[j % size] + ':flip'
        else:
            imgs[i + i + 1] = imgs_original[j]

        issame[i] = imgs_cls[i % size] == imgs_cls[j % size]

    print('Done.')
    return imgs, issame


def _predict(serving, imgs):
    if serving.driver_name == 'tensorflow':
        input_sizes = list(serving.inputs.values())[0]
        if input_sizes[1] == 112:
            # Arcface
            input_name = list(serving.inputs.keys())[0]
            feed_dict = {input_name: imgs}
        else:
            feed_dict = {'input:0': imgs, 'phase_train:0': False}
    elif serving.driver_name == 'openvino':
        input_name = list(serving.inputs.keys())[0]
        # Transpose image for channel first format
        imgs = imgs.transpose([0, 3, 1, 2])
        feed_dict = {input_name: imgs}
    else:
        raise RuntimeError('Driver %s currently not supported' % serving.driver_name)
    outputs = serving.predict(feed_dict)
    return list(outputs.values())[0]


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('lfw_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
                        help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--use_flipped_images',
                        help='Concatenates embeddings for the image and its horizontally flipped counterpart.',
                        action='store_true')
    parser.add_argument('--subtract_mean',
                        help='Subtract feature mean before calculating distance.', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
