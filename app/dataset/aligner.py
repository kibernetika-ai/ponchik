"""Performs face alignment and stores face thumbnails in the output directory."""
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

import hashlib
import json
import os
import pickle
import shutil

import cv2
import numpy as np
from ml_serving.drivers import driver

from app.recognize import defaults
from app.tools import bg_remove
from app.tools import images
from app.tools import downloader
from app.tools import dataset
from app.tools import utils


DEFAULT_INPUT_DIR = "./data/faces"


class Aligner:
    def __init__(
            self,
            input_dir=DEFAULT_INPUT_DIR,
            clarified=False,
            clear_input_dir=False,
            download=None,
            aligned_dir=defaults.ALIGNED_DIR,
            complementary_align=False,
            min_face_size=defaults.MIN_FACE_SIZE,
            image_size=defaults.IMAGE_SIZE,
            margin=defaults.IMAGE_MARGIN,
            face_detection_path=defaults.FACE_DETECTION_PATH,
            bg_remove_path=None,
            device=defaults.DEVICE,
    ):
        self.input_dir = input_dir
        self.clarified = clarified
        self.aligned_dir = aligned_dir
        self.complementary_align = complementary_align or clarified
        self.min_face_size = min_face_size
        self.image_size = image_size
        self.margin = margin
        self.face_detection_path = face_detection_path
        self.bg_remove_path = None if clarified else bg_remove_path
        self.device = device
        if clear_input_dir:
            shutil.rmtree(self.input_dir, ignore_errors=True)
        if download is not None:
            err = downloader.Downloader(download, destination=self.input_dir).extract()
            if err is not None:
                raise RuntimeError(err)
        self.serving: driver.ServingDriver = None
        self.threshold = 0.5
        self.min_face_area = self.min_face_size ** 2

    def align(self, images_limit=None):

        if self.complementary_align:
            utils.print_fun('Complementary align %simages to %s' % ("clarified " if self.clarified else "", self.aligned_dir))
        else:
            utils.print_fun('Align images to %s' % self.aligned_dir)

        aligned_dir = os.path.expanduser(self.aligned_dir)
        bounding_boxes_filename = os.path.join(aligned_dir, 'bounding_boxes.txt')
        align_filename = os.path.join(aligned_dir, 'align.pkl')

        align_data_args = {
            "min_face_size": self.min_face_size,
            "image_size": self.image_size,
            "margin": self.margin,
            # used for dataset alignment and do not used for clarified alignment
            # "bg_remove_path": self.bg_remove_path,
        }

        align_data = {}
        if os.path.isfile(align_filename):
            utils.print_fun("Check previous align data")
            with open(align_filename, 'rb') as infile:
                (align_data_args_loaded, align_data_loaded) = pickle.load(infile)
                if align_data_args == align_data_args_loaded:
                    utils.print_fun("Loaded data about %d aligned classes" % len(align_data_loaded))
                    align_data = align_data_loaded
                else:
                    utils.print_fun("Previous align data is for another arguments, deleting existing data")
                    shutil.rmtree(aligned_dir, ignore_errors=True)

        if not os.path.isdir(aligned_dir):
            utils.print_fun("Creating output dir")
            os.makedirs(aligned_dir)

        # Store some git revision info in a text file in the log directory
        src_path, _ = os.path.split(os.path.realpath(__file__))
        # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
        loaded_dataset = dataset.get_dataset(self.input_dir)
        loaded_dataset_meta = dataset.get_meta(loaded_dataset)

        utils.print_fun('Creating networks and loading parameters')

        # Load driver
        self._load_driver()

        bg_rm_drv = bg_remove.get_driver(self.bg_remove_path)

        bounding_boxes_contents = ""

        # clear not actual previous aligned stored data
        if not self.complementary_align and len(align_data) > 0:
            stored_classes = []
            for cls in loaded_dataset:
                stored_classes.append(cls.name)
            for adcl in list(align_data):
                if adcl not in stored_classes:
                    del align_data[adcl]

        nrof_images_total = 0
        nrof_images_skipped = 0
        nrof_images_cached = 0
        nrof_successfully_aligned = 0
        nrof_has_meta = 0
        for cls in loaded_dataset:
            output_class_dir = os.path.join(aligned_dir, cls.name)
            output_class_dir_created = False
            # meta_file = None
            aligned_class_images = []
            if cls.name in align_data:
                align_data_class = align_data[cls.name]
            else:
                align_data_class = {}
            for image_path in cls.image_paths:
                # if os.path.basename(image_path) == dataset.META_FILENAME:
                #     meta_file = image_path
                #     continue
                nrof_images_total += 1
                nrof_images_skipped += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                if not os.path.exists(output_filename):
                    try:
                        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
                    except Exception as e:
                        error_message = '{}: {}'.format(image_path, e)
                        utils.print_fun('ERROR: %s' % error_message)
                        continue

                    img_hash = hashlib.sha1(img.tostring()).hexdigest()
                    if image_path in align_data_class:
                        if 'hash' in align_data_class[image_path]:
                            if align_data_class[image_path]['hash'] == img_hash:
                                all_aligned_exists = True
                                if 'aligned' in align_data_class[image_path]:
                                    for a in align_data_class[image_path]['aligned']:
                                        if not os.path.isfile(a):
                                            all_aligned_exists = False
                                            break
                                if all_aligned_exists:
                                    if 'aligned' in align_data_class[image_path]:
                                        aligned_class_images.extend(
                                            list(align_data_class[image_path]['aligned'].keys()))
                                        for _, a in enumerate(align_data_class[image_path]['aligned']):
                                            b = align_data_class[image_path]['aligned'][a]
                                            bounding_boxes_contents += \
                                                '%s %d %d %d %d cached\n' % (a, b[0], b[1], b[2], b[3])
                                            nrof_images_skipped -= 1
                                    else:
                                        bounding_boxes_contents += \
                                            '%s ERROR no aligned cached\n' % image_path

                                    nrof_images_cached += 1
                                    continue

                    align_data_class[image_path] = {'hash': hashlib.sha1(img.tostring()).hexdigest()}
                    utils.print_fun(image_path)

                    if len(img.shape) <= 2:
                        utils.print_fun('WARNING: Unable to align "%s", shape %s' % (image_path, img.shape))
                        bounding_boxes_contents += '%s ERROR invalid shape\n' % image_path
                        continue

                    if self.clarified:

                        # get clarified image as is and make one with aligned size
                        bounding_boxes = np.stack([[0, 0, img.shape[1], img.shape[0]]])
                        face_crop_margin = 0

                    else:

                        # detect faces previously with bg_remove if set, if not found, try to detect w/o bg_remove
                        if bg_rm_drv is not None:
                            img = bg_rm_drv.apply_mask(img)

                        bounding_boxes = None
                        if bg_rm_drv is not None:
                            img_masked = bg_rm_drv.apply_mask(img)
                            bounding_boxes = self._get_boxes(image_path, img_masked)
                            if bounding_boxes is None:
                                utils.print_fun('WARNING: no faces on image with removed bg, trying without bg removing')

                        if bounding_boxes is None or bg_rm_drv is not None:
                            bounding_boxes = self._get_boxes(image_path, img)

                        if bounding_boxes is None:
                            bounding_boxes_contents += '%s ERROR no faces detected\n' % image_path
                            continue

                        face_crop_margin = self.margin

                    imgs = images.get_images(
                        img,
                        bounding_boxes,
                        face_crop_size=self.image_size,
                        face_crop_margin=face_crop_margin,
                        normalization=None,
                    )

                    align_data_class[image_path]['aligned'] = {}
                    for i, cropped in enumerate(imgs):
                        nrof_successfully_aligned += 1
                        bb = bounding_boxes[i]
                        filename_base, file_extension = os.path.splitext(output_filename)
                        output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                        bounding_boxes_contents += '%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3])
                        if not output_class_dir_created:
                            output_class_dir_created = True
                            if not os.path.exists(output_class_dir):
                                os.makedirs(output_class_dir)
                        cv2.imwrite(output_filename_n, cropped)
                        align_data_class[image_path]['aligned'][output_filename_n] = (bb[0], bb[1], bb[2], bb[3])

                        if images_limit and nrof_successfully_aligned >= images_limit:
                            break

                    aligned_class_images.extend(list(align_data_class[image_path]['aligned'].keys()))

                nrof_images_skipped -= 1

                if images_limit and nrof_successfully_aligned >= images_limit:
                    break

            if os.path.isdir(output_class_dir):
                cls_ = cls.name.replace(' ', '_')
                if cls_ in loaded_dataset_meta:
                    with open(os.path.join(output_class_dir, dataset.META_FILENAME), 'w') as mf:
                        json.dump(loaded_dataset_meta[cls_], mf)

            # clear not existing in input already exists aligned class images
            if not self.complementary_align:
                if os.path.isdir(output_class_dir):
                    for f in os.listdir(output_class_dir):
                        if f == dataset.META_FILENAME:
                            continue
                        fp = os.path.join(output_class_dir, f)
                        if os.path.isfile(fp) and fp not in aligned_class_images:
                            os.remove(fp)

            align_data[cls.name] = align_data_class

            if images_limit and images_limit <= nrof_successfully_aligned:
                utils.print_fun("Limit for aligned images %d is reached" % images_limit)
                break

        dataset.get_meta(loaded_dataset)

        # clear not existing in input already exists aligned classes (dirs)
        if not self.complementary_align:
            for d in os.listdir(aligned_dir):
                dd = os.path.join(aligned_dir, d)
                if os.path.isdir(dd) and d not in align_data:
                    shutil.rmtree(dd, ignore_errors=True)

        with open(bounding_boxes_filename, "w") as text_file:
            text_file.write(bounding_boxes_contents)

        with open(align_filename, 'wb') as align_file:
            pickle.dump((align_data_args, align_data), align_file, protocol=2)

        utils.print_fun('Total number of images: %d' % nrof_images_total)
        if nrof_images_cached > 0:
            utils.print_fun('Number of cached images: %d' % nrof_images_cached)
        if nrof_images_skipped > 0:
            utils.print_fun('Number of skipped images: %d' % nrof_images_skipped)
        if nrof_has_meta > 0:
            utils.print_fun('Number of classes with meta: %d' % nrof_has_meta)
        utils.print_fun('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    def _load_driver(self):
        if self.serving is None:
            drv = driver.load_driver("openvino")
            # Instantinate driver
            self.serving = drv()
            self.serving.load_model(
                self.face_detection_path,
                # device=self.device,
                flexible_batch_size=True,
            )
            self.input_name = list(self.serving.inputs.keys())[0]
            self.input_size = tuple(list(self.serving.inputs.values())[0][:-3:-1])
            self.output_name = list(self.serving.outputs.keys())[0]

    def _get_boxes(self, image_path, img):
        serving_img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        serving_img = np.transpose(serving_img, [2, 0, 1]).reshape([1, 3, *self.input_size[::-1]])
        raw = self.serving.predict({self.input_name: serving_img})[self.output_name].reshape([-1, 7])
        # 7 values:
        # class_id, label, confidence, x_min, y_min, x_max, y_max
        # Select boxes where confidence > factor
        bboxes_raw = raw[raw[:, 2] > self.threshold]
        bboxes_raw[:, 3] = bboxes_raw[:, 3] * img.shape[1]
        bboxes_raw[:, 5] = bboxes_raw[:, 5] * img.shape[1]
        bboxes_raw[:, 4] = bboxes_raw[:, 4] * img.shape[0]
        bboxes_raw[:, 6] = bboxes_raw[:, 6] * img.shape[0]

        bounding_boxes = np.zeros([len(bboxes_raw), 5])

        bounding_boxes[:, 0:4] = bboxes_raw[:, 3:7]
        bounding_boxes[:, 4] = bboxes_raw[:, 2]

        # Get the biggest box: find the box with largest square:
        # (y1 - y0) * (x1 - x0) - size of box.
        bbs = bounding_boxes
        area = (bbs[:, 3] - bbs[:, 1]) * (bbs[:, 2] - bbs[:, 0])

        if len(area) < 1:
            utils.print_fun('WARNING: Unable to align "%s", n_faces=%s' % (image_path, len(area)))
            return None

        num = np.argmax(area)
        if area[num] < self.min_face_area:
            utils.print_fun(
                'WARNING: Unable to align "{}", face found but too small - about {}px '
                'width against required minimum of {}px. Try adjust parameter --min-face-size'.format(
                    image_path, int(np.sqrt(area[num])), self.min_face_size
                )
            )
            return None

        bounding_boxes = np.stack([bbs[num]])
        return bounding_boxes

