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
import os
import pickle
import shutil

import cv2
import numpy as np
from ml_serving.drivers import driver

from svod_rcgn.recognize import defaults
from svod_rcgn.tools import bg_remove, images, downloader, dataset

DEFAULT_INPUT_DIR = "./data/faces"


def add_aligner_args(parser):
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Directory with source images.',
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        '--clear_input_dir',
        help='Clear input dir before extracting downloaded archive.',
        action="store_true",
    )
    parser.add_argument(
        '--complementary_align',
        help='Existing aligned images in aligned dir supplements with new ones from input dir.',
        action='store_true',
    )
    parser.add_argument(
        '--download',
        type=str,
        help='URL to .tar or .tar.gz dataset with source faces.',
        default=None,
    )


def aligner_args(args):
    return Aligner(
        input_dir=args.input_dir,
        clear_input_dir=args.clear_input_dir,
        download=args.download,
        aligned_dir=args.aligned_dir,
        complementary_align=args.complementary_align,
        min_face_size=args.min_face_size,
        image_size=args.image_size,
        margin=args.margin,
        face_detection_path=args.face_detection_path,
        bg_remove_path=args.bg_remove_path,
        device=args.device,
    )


class Aligner:
    def __init__(
            self,
            input_dir=DEFAULT_INPUT_DIR,
            clear_input_dir=False,
            download=None,
            aligned_dir=defaults.ALIGNED_DIR,
            complementary_align=False,
            min_face_size=defaults.MIN_FACE_SIZE,
            image_size=defaults.IMAGE_SIZE,
            margin=defaults.IMAGE_MARGIN,
            face_detection_path=defaults.FACE_DETECTION_PATH,
            bg_remove_path=bg_remove.DEFAULT_BG_REMOVE_DIR,
            device=defaults.DEVICE,
    ):
        self.input_dir = input_dir
        self.aligned_dir = aligned_dir
        self.complementary_align = complementary_align
        self.min_face_size = min_face_size
        self.image_size = image_size
        self.margin = margin
        self.face_detection_path = face_detection_path
        self.bg_remove_path = bg_remove_path
        self.device = device
        if clear_input_dir:
            shutil.rmtree(self.input_dir, ignore_errors=True)
        if download is not None:
            err = downloader.Downloader(download, destination=self.input_dir).extract()
            if err is not None:
                raise RuntimeError(err)

    def align(self):

        print('Align images to %s' % self.aligned_dir)

        aligned_dir = os.path.expanduser(self.aligned_dir)
        bounding_boxes_filename = os.path.join(aligned_dir, 'bounding_boxes.txt')
        align_filename = os.path.join(aligned_dir, 'align.pkl')

        align_data_args = {
            "min_face_size": self.min_face_size,
            "image_size": self.image_size,
            "margin": self.margin,
        }

        align_data = {}
        if os.path.isfile(align_filename):
            print("Check previous align data")
            with open(align_filename, 'rb') as infile:
                (align_data_args_loaded, align_data_loaded) = pickle.load(infile)
                if align_data_args == align_data_args_loaded:
                    print("Loaded data about %d aligned classes" % len(align_data_loaded))
                    align_data = align_data_loaded

                else:
                    print("Previous align data is for another arguments, deleting existing data")
                    shutil.rmtree(self.aligned_dir, ignore_errors=True)

        if not os.path.isdir(aligned_dir):
            print("Creating output dir")
            os.makedirs(aligned_dir)

        if not self.complementary_align:
            print("Clearing existing aligned images")
            in_dir = os.listdir(aligned_dir)
            for i in in_dir:
                fi = os.path.join(aligned_dir, i)
                if os.path.isdir(fi):
                    shutil.rmtree(fi, ignore_errors=True)

        # Store some git revision info in a text file in the log directory
        src_path, _ = os.path.split(os.path.realpath(__file__))
        # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
        loaded_dataset = dataset.get_dataset(self.input_dir)

        print('Creating networks and loading parameters')

        # Load driver
        drv = driver.load_driver("openvino")
        # Instantinate driver
        serving = drv()
        serving.load_model(
            self.face_detection_path,
            device=self.device,
            flexible_batch_size=True,
        )

        bg_rm_drv = bg_remove.get_driver(self.bg_remove_path)

        input_name = list(serving.inputs.keys())[0]
        output_name = list(serving.outputs.keys())[0]

        threshold = 0.5

        min_face_area = self.min_face_size ** 2

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_images_cached = 0
            nrof_successfully_aligned = 0
            for cls in loaded_dataset:
                output_class_dir = os.path.join(aligned_dir, cls.name)
                output_class_dir_created = False
                if cls.name in align_data:
                    align_data_class = align_data[cls.name]
                else:
                    align_data_class = {}
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    if not os.path.exists(output_filename):
                        try:
                            img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
                        except Exception as e:
                            error_message = '{}: {}'.format(image_path, e)
                            print('ERROR: %s' % error_message)
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
                                        # print("%s - cached" % image_path)
                                        nrof_images_cached += 1
                                        continue
                        align_data_class[image_path] = {'hash': hashlib.sha1(img.tostring()).hexdigest()}
                        print(image_path)

                        if len(img.shape) <= 2:
                            print('WARNING: Unable to align "%s", shape %s' % (image_path, img.shape))
                            text_file.write('%s\n' % output_filename)
                            continue

                        if bg_rm_drv is not None:
                            img = bg_rm_drv.apply_mask(img)

                        serving_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
                        serving_img = np.transpose(serving_img, [2, 0, 1]).reshape([1, 3, 300, 300])
                        raw = serving.predict({input_name: serving_img})[output_name].reshape([-1, 7])
                        # 7 values:
                        # class_id, label, confidence, x_min, y_min, x_max, y_max
                        # Select boxes where confidence > factor
                        bboxes_raw = raw[raw[:, 2] > threshold]
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
                            print('WARNING: Unable to align "%s", n_faces=%s' % (image_path, len(area)))
                            text_file.write('%s\n' % output_filename)
                            continue

                        num = np.argmax(area)
                        if area[num] < min_face_area:
                            print(
                                'WARNING: Face found but too small - about {}px '
                                'width against required minimum of {}px. Try'
                                ' adjust parameter --min-face-size'.format(
                                    int(np.sqrt(area[num])), self.min_face_size
                                )
                            )
                            continue

                        bounding_boxes = np.stack([bbs[num]])

                        imgs = images.get_images(
                            img,
                            bounding_boxes,
                            face_crop_size=self.image_size,
                            face_crop_margin=self.margin,
                            do_prewhiten=False,
                        )
                        align_data_class[image_path]['aligned'] = []
                        for i, cropped in enumerate(imgs):
                            nrof_successfully_aligned += 1
                            bb = bounding_boxes[i]
                            filename_base, file_extension = os.path.splitext(output_filename)
                            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                            if not output_class_dir_created:
                                output_class_dir_created = True
                                if not os.path.exists(output_class_dir):
                                    os.makedirs(output_class_dir)
                            cv2.imwrite(output_filename_n, cropped)
                            align_data_class[image_path]['aligned'].append(output_filename_n)

                align_data[cls.name] = align_data_class

        with open(align_filename, 'wb') as align_file:
            pickle.dump((align_data_args, align_data), align_file, protocol=2)

        print('Total number of images: %d' % nrof_images_total)
        if nrof_images_cached > 0:
            print('Number of cached images: %d' % nrof_images_cached)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
