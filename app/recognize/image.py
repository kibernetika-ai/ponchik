import os
from app.recognize import detector

import cv2


def image_args(dtct: detector.Detector, args):
    return Image(dtct, image_source=args.image_source)


class Image:
    def __init__(self, dtct: detector.Detector, image_source=None):
        self.detector: detector.Detector = dtct
        self.image_source = image_source

    def process(self):
        if not os.path.isfile(self.image_source):
            raise ValueError('Image file "%s" does not exist' % self.image_source)
        self.detector.init()
        img = cv2.imread(self.image_source)
        self.detector.process_frame(img)
        cv2.imshow('Image', img)
        cv2.waitKey()


def add_image_args(parser):
    parser.add_argument(
        '--image_source',
        help='Source image file',
        required=True,
    )
