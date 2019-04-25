import os

import cv2


def image_args(detector, args):
    return Image(
        detector,
        image_source=args.image_source,
    )


class Image:
    def __init__(self, detector, image_source=None):
        self.detector = detector
        self.image_source = image_source

    def process(self):
        if not os.path.isfile(self.image_source):
            raise ValueError('Image file "%s" does not exist' % self.image_source)
        self.detector.init()
        img = cv2.imread(self.image_source)
        self.detector.process_frame(img, overlays=True)
        cv2.imshow('Image', img)
        cv2.waitKey()


def add_image_args(parser):
    parser.add_argument(
        '--image_source',
        help='Source image file',
        required=True,
    )
