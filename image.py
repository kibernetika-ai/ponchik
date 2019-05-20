import argparse

from app.recognize import add_common_args
from app.recognize.detector import add_detector_args, detector_args
from app.recognize.image import add_image_args, image_args
from app.tools.bg_remove import add_bg_remove_args


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_bg_remove_args(parser)
    add_detector_args(parser)
    add_image_args(parser)
    args = parser.parse_args()
    detector = detector_args(args)
    image = image_args(detector, args)
    image.process()


if __name__ == '__main__':
    main()
