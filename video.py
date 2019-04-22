import argparse

from svod_rcgn.control.listener import add_listener_args, listener_args
from svod_rcgn.recognize.args import add_common_args
from svod_rcgn.recognize.detector import add_detector_args, detector_args
from svod_rcgn.recognize.video import video_args, add_video_args
from svod_rcgn.tools.bg_remove import add_bg_remove_args


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_bg_remove_args(parser)
    add_detector_args(parser)
    add_listener_args(parser)
    add_video_args(parser)
    args = parser.parse_args()

    detector = detector_args(args)
    listener = listener_args(args)
    camera = video_args(detector, listener, args)

    camera.start()


if __name__ == '__main__':
    main()

