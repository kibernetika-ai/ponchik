from threading import Thread

import cv2
import numpy as np

from svod_rcgn.tools import images
from svod_rcgn.tools.print import print_fun


def video_args(detector, listener, args):
    return Video(
        detector,
        listener=listener,
        video_source=args.video_source,
        video_async=args.video_async,
    )


class Video:
    def __init__(self, detector, listener=None, video_source=None, video_async=False):
        self.detector = detector
        self.video_source = video_source
        self.frame = None
        self.processed = None
        self.vs = None
        self.listener = listener
        self.video_async = video_async
        self.pipeline = None

    def start(self):
        self.detector.init()
        if self.vs is None:
            self.init_video()
        if self.video_async:
            frame_thread = Thread(target=self.process_frame, daemon=True)
            frame_thread.start()
        if self.listener is not None:
            listen_thread = Thread(target=self.listen, daemon=True)
            listen_thread.start()
        try:
            while True:
                # Capture frame-by-frame
                self.get_frame()
                if self.frame is None:
                    continue
                if self.video_async:
                    self.detector.add_overlays(self.frame, self.processed)
                else:
                    self.processed = self.detector.process_frame(self.frame)
                cv2.imshow('Video', self.frame)
                key = cv2.waitKey(1)
                # Wait 'q' or Esc or 'q' in russian layout
                if key in [ord('q'), 202, 27]:
                    break
        except (KeyboardInterrupt, SystemExit) as e:
            print_fun('Caught %s: %s' % (e.__class__.__name__, e))
        finally:
            if self.pipeline is not None:
                self.pipeline.stop()

    def init_video(self):
        if self.video_source is None:
            self.vs = cv2.VideoCapture(0)
        elif self.video_source=="realscense":
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8,1)
            self.pipeline.start(config)
        else:
            self.vs = cv2.VideoCapture(self.video_source)

    def get_frame(self):
        if self.pipeline is None:
            new_frame = self.vs.read()
        else:
            frames = self.pipeline.wait_for_frames()
            new_frame = frames.get_color_frame()
            new_frame = np.asanyarray(new_frame.get_data())
        if isinstance(new_frame, tuple):
            new_frame = new_frame[1]
        if new_frame is None:
            print_fun("frame is None. Possibly camera or display does not work")
            return None
        #if new_frame.shape[0] > 480:
        #    new_frame = images.image_resize(new_frame, height=480)
        self.frame = new_frame
        return self.frame

    def process_frame(self):
        while True:
            if self.frame is not None:
                self.processed = self.detector.process_frame(self.frame, overlays=False)

    def listen(self):
        while True:
            err = None
            command, data = self.listener.listen()
            if command == 'reload_classifiers':
                print_fun("reload classifiers")
                self.detector.load_classifiers()
            elif command == 'debug':
                deb = bool(data)
                print_fun("set debug " + ("on" if deb else "off"))
                self.detector.debug = deb
            elif command == 'test':
                print_fun("get test data:")
                print_fun(data)
            else:
                err = ValueError('unknown command %s' % command)
            self.listener.result(err)


def add_video_args(parser):
    parser.add_argument(
        '--video_source',
        help='Video source. If not set, current webcam is used (value 0).',
        default=None,
    )
    parser.add_argument(
        '--video_async',
        help='Asynchronous video (each frame does not wait for calculating boxes and labels).',
        action='store_true',
    )
