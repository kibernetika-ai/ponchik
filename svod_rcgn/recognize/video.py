import time
from threading import Thread

import cv2
import numpy as np

from svod_rcgn.tools import images
from svod_rcgn.tools.print import print_fun
from svod_rcgn.recognize.video_notify import InVideoDetected
from svod_rcgn.notify import notify


def video_args(detector, listener, args):
    return Video(
        detector,
        listener=listener,
        video_source=args.video_source,
        video_async=args.video_async,
        video_max_width=args.video_max_width,
        video_max_height=args.video_max_height,
    )


class Video:
    def __init__(self, detector,
                 listener=None, video_source=None, video_async=False,
                 video_max_width=None, video_max_height=None):
        self.detector = detector
        self.video_source = video_source
        self.frame = None
        self.processed = None
        self.vs = None
        self.listener = listener
        self.video_async = video_async
        self.pipeline = None
        self.video_max_width = video_max_width
        self.video_max_height = video_max_height
        self.faces_detected = {}
        self.notifies_queqe = []

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
        notify_thread = Thread(target=self.notify, daemon=True)
        notify_thread.start()
        try:
            while True:
                # Capture frame-by-frame
                self.get_frame()
                if self.frame is None:
                    continue
                frame_to_show = self.frame.copy()
                if self.video_async:
                    self.detector.add_overlays(frame_to_show, self.processed)
                else:
                    self.processed = self.detector.process_frame(frame_to_show)
                    self.check_detected()
                cv2.imshow('Video', frame_to_show)
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
        elif self.video_source == "realsense":
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()

            config = rs.config()
            #rs-enumerate-devices
            #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8,6)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8,6)
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
        if self.video_max_width is not None and new_frame.shape[1] > self.video_max_width or \
                self.video_max_height is not None and new_frame.shape[0] > self.video_max_height:
            new_frame = images.image_resize(new_frame, width=self.video_max_width, height=self.video_max_height)
        self.frame = new_frame
        return self.frame

    def process_frame(self):
        while True:
            if self.frame is not None:
                self.processed = self.detector.process_frame(self.frame, overlays=False)
                self.check_detected()

    def check_detected(self):
        for fd in self.faces_detected:
            self.faces_detected[fd].prepare()
        if self.processed:
            for p in self.processed:
                if p.detected:
                    name = p.classes[0]
                    if name not in self.faces_detected:
                        self.faces_detected[name] = InVideoDetected()
                    self.faces_detected[name].exists_in_frame(True, bbox=p.bbox)
        for fd in list(self.faces_detected):
            self.faces_detected[fd].exists_in_frame(False)
            if self.faces_detected[fd].not_detected_anymore and not self.faces_detected[fd].notified:
                del self.faces_detected[fd]
                continue
            if self.faces_detected[fd].make_notify():
                fd_ = fd.replace(' ', '_')
                position, company, image = None, None, None
                if fd_ in self.detector.meta:
                    meta = self.detector.meta[fd_]
                    if 'position' in meta:
                        position = meta['position']
                    if 'company' in meta:
                        company = meta['company']
                bbox = self.faces_detected[fd].bbox
                if self.frame is not None and bbox is not None:
                    cropped = images.crop_by_boxes(self.frame, [bbox])
                    image = cropped[0]
                self.notifies_queqe.append({
                    'name': fd,
                    'position': position,
                    'company': company,
                    'image': image,
                })

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

    def notify(self):
        while True:
            if len(self.notifies_queqe) == 0:
                time.sleep(1)
            else:
                n = self.notifies_queqe.pop()
                notify(n['name'], position=n['position'], company=n['company'], image=n['image'])


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
    parser.add_argument(
        '--video_max_width',
        help='Resize video if width more than specified value.',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--video_max_height',
        help='Resize video if height more than specified value.',
        type=int,
        default=None,
    )
