import os
import time
from threading import Thread

import cv2
import numpy as np
from datetime import timedelta

from app.recognize import defaults
from app.recognize.clusterizator import Clusterizator
from app.tools import images, print_fun
from app.recognize.video_notify import InVideoDetected
from app.recognize import detector
from app.notify import notify


def video_args(detector, listener, args):
    return Video(
        detector,
        listener=listener,
        video_source=args.video_source,
        video_async=args.video_async,
        video_max_width=args.video_max_width,
        video_max_height=args.video_max_height,
        video_each_of_frame=args.video_each_of_frame,
        not_detected_store=args.video_not_detected_store,
        not_detected_check_period=args.video_not_detected_check_period,
        not_detected_dir=args.video_not_detected_dir,
        process_not_detected=args.process_not_detected,
        video_export_srt=args.video_export_srt,
    )


class Video:
    def __init__(self, detector,
                 listener=None, video_source=None, video_async=False,
                 video_max_width=None, video_max_height=None, video_each_of_frame=1, video_export_srt=False,
                 not_detected_store=False, not_detected_check_period=defaults.NOT_DETECTED_CHECK_PERIOD,
                 not_detected_dir=defaults.NOT_DETECTED_DIR, process_not_detected=False):
        self.detector = detector
        self.video_source = video_source
        self.video_source_is_file = False
        self.video_source_fps = None
        self.frame = None
        self.processed = None
        self.vs = None
        self.listener = listener
        self.video_async = video_async
        self.pipeline = None
        self.video_max_width = video_max_width
        self.video_max_height = video_max_height
        self.video_each_of_frame = max(1, video_each_of_frame)
        self.faces_detected = {}
        self.notify_started = False
        self.notifies_queue = []
        self.not_detected_store = not_detected_store
        self.not_detected_check_period = not_detected_check_period
        self.not_detected_dir = not_detected_dir
        if self.not_detected_store and not os.path.isdir(self.not_detected_dir):
            raise RuntimeError('directory %s is not exists' % self.not_detected_dir)
        self.not_detected_check_ts = time.time()
        self.process_not_detected = process_not_detected
        self.video_export_srt = video_export_srt


    def start_notify(self):
        if self.notify_started:
            return
        notify_thread = Thread(target=self.notify, daemon=True)
        notify_thread.start()
        self.notify_started = True

    def start(self):
        self.detector.init()
        if self.vs is None:
            self.init_video()
        if self.video_async:
            frame_thread = Thread(target=self.process_frame_async, daemon=True)
            frame_thread.start()
        if self.listener is not None:
            listen_thread = Thread(target=self.listen, daemon=True)
            listen_thread.start()

        self.start_notify()

        try:
            iframe = 0
            rframe = 0
            while True:
                # Capture frame-by-frame
                self.get_frame()
                rframe += 1
                if self.frame is None and self.video_source_is_file:
                    break
                if self.frame is not None and (iframe % self.video_each_of_frame == 0):
                    frame = self.frame.copy()
                    if self.video_async:
                        self.detector.add_overlays(frame, self.processed)
                    else:
                        self.process_frame(frame=frame)
                    cv2.imshow('Video', frame)
                iframe += 1
                key = cv2.waitKey(1)
                # Wait 'q' or Esc or 'q' in russian layout
                if key in [ord('q'), 202, 27]:
                    break
                if iframe % 100 == 0:
                    t = timedelta(milliseconds=rframe / self.video_source_fps * 1000) if self.video_source_fps else '-'
                    print_fun("Processed %d frames, %s" % (iframe, t))
        except (KeyboardInterrupt, SystemExit) as e:
            print_fun('Caught %s: %s' % (e.__class__.__name__, e))
        finally:
            if self.pipeline is not None:
                self.pipeline.stop()

        if self.video_export_srt:

            import srt

            detected_persons = self.detector.detected_names
            not_detected_persons = []

            if self.process_not_detected:
                print_fun('Start unrecognized faces clusterization')
                cl = Clusterizator()
                not_detected_persons = cl.clusterize_frame_faces(self.detector.not_detected_embs)
                print_fun('Clusterization done')

            subs = []
            cur_sub, cur_sub_start, cur_sub_end = None, None, None
            for i in range(max(len(detected_persons), len(not_detected_persons))):
                ts = timedelta(milliseconds=(i / self.video_source_fps) * 1000 * self.video_each_of_frame)
                frame_persons = []
                if len(detected_persons) > i:
                    names = detected_persons[i]
                    names.sort()
                    frame_persons.extend(names)
                if len(not_detected_persons) > i:
                    persons = [d for d in not_detected_persons[i] if d >= 0]
                    persons.sort()
                    frame_persons.extend(['Person %d' % d for d in persons])
                sub = None if len(frame_persons) == 0 else ", ".join(frame_persons)
                if sub != cur_sub:
                    if cur_sub is not None:
                        cur_sub_end = ts
                        subs.append(
                            srt.Subtitle(index=len(subs)+1, start=cur_sub_start, end=cur_sub_end, content=cur_sub))
                    else:
                        cur_sub_start = ts
                    cur_sub = sub

            if cur_sub is not None:
                subs.append(
                    srt.Subtitle(index=len(subs) + 1, start=cur_sub_start, end=cur_sub_end, content=cur_sub))

            with open(os.path.splitext(self.video_source)[0]+'.srt', 'w') as sw:
                sw.write(srt.compose(subs))

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
            self.video_source_is_file = os.path.isfile(self.video_source)
            if self.video_source_is_file:
                self.video_source_fps = self.vs.get(cv2.CAP_PROP_FPS)
            if self.video_export_srt:
                if not self.video_source_is_file:
                    raise ValueError('srt creation allowed only for video file')
                if not self.video_source_fps:
                    raise ValueError('unable to detect fps, srt creation is unavailable')

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
            print_fun("Oops frame is None. Possibly camera or display does not work")
            self.frame = None
            return None
        if self.video_max_width is not None and new_frame.shape[1] > self.video_max_width or \
                self.video_max_height is not None and new_frame.shape[0] > self.video_max_height:
            new_frame = images.image_resize(new_frame, width=self.video_max_width, height=self.video_max_height)
        self.frame = new_frame
        return self.frame

    def process_frame_async(self):
        while True:
            self.process_frame(self.frame, overlays=False)

    def process_frame(self, frame, overlays=True):
        if frame is not None:
            original_copy = np.copy(frame)
            self.processed = self.detector.process_frame(frame, overlays=overlays)
            self.research_processed(frame=original_copy)

    def research_processed(self, frame=None):
        if frame is None:
            frame = self.frame

        for fd in self.faces_detected:
            self.faces_detected[fd].prepare()
        if self.processed is not None:
            store_not_detected = False
            now = time.time()
            if self.not_detected_store:
                if now - self.not_detected_check_ts > self.not_detected_check_period:
                    self.not_detected_check_ts = now
                    store_not_detected = True

            for p in self.processed:
                if p.state == detector.DETECTED:
                    name = p.classes[0]
                    if name not in self.faces_detected:
                        self.faces_detected[name] = InVideoDetected()
                    self.faces_detected[name].exists_in_frame(processed=p, frame=frame)
                elif p.state == detector.NOT_DETECTED and store_not_detected:
                    img = images.crop_by_box(frame, p.bbox)
                    cv2.imwrite(os.path.join(self.not_detected_dir, '%s.jpg' % now), img)

        for fd in list(self.faces_detected):
            self.faces_detected[fd].exists_in_frame()
            if self.faces_detected[fd].not_detected_anymore and not self.faces_detected[fd].notified:
                del self.faces_detected[fd]
                continue
            if self.faces_detected[fd].make_notify():
                n = {'name': fd}
                fd_ = fd.replace(' ', '_')
                if fd_ in self.detector.meta:
                    meta = self.detector.meta[fd_]
                    if 'position' in meta:
                        n['position'] = meta['position']
                    if 'company' in meta:
                        n['company'] = meta['company']
                    if 'url' in meta:
                        n['url'] = meta['url']
                if self.faces_detected[fd].image is not None:
                    n['image'] = self.faces_detected[fd].image
                if len(self.faces_detected[fd].looks_like) > 0:
                    n['action_options'] = self.faces_detected[fd].looks_like.copy()
                self.notifies_queue.append(n)

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
            if len(self.notifies_queue) == 0:
                time.sleep(1)
            else:
                n = self.notifies_queue.pop()
                notify(**n)


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
    parser.add_argument(
        '--video_each_of_frame',
        help='Process every N frame (1 for not skipping).',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--video_export_srt',
        help='Export SRT-file with detected/recognized persons',
        action='store_true',
    )
    parser.add_argument(
        '--video_not_detected_store',
        help='Store not detected faces.',
        action='store_true',
    )
    parser.add_argument(
        '--video_not_detected_dir',
        help='Store not detected faces to specified directory.',
        type=str,
        default=defaults.NOT_DETECTED_DIR
    )
    parser.add_argument(
        '--video_not_detected_check_period',
        help='Check not detected faces for every N seconds.',
        type=int,
        default=defaults.NOT_DETECTED_CHECK_PERIOD,
    )
