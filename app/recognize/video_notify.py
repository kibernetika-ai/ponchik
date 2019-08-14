from time import time

from app.tools import images


class InVideoDetected:
    notify_period = 4
    notify_prob = .5
    stay_notified = 2

    def __init__(self):
        self.done = False
        self.prob = 0
        self.not_detected_anymore = False
        self.in_frames = []
        self.in_frames_ts = []
        self.notified_awaiting = False
        self.notified = False
        self.notified_ts = None
        self.looks_like = []
        self.prob = 0
        self.image = None
        self.dists = 0
        self.counter = 0
        self.last = 0

    def prepare(self):
        self.done = False
        if self.last == 0:
            self.last = time()

    def exists_in_frame(self, face_info=None, frame=None):
        if not self.done:
            if face_info is None:
                c = 0
                d = 0
            else:
                d = 0 if face_info.dist is None else face_info.dist
                c=1
            self.dists += d
            self.counter += c
            now = time()
            if (now - self.last) > self.notify_period:
                if self.counter > 1:
                    if self.dists / self.counter < 0.4:
                        self.notified = True
                        self.notified_awaiting = True
                    else:
                        self.dists = d
                        self.counter = c
                        self.notified = False
                        self.last = now
            if face_info:
                if face_info.looks_like:
                    self.looks_like.extend(face_info.looks_like)
                    self.looks_like = list(set(self.looks_like))
                    self.looks_like.sort()
                if face_info.prob is not None and face_info.prob > self.prob and frame is not None:
                    self.prob = face_info.prob
                    self.image = images.crop_by_box(frame, face_info.bbox)
            self.done = True


    def exists_in_frame_old(self, face_info=None, frame=None):
        if not self.done:
            exists = face_info is not None
            self.in_frames.append(1 if exists else 0)
            now = time()
            self.in_frames_ts.append(now)
            period_filled = False
            while len(self.in_frames_ts) > 1 and now - self.in_frames_ts[0] > self.notify_period:
                del self.in_frames[0]
                del self.in_frames_ts[0]
                period_filled = True
            if period_filled:
                self.prob = sum(self.in_frames) / len(self.in_frames)
                if self.notified:
                    if now - self.notified_ts > self.stay_notified:
                        self.notified = False
                        self.notified_ts = None
                if self.prob > self.notify_prob and not self.notified:
                    self.notified = True
                    self.notified_awaiting = True
                    self.notified_ts = now
                if self.prob == 0:
                    self.not_detected_anymore = True
            if face_info:
                if face_info.looks_like:
                    self.looks_like.extend(face_info.looks_like)
                    self.looks_like = list(set(self.looks_like))
                    self.looks_like.sort()
                if face_info.prob is not None and face_info.prob > self.prob and frame is not None:
                    self.prob = face_info.prob
                    self.image = images.crop_by_box(frame, face_info.bbox)
            self.done = True

    def make_notify(self):
        if self.notified and self.notified_awaiting:
            self.notified_awaiting = False
            return True
        return False


def init_in_video_detected(args):
    InVideoDetected.notify_period = args.notify_face_detection_period
    InVideoDetected.prob = args.notify_face_detection_prob
    InVideoDetected.stay_notified = args.notify_face_detection_stay


def add_video_notify_args(parser):
    parser.add_argument(
        '--notify_face_detection_period',
        help='Period (seconds) for face detection for notification.',
        type=int,
        default=InVideoDetected.notify_period,
    )
    parser.add_argument(
        '--notify_face_detection_prob',
        help='Probability for notification.',
        type=float,
        default=InVideoDetected.notify_prob,
    )
    parser.add_argument(
        '--notify_face_detection_stay',
        help='Period (seconds) prevents sending repeated notification.',
        type=float,
        default=InVideoDetected.stay_notified,
    )
