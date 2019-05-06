from time import time
from svod_rcgn.notify import notify


class InVideoDetected:

    notify_period = 3
    notify_prob = .5
    stay_notified = 600

    def __init__(self, name):
        self.name = name
        self.processed = False
        self.prob = 0
        self.not_detected_anymore = False
        self.in_frames = []
        self.in_frames_ts = []
        self.notified = False
        self.notified_ts = None

    def prepare(self):
        self.processed = False

    def exists_in_frame(self, exists):
        if not self.processed:
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
                    self.notified_ts = now
                    notify(self.name)
                if self.prob == 0:
                    self.not_detected_anymore = True
            self.processed = True


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