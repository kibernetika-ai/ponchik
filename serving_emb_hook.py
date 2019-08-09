import logging
import threading

from app.recognize import detector
from ml_serving.utils import helpers

net_loaded = False
load_lock = threading.Lock()

app_detector: detector.Detector = None

LOG = logging.getLogger(__name__)


def process(inputs, ctx):

    global net_loaded

    # check-lock-check
    if not net_loaded:
        with load_lock:
            if not net_loaded:
                _load_detector(ctx)
                net_loaded = True

    img, _ = helpers.load_image(inputs, 'input')
    faces = app_detector.process_frame(img, overlays=False)

    if len(faces) == 0:
        raise RuntimeError('no faces detected')
    if len(faces) > 1:
        raise RuntimeError('more than one face detected')

    return {'output': faces[0].embedding}


def _load_detector(ctx):

    LOG.info('Loading detector...')

    face_driver = ctx.drivers[0]
    facenet_driver = ctx.drivers[1]
    head_pose_driver = ctx.drivers[2]

    d = detector.Detector(
        face_driver=face_driver,
        facenet_driver=facenet_driver,
        head_pose_driver=head_pose_driver,
    )
    d.init()
    global app_detector
    app_detector = d

    LOG.info('Loading detector done!')

