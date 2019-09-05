import tempfile

import cv2
import logging
import threading
from app.tools import images

import numpy as np

from app.recognize import detector, classifiers
from ml_serving.utils import helpers

net_loaded = False
load_lock = threading.Lock()

app_detector: detector.Detector = None
app_classifier: classifiers.Classifiers = None

LOG = logging.getLogger(__name__)


def process(inputs, ctx):

    global net_loaded

    # check-lock-check
    if not net_loaded:
        with load_lock:
            if not net_loaded:
                _load(ctx)
                net_loaded = True

    img, _ = helpers.load_image(inputs, 'input', rgb=False)
    bboxes_, imgs_, _, skips = app_detector.process_faces_info(img)
    for i, ii in enumerate(imgs_):
        cv2.imwrite('/test/imgs_{}.png'.format(i), ii)
    bboxes, imgs = [], []
    for idx, img_ in enumerate(imgs_):
        if idx not in skips:
            img_ = img_[:, :, ::-1]
            imgs.append(img_)
            bboxes.append(bboxes_[idx])
    if len(bboxes) == 0:
        raise RuntimeError('no faces detected')
    if len(bboxes) > 1:
        raise RuntimeError('more than one face detected')

    bbox = bboxes[0][:4].astype(int)

    aug_imgs = app_classifier.apply_augmentation(imgs)
    embeddings = app_classifier.embeddings(aug_imgs)

    _, tmp_img = tempfile.mkstemp(suffix='.png')
    previews = images.get_images(
        img, np.array(bboxes),
        face_crop_size=app_detector.facenet_image_size,
    )
    cv2.imwrite(tmp_img, previews[0])

    with open(tmp_img, 'rb') as f:
        face = f.read()

    return dict(
        bbox=bbox,
        face=face,
        embeddings=embeddings
    )


def _load(ctx):

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

    LOG.info('Loading preparer...')
    clf = classifiers.Classifiers()
    clf.serving = facenet_driver
    global app_classifier
    app_classifier = clf
    LOG.info('Loading preparer done!')
