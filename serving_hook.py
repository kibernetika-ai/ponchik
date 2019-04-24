import base64
import json
import logging
import os
import threading

import cv2
import numpy as np

from svod_rcgn.recognize import detector
from svod_rcgn.tools import images

LOG = logging.getLogger(__name__)
PARAMS = {
    'device': 'CPU',
    'classifiers_dir': '',
    'model_dir': '',
    'threshold': [0.3, 0.7, 0.7],
    'debug': 'false',
    'bg_remove_path': '',
    'output_type': 'bytes',
    'need_table': True,
}
lock = threading.Lock()
width = 640
height = 480
net_loaded = False
openvino_facenet: detector.Detector = None


def boolean_string(s):
    if isinstance(s, bool):
        return s

    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)

    if not isinstance(PARAMS['threshold'], list):
        PARAMS['threshold'] = [
            float(x) for x in PARAMS['threshold'].split(',')
        ]

    PARAMS['need_table'] = boolean_string(PARAMS['need_table'])
    # PARAMS['use_tf'] = boolean_string(PARAMS['use_tf'])
    LOG.info('Init with params:')
    LOG.info(json.dumps(PARAMS, indent=2))


def load_nets(**kwargs):

    LOG.info('Load FACE DETECTION')
    clf_dir = PARAMS['classifiers_dir']
    if not os.path.isdir(clf_dir):
        raise RuntimeError("Classifiers path %s is absent or is not directory" % clf_dir)
    classifiers = \
        [os.path.join(clf_dir, f) for f in os.listdir(clf_dir) if os.path.isfile(os.path.join(clf_dir, f))]
    if len(classifiers) == 0:
        raise RuntimeError("Classifiers path %s has no any files" % clf_dir)
    # LOG.info('Classifiers path: {}'.format(classifiers_path))
    # LOG.info('Classifier files: {}'.format(classifiers))
    global openvino_facenet
    openvino_facenet = detector.Detector(
        device='CPU',
        classifiers_dir=clf_dir,
        model_dir=PARAMS['model_dir'],
        debug=PARAMS['debug'] == 'true',
        bg_remove_path=PARAMS['bg_remove_path'],
        loaded_plugin=kwargs['plugin']
    )
    openvino_facenet.init()

    LOG.info('Done.')


def load_image_from_inputs(inputs, image_key):
    image = inputs.get(image_key)
    if image is None:
        raise RuntimeError('Missing "{0}" key in inputs. Provide an image in "{0}" key'.format(image_key))

    if len(image.shape) == 0:
        image = np.stack([image.tolist()])

    if len(image.shape) < 3:
        image = cv2.imdecode(np.frombuffer(image[0], np.uint8), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def process(inputs, ctx, **kwargs):
    global net_loaded
    # check-lock-check
    if not net_loaded:
        with lock:
            if not net_loaded:
                load_nets(**kwargs)
                net_loaded = True

    frame = load_image_from_inputs(inputs, 'input')
    # convert to BGR
    data = frame[:, :, ::-1]

    bounding_boxes = openvino_facenet.detect_faces(data, PARAMS['threshold'][0])

    imgs = images.get_images(frame, bounding_boxes)

    if len(imgs) > 0:
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        skip = False
    else:
        imgs = np.random.randn(1, 3, 160, 160).astype(np.float32)
        skip = True

    model_input = list(kwargs['model_inputs'].keys())[0]
    if not skip:
        outputs = ctx.driver.predict({model_input: imgs})
    else:
        outputs = {'dummy': []}

    facenet_output = list(outputs.values())[0]
    # LOG.info('output shape = {}'.format(facenet_output.shape))

    labels = []
    box_overlays = []
    scores_out = []
    for img_idx, item_output in enumerate(facenet_output):
        if skip:
            break

        box_overlay, label, prob = openvino_facenet.process_output(
            item_output, bounding_boxes[img_idx]
        )
        box_overlays.append(box_overlay)
        labels.append(label)
        scores_out.append(prob)
        # LOG.info("prob = {}".format(prob))
        # LOG.info("scores_out = {}".format(scores_out))

    text_labels = ["" if l is None else l['label'] for l in labels]

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if PARAMS['output_type'] == 'bytes':
        image_bytes = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()
    else:
        image_bytes = frame

    ret = {
        'boxes': bounding_boxes,
        'output': image_bytes,
        'labels': np.array(text_labels, dtype=np.string_),
    }

    if PARAMS['need_table']:

        table = []

        text_labels = ["" if l is None else l['label'] for l in labels]
        for i, b in enumerate(bounding_boxes):
            x_min = int(max(0, b[0]))
            y_min = int(max(0, b[1]))
            x_max = int(min(frame.shape[1], b[2]))
            y_max = int(min(frame.shape[0], b[3]))
            cim = frame[y_min:y_max, x_min:x_max]
            # image_bytes = io.BytesIO()
            cim = cv2.cvtColor(cim, cv2.COLOR_RGB2BGR)
            image_bytes = cv2.imencode(".jpg", cim, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()

            encoded = base64.encodebytes(image_bytes).decode()
            table.append(
                {
                    'type': 'text',
                    'name': text_labels[i],  # if i in text_labels else '',
                    'prob': float(scores_out[i]),
                    'image': encoded
                }
            )

        ret['table_output'] = json.dumps(table)
        ret['table_meta'] = json.dumps([
            {
                "name": "type",
                "filtered": True
            },
            {
                "name": "name"
            },
            {
                "name": "prob",
                "type": "number",
                "format": ".2f"
            },
            {
                "name": "image",
                "type": "image"
            }
        ])

    return ret
