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
    'threshold': [0.3, 0.7, 0.7],
    'debug': 'false',
    'bg_remove_path': '',
    'output_type': 'bytes',
    'need_table': True,
    'add_image_dir': '',
}
lock = threading.Lock()
width = 640
height = 480
net_loaded = False
openvino_facenet: detector.Detector = None


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)

    if not isinstance(PARAMS['threshold'], list):
        PARAMS['threshold'] = [
            float(x) for x in PARAMS['threshold'].split(',')
        ]

    PARAMS['need_table'] = _boolean_string(PARAMS['need_table'])
    LOG.info('Init with params:')
    LOG.info(json.dumps(PARAMS, indent=2))


def process(inputs, ctx, **kwargs):
    global net_loaded
    # check-lock-check
    if not net_loaded:
        with lock:
            if not net_loaded:
                _load_nets(**kwargs)
                net_loaded = True

    action = _string_input_value(inputs, 'action')
    LOG.info("!!! action")
    LOG.info(action)
    LOG.info(type(action))
    LOG.info(action == 'classes')
    LOG.info(str(action) == 'classes')

    if action == "test":
        return process_test()
    elif action == "classes":
        return process_classes()
    elif action == "clarify":
        return process_clarify(inputs)
    else:
        return process_recognize(inputs, ctx, kwargs['model_inputs'])


def process_classes():
    global openvino_facenet
    return {'classes': np.array(openvino_facenet.classes, dtype=np.string_)}


def process_clarify(inputs):
    face_image = _load_image(inputs, 'input')
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_class = _string_input_value(inputs, 'name')
    if face_class is None:
        raise ValueError('name is not specified')
    position = _string_input_value(inputs, 'position')
    company = _string_input_value(inputs, 'company')
    return {
        'result': "clarified class '%s' (position '%s', company '%s') with image %dx%d" \
                  % (face_class, position, company, face_image.shape[1], face_image.shape[0]),
        'mock': True,
    }
    # if PARAMS['add_image_dir'] == '' or not os.path.isdir(PARAMS['add_image_dir']):
    #     raise EnvironmentError('directory for images adding "%s" is not set or absent' % PARAMS['add_image_dir'])
    # face_class = _string_input_value(inputs, 'class')
    # if face_class is None:
    #     raise ValueError('face class is not specified')
    # face_image = _load_image(inputs, 'input')
    # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    # class_dir = os.path.join(PARAMS['add_image_dir'], face_class)
    # if not os.path.isdir(class_dir):
    #     shutil.rmtree(class_dir, ignore_errors=True)
    #     os.makedirs(class_dir)
    # cv2.imwrite("%s.png" % os.path.join(class_dir, str(round(time.time() * 1000))), face_image)
    # return {'added': True}


def process_recognize(inputs, ctx, model_inputs):
    frame = _load_image(inputs, 'input')
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

    model_input = list(model_inputs.keys())[0]
    if not skip:
        outputs = ctx.driver.predict({model_input: imgs})
    else:
        outputs = {'dummy': []}

    facenet_output = list(outputs.values())[0]

    processed_frame = []

    for img_idx, item_output in enumerate(facenet_output):
        if skip:
            break
        processed = openvino_facenet.process_output(
            item_output, bounding_boxes[img_idx]
        )
        processed_frame.append(processed)

    ret = {
        'boxes': bounding_boxes,
        'labels': np.array([processed.label for processed in processed_frame], dtype=np.string_),
    }

    if PARAMS['need_table']:

        table = []

        for processed in processed_frame:
            x_min = int(max(0, processed.bbox[0]))
            y_min = int(max(0, processed.bbox[1]))
            x_max = int(min(frame.shape[1], processed.bbox[2]))
            y_max = int(min(frame.shape[0], processed.bbox[3]))
            cim = frame[y_min:y_max, x_min:x_max]
            cim = cv2.cvtColor(cim, cv2.COLOR_RGB2BGR)
            image_bytes = cv2.imencode(".jpg", cim, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()

            encoded = base64.encodebytes(image_bytes).decode()
            row_data = {
                'name': processed.label,
                'prob': processed.prob,
                'image': encoded
            }
            if processed.meta is not None:
                row_data['meta'] = processed.meta
                if 'position' in processed.meta:
                    row_data['position'] = processed.meta['position']
                if 'company' in processed.meta:
                    row_data['company'] = processed.meta['company']

            image_clarify = {
                'alternate': True,
            }

            if not processed.detected:
                image_clarify['values'] = []
                for cls in processed.classes:
                    cl_cls = {'name': cls}
                    cls_ = cls.replace(" ", "_")
                    if cls_ in processed.classes_meta:
                        m = processed.classes_meta[cls_]
                        if 'position' in m:
                            cl_cls['position'] = m['position']
                        if 'company' in m:
                            cl_cls['company'] = m['company']
                    image_clarify['values'].append(cl_cls)

            row_data['image_clarify'] = image_clarify

            table.append(row_data)

        ret['table_output'] = json.dumps(table)
        ret['table_meta'] = json.dumps([
            {
                'name': 'name',
                'label': 'Name'
            },
            {
                'name': 'position',
                'label': 'Position'
            },
            {
                'name': 'company',
                'label': 'Company'
            },
            {
                'name': 'prob',
                'label': 'Probability',
                'type': 'number',
                'format': '.2f'
            },
            {
                'name': 'meta',
                'label': 'Metadata',
                'type': 'data'
            },
            {
                'name': 'image',
                'label': 'Image',
                'type': 'image'
            },
            {
                'name': 'image_clarify',
                'label': 'Clarify',
                'type': 'edit',
                'action': 'clarify',
                'values_label': 'name',
                'fields': ['name', 'position', 'company', 'image']
            },
        ])

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not skip:
        openvino_facenet.add_overlays(frame, processed_frame)

    if PARAMS['output_type'] == 'bytes':
        image_bytes = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()
    else:
        image_bytes = frame

    ret['output'] = image_bytes

    return ret


def process_test():
    return {'test': 'test'}


def _string_input_value(inputs, key):
    array = inputs.get(key)
    if array is None:
        return None

    if len(array.shape) == 0:
        elem = array.tolist()
    else:
        elem = array[0]

    if isinstance(elem, bytes):
        elem = elem.decode()

    return str(elem)


def _load_image(inputs, image_key):
    image = inputs.get(image_key)
    if image is None:
        raise RuntimeError('Missing "{0}" key in inputs. Provide an image in "{0}" key'.format(image_key))

    if len(image.shape) == 0:
        image = np.stack([image.tolist()])

    if len(image.shape) < 3:
        image = cv2.imdecode(np.frombuffer(image[0], np.uint8), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def _boolean_string(s):
    if isinstance(s, bool):
        return s

    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def _load_nets(**kwargs):
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
        model_path=PARAMS['model_path'],
        debug=PARAMS['debug'] == 'true',
        bg_remove_path=PARAMS['bg_remove_path'],
        loaded_plugin=kwargs['plugin']
    )
    openvino_facenet.init()

    LOG.info('Done.')
