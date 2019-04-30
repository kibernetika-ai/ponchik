import base64
import json
import logging
import os
import shutil
import threading
import time

import cv2
import numpy as np

from svod_rcgn.recognize import detector
from svod_rcgn.tools import images, dataset
from svod_rcgn.mlboard import mlboard

LOG = logging.getLogger(__name__)
PARAMS = {
    'device': 'CPU',
    'classifiers_dir': '',
    'threshold': [0.3, 0.7, 0.7],
    'debug': 'false',
    'bg_remove_path': '',
    'output_type': 'bytes',
    'need_table': True,
    'clarified_dir': '',
    'uploaded_dir': '',
    'project_name': '',
}
load_lock = threading.Lock()
width = 640
height = 480
net_loaded = False
openvino_facenet: detector.Detector = None
clarified_in_process = False
uploaded_in_process = False
process_lock = threading.Lock()


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

    clarify_checker = threading.Thread(target=_retrain_checker, daemon=True)
    clarify_checker.start()


def process(inputs, ctx, **kwargs):
    global net_loaded
    # check-lock-check
    if not net_loaded:
        with load_lock:
            if not net_loaded:
                _load_nets(**kwargs)
                net_loaded = True

    action = _string_input_value(inputs, 'action')
    if action == "test":
        return process_test()
    elif action == "classes":
        return process_classes()
    elif action == "clarify":
        return process_clarified(inputs)
    elif action == "image":
        return process_uploaded(inputs)
    else:
        return process_recognize(inputs, ctx, kwargs['model_inputs'])


def process_classes():
    global openvino_facenet
    return {'classes': np.array(openvino_facenet.classes, dtype=np.string_)}


def process_clarified(inputs):
    e, d = _clarify_enabled()
    if not e:
        raise EnvironmentError('directory for clarified data "%s" is not set or absent' % d)
    res = _upload_processed_image(inputs, 'face', d, 'clarified')
    global clarified_in_process
    clarified_in_process = True
    return res


def process_uploaded(inputs):
    e, d = _upload_enabled()
    if not e:
        raise EnvironmentError('directory for images to recognition "%s" is not set or absent' % d)
    res = _upload_processed_image(inputs, 'image', d)
    global uploaded_in_process
    uploaded_in_process = True
    return res


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
        clarify_enabled, _ = _clarify_enabled()

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
                'face': encoded
            }
            if processed.meta is not None:
                row_data['meta'] = processed.meta
                if 'position' in processed.meta:
                    row_data['position'] = processed.meta['position']
                if 'company' in processed.meta:
                    row_data['company'] = processed.meta['company']

            if clarify_enabled:

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
        meta = [
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
                'name': 'face',
                'label': 'Face',
                'type': 'image'
            },
        ]
        if clarify_enabled:
            meta.append({
                'name': 'image_clarify',
                'label': 'Clarify',
                'type': 'edit',
                'action': 'clarify',
                'values_label': 'name',
                'fields': ['name', 'position', 'company', 'face']
            })
        ret['table_meta'] = json.dumps(meta)

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


def _clarify_enabled():
    cd = PARAMS['clarified_dir'] if 'clarified_dir' in PARAMS else ''
    e = (cd != '' and os.path.isdir(cd))
    return e, cd


def _upload_enabled():
    pid = PARAMS['uploaded_dir'] if 'uploaded_dir' in PARAMS else ''
    e = (pid != '' and os.path.isdir(pid))
    return e, pid


def _upload_processed_image(inputs, image_name, upload_dir, file_prefix='uploaded'):
    name = _string_input_value(inputs, 'name')
    if name is None:
        raise ValueError('name is not specified')
    image = _load_image(inputs, image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    class_dir = os.path.join(upload_dir, name.replace(" ", "_"))
    if not os.path.isdir(class_dir):
        shutil.rmtree(class_dir, ignore_errors=True)
        os.makedirs(class_dir)
    img_file = os.path.join(class_dir, "%s_%s.png" % (file_prefix, str(round(time.time() * 1000))))
    cv2.imwrite(img_file, image)

    meta = {}
    position = _string_input_value(inputs, 'position')
    if position is not None:
        meta['position'] = position
    company = _string_input_value(inputs, 'company')
    if company is not None:
        meta['company'] = company

    res = {'saved': True, 'meta_saved': False}

    if len(meta) > 0:
        with open(os.path.join(class_dir, dataset.META_FILENAME), 'w') as mw:
            json.dump(meta, mw)
        res['meta_saved'] = True

    return res


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


def _retrain_checker():
    global clarified_in_process, uploaded_in_process
    while True:
        if clarified_in_process:
            clarified_in_process = False
            with process_lock:
                _run_retrain_task('prepare-clarified')
        if uploaded_in_process:
            uploaded_in_process = False
            with process_lock:
                _run_retrain_task('prepare-uploaded')
        time.sleep(10)


def _run_retrain_task(task_name):
    if mlboard is not None:
        app_name = '%s-%s' % (os.environ.get('WORKSPACE_ID'), PARAMS['project_name'])
        LOG.info('retrain with task "%s:%s"' % (app_name, task_name))
        try:
            app = mlboard.apps.get(app_name)
        except Exception as e:
            LOG.error('get app "%s" error: %s' % (app_name, e))
            return
        task = app.task(task_name)
        if task is None:
            LOG.error('app "%s" has no task "%s"' % (app_name, task_name))
            return
        try:
            task.run()
            # LOG.info('retrain with task "%s:%s" DONE' % (app_name, task_name))
        except Exception as e:
            LOG.error('run task "%s:%s" error "%s"' % (app_name, task_name, e))
    else:
        LOG.warning("Unable to retrain: mlboard is absent")
