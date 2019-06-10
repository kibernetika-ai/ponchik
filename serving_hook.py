import base64
import json
import logging
import os
import shutil
import tarfile
import threading
import time

import cv2
import numpy as np

from app.recognize import detector
from app.recognize import defaults
from app.recognize import video
from app.recognize import video_notify
from app import notify
from app.tools import images, dataset
from app.mlboard import mlboard

import pull_model

LOG = logging.getLogger(__name__)
PARAMS = {
    'device': 'CPU',
    'classifiers_dir': '',
    'threshold': [0.3, 0.7, 0.7],
    'head_pose_thresholds': defaults.HEAD_POSE_THRESHOLDS,
    'debug': 'false',
    'bg_remove_path': '',
    'output_type': 'bytes',
    'need_table': True,
    'clarified_dir': '',
    'uploaded_dir': '',
    'project_name': '',

    'enable_log': False,
    'logdir': 'faces_dir',
    'timing': True,
    'skip_frames': False,
    'inference_fps': 2,

    'enable_pull_model': False,
    'base_url': 'https://dev.kibernetika.io/api/v0.2',
    'token': '',
    'model_name': 'ponchik',
    'workspace_name': 'kuberlab-demo',

    'slack_token': '',
    'slack_channel': '',
    'slack_server': '',
    'badge_detector': '',
    'fixed_normalization': False,

    'min_face_size': defaults.MIN_FACE_SIZE,
    'multi_detect': [],
}
load_lock = threading.Lock()
width = 640
height = 480
net_loaded = False

processing: video.Video = None
openvino_facenet: detector.Detector = None
clarified_in_process = False
uploaded_in_process = False
process_lock = threading.Lock()
frame_num = 0
unknown_num = 0
last_fully_processed = None
freq = None
skip_threshold = 0
badge_detector = None


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)

    if not isinstance(PARAMS['threshold'], list):
        PARAMS['threshold'] = [
            float(x) for x in PARAMS['threshold'].split(',')
        ]

    if not isinstance(PARAMS['multi_detect'], list):
        PARAMS['multi_detect'] = [
            int(x) for x in PARAMS['multi_detect'].split(',')
        ]

    if not isinstance(PARAMS['head_pose_thresholds'], list):
        PARAMS['head_pose_thresholds'] = [
            float(x) for x in PARAMS['head_pose_thresholds'].split(',')
        ]

    PARAMS['need_table'] = _boolean_string(PARAMS['need_table'])
    PARAMS['fixed_normalization'] = _boolean_string(PARAMS['fixed_normalization'])
    PARAMS['enable_log'] = _boolean_string(PARAMS['enable_log'])
    PARAMS['timing'] = _boolean_string(PARAMS['timing'])
    PARAMS['skip_frames'] = _boolean_string(PARAMS['skip_frames'])
    PARAMS['enable_pull_model'] = _boolean_string(PARAMS['enable_pull_model'])
    PARAMS['inference_fps'] = int(PARAMS['inference_fps'])
    PARAMS['min_face_size'] = int(PARAMS['min_face_size'])
    LOG.info('Init with params:')
    LOG.info(json.dumps(PARAMS, indent=2))

    clarify_checker = threading.Thread(target=_retrain_checker, daemon=True)
    clarify_checker.start()

    video_notify.InVideoDetected.notify_period = 3.0

    if PARAMS['enable_pull_model']:
        assert PARAMS['workspace_name'] != ''
        assert PARAMS['base_url'] != ''
        assert PARAMS['model_name'] != ''
        assert PARAMS['token'] != ''

        pull_thread = threading.Thread(
            target=pull_model.loop,
            kwargs=dict(
                pattern='* * * * * */20',
                base_url=PARAMS['base_url'],
                ws=PARAMS['workspace_name'],
                name=PARAMS['model_name'],
                token=PARAMS['token'],
                callback=reload_detector,
            ),
            daemon=True
        )
        pull_thread.start()

    if PARAMS['enable_log']:
        if not os.path.exists(PARAMS['logdir']):
            os.mkdir(PARAMS['logdir'])

        log_file = os.path.join(PARAMS['logdir'], 'log.txt')
        if os.path.exists(log_file):
            os.remove(log_file)


def reload_detector(version, fileobj):
    global openvino_facenet
    clf_dir = PARAMS['classifiers_dir']

    tar = tarfile.open(fileobj=fileobj)
    LOG.info('Extracting new version %s to %s...' % (version, clf_dir))

    shutil.rmtree(clf_dir, ignore_errors=True)
    os.mkdir(clf_dir)
    tar.extractall(clf_dir)

    LOG.info('Reloading classifiers...')
    if openvino_facenet is not None:
        openvino_facenet.load_classifiers()


def process(inputs, ctx, **kwargs):
    global net_loaded
    # check-lock-check
    if not net_loaded:
        with load_lock:
            if not net_loaded:
                _load_nets(ctx=ctx)
                net_loaded = True

    if PARAMS['badge_detector'] == 'yes':
        import app.badge.badge_detector as badge
        global badge_detector
        if badge_detector is None:
            badge_detector = badge.BadgePorcessor(openvino_facenet.classes, ctx.drivers[1], ctx.drivers[2], 0.5, 0.5)
    action = _string_input_value(inputs, 'action')
    if action == "test":
        return process_test()
    elif action == "classes" or action == "names":
        return process_names(action)
    elif action == "meta":
        return process_meta()
    elif action == "clarify":
        return process_clarified(inputs)
    elif action == "image":
        return process_uploaded(inputs)
    else:
        return process_recognize(inputs, ctx, **kwargs)


def process_names(action):
    global openvino_facenet
    return {action: np.array(openvino_facenet.classes, dtype=np.str)}


def process_meta():
    global openvino_facenet
    ret = []
    for cl in openvino_facenet.classes:
        item = {
            'name': cl,
        }
        cl_k = cl.replace(' ', '_')
        if cl_k in openvino_facenet.meta:
            cl_m = openvino_facenet.meta[cl_k]
            if 'position' in cl_m:
                item['position'] = cl_m['position']
            if 'company' in cl_m:
                item['company'] = cl_m['company']
            if 'url' in cl_m:
                item['url'] = cl_m['url']
        ret.append(item)
    return {'meta': json.dumps(ret)}


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


def process_recognize(inputs, ctx, **kwargs):
    global frame_num
    global last_fully_processed
    global freq
    global skip_threshold
    frame_num += 1

    if PARAMS['skip_frames'] and last_fully_processed is not None:
        if freq is None:
            fps = int(_get_fps(**kwargs))

            freq = (fps - PARAMS['inference_fps']) / float(fps)

        if PARAMS['inference_fps'] != 0:
            skip_threshold += freq
            if skip_threshold > 1:
                skip_threshold -= 1
                return last_fully_processed

    frame = _load_image(inputs, 'input')
    # convert to BGR
    bgr_frame = np.copy(frame[:, :, ::-1])

    face_infos = processing.process_frame(bgr_frame, overlays=True)
    faces_bbox = [fi.bbox for fi in face_infos]
    if badge_detector is not None:
        badge_detector.process(frame[:, :, :].copy(), faces_bbox)
    ret = {
        'boxes': np.array(faces_bbox),
        'labels': np.array([processed.label for processed in face_infos], dtype=np.str),
    }

    if PARAMS['enable_log']:
        log_recognition(frame, ret, **kwargs)

    if PARAMS['need_table']:
        table_result = build_table(frame, face_infos)
        ret.update(table_result)

    if PARAMS['output_type'] == 'bytes':
        image_output = cv2.imencode(".jpg", bgr_frame, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()
    else:
        rgb_frame = bgr_frame[:, :, ::-1]
        image_output = rgb_frame

    ret['output'] = image_output

    last_fully_processed = ret

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
    url = _string_input_value(inputs, 'url')
    if url is not None:
        meta['url'] = url

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
        image = image[:, :, ::-1]

    return image


def _boolean_string(s):
    if isinstance(s, bool):
        return s

    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def _load_nets(ctx):
    head_pose_driver = None
    face_driver = None
    facenet_driver = None
    if hasattr(ctx, 'drivers'):
        if len(ctx.drivers) >= 2:
            face_driver = ctx.drivers[0]
            facenet_driver = ctx.drivers[1]
        if len(ctx.drivers) >= 5:
            head_pose_driver = ctx.drivers[4]
        elif len(ctx.drivers) == 3:
            head_pose_driver = ctx.drivers[2]
    else:
        face_driver = ctx.driver

    clf_dir = PARAMS['classifiers_dir']
    if not os.path.isdir(clf_dir):
        raise RuntimeError("Classifiers path %s is absent or is not directory" % clf_dir)

    classifiers = []
    for path in os.listdir(clf_dir):
        file_path = os.path.join(clf_dir, path)
        if os.path.isfile(file_path):
            classifiers.append(file_path)

    if len(classifiers) == 0:
        raise RuntimeError("Classifiers path %s has no any files" % clf_dir)
    # LOG.info('Classifiers path: {}'.format(classifiers_path))
    # LOG.info('Classifier files: {}'.format(classifiers))
    global openvino_facenet
    ot = detector.Detector(
        classifiers_dir=clf_dir,
        face_driver=face_driver,
        debug=PARAMS['debug'] == 'true',
        facenet_driver=facenet_driver,
        head_pose_driver=head_pose_driver,
        head_pose_thresholds=PARAMS['head_pose_thresholds'],
        min_face_size=PARAMS['min_face_size'],
        multi_detect=PARAMS['multi_detect'],
        threshold=PARAMS['threshold'][0],
        normalization=PARAMS['normalization'],
    )
    ot.init()
    openvino_facenet = ot

    if PARAMS['slack_channel'] and PARAMS['slack_token'] and PARAMS['slack_server']:
        notify.init_notifier_slack(
            PARAMS['slack_token'],
            PARAMS['slack_channel'],
            PARAMS['slack_server']
        )

    global processing
    processing = video.Video(
        detector=ot,
        video_async=False,
        not_detected_store=False,
    )
    processing.start_notify()

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
        ws_id, prj_name = os.environ.get('WORKSPACE_ID'), PARAMS['project_name']
        if ws_id and prj_name:
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
            LOG.warning("Unable to retrain: no project info")
    else:
        LOG.warning("Unable to retrain: mlboard is absent")


def _get_fps(**kwargs):
    fps = 30.
    if kwargs.get('metadata'):
        fps = int(round(kwargs['metadata']['fps']))
    return fps


def log_recognition(rgb_frame, ret, **kwargs):
    fps = _get_fps(**kwargs)

    current_time = float(frame_num) / fps

    # Log all in text
    log_file = os.path.join(PARAMS['logdir'], 'log.txt')
    str_labels = ret['labels'].astype(str)
    with open(log_file, 'a+') as f:
        msg = '{:.6f} {}\n'.format(current_time, ','.join(str_labels))
        f.write(msg)

    # Save unknowns
    if PARAMS['inference_fps'] == 0:
        relative_fps = fps
    else:
        relative_fps = float(fps) / PARAMS['inference_fps']
    log_freq = 1. / fps
    if int(log_freq * frame_num) - int(log_freq * (frame_num - relative_fps)) == 1:
        not_detected_indices = [i for i, e in enumerate(str_labels) if e == '']
        # not_detected_imgs = imgs[not_detected_indices].transpose(0, 2, 3, 1)
        not_detected_imgs = images.get_images(
            rgb_frame, ret['boxes'][not_detected_indices].astype(int), normalization=None,
        )

        # find free dir
        global unknown_num
        for img in not_detected_imgs:
            dir_name = os.path.join(PARAMS['logdir'], 'unknown_{:05d}'.format(unknown_num))
            while os.path.exists(dir_name):
                unknown_num += 1
                dir_name = os.path.join(PARAMS['logdir'], 'unknown_{:05d}'.format(unknown_num))

            os.mkdir(dir_name)

            # save unknown image
            image_file = os.path.join(dir_name, 'image.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_file, img)


def build_table(frame, face_infos):
    result = {}
    table = []
    clarify_enabled, _ = _clarify_enabled()

    for processed in face_infos:
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
            if 'url' in processed.meta:
                row_data['url'] = processed.meta['url']

        if clarify_enabled:

            image_clarify = {
                'alternate': True,
            }

            if processed.state == detector.NOT_DETECTED:
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
                        if 'url' in m:
                            cl_cls['url'] = m['url']
                    image_clarify['values'].append(cl_cls)

            row_data['image_clarify'] = image_clarify

        table.append(row_data)

    result['table_output'] = json.dumps(table)
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
            'name': 'url',
            'label': 'URL'
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
            'fields': ['name', 'position', 'company', 'url', 'face']
        })
    result['table_meta'] = json.dumps(meta)
    return result
