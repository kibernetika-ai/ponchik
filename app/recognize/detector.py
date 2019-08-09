import glob
import json
import os
import pickle

import cv2
from ml_serving.drivers import driver
import numpy as np
import six
from scipy.spatial import distance
from sklearn import svm, neighbors

from app.recognize import classifiers, defaults
from app.tools import images, add_normalization_args, utils
from app import tools
import math

DETECTED = 1
NOT_DETECTED = 2
WRONG_FACE_POS = 3


class DetectorClassifiers:
    def __init__(self):
        self.classifiers = []
        self.classifier_names = []
        self.embedding_sizes = []
        self.embeddings = None
        self.plain_embeddings = None
        self.class_index = None
        self.class_names = None
        self.class_stats = None


def add_detector_args(parser):
    parser.add_argument(
        '--threshold',
        type=float,
        default=defaults.THRESHOLD,
        help='Threshold for detecting faces',
    )
    parser.add_argument(
        '--person_detection_driver',
        default=defaults.PERSON_DETECTION_DRIVER,
        choices=[defaults.PERSON_DETECTION_DRIVER, "tensorflow"],
        help='person-detection driver',
    )
    parser.add_argument(
        '--person_detection_path',
        default=defaults.PERSON_DETECTION_PATH,
        help='Path to person-detection model',
    )
    parser.add_argument(
        '--person_threshold',
        type=float,
        default=defaults.PERSON_THRESHOLD,
        help='Threshold for detecting persons',
    )
    parser.add_argument(
        '--debug',
        help='Full debug output for each detected face.',
        action='store_true',
    )
    add_normalization_args(parser)


def detector_args(args):
    head_pose_driver = None
    if args.head_pose_path is not None:
        if os.path.isfile(args.head_pose_path):
            from ml_serving.drivers import driver
            tools.print_fun("Load HEAD POSE ESTIMATION model")
            drv = driver.load_driver('openvino')
            head_pose_driver = drv()
            head_pose_driver.load_model(args.head_pose_path)
        else:
            tools.print_fun("head-pose-estimation openvino model is not found, skipped")

    face_driver = None
    if args.face_detection_path is not None:
        if os.path.isfile(args.face_detection_path):
            from ml_serving.drivers import driver
            tools.print_fun("Load FACE DETECTION model %s" % args.face_detection_path)
            drv = driver.load_driver('openvino')
            face_driver = drv()
            face_driver.load_model(args.face_detection_path)
        else:
            tools.print_fun("face-detection openvino model is not found, skipped")

    facenet_driver = None
    if not args.without_facenet and args.model_path is not None:
        if os.path.isfile(args.model_path):
            from ml_serving.drivers import driver
            tools.print_fun("Load FACENET model")
            drv = driver.load_driver('openvino')
            facenet_driver = drv()
            facenet_driver.load_model(args.model_path)
        else:
            tools.print_fun("facenet openvino model is not found, skipped")

    person_driver = None
    if args.person_detection_path is not None:
        if args.person_detection_driver == "openvino" and os.path.isfile(args.person_detection_path) \
                or args.person_detection_driver == "tensorflow" and os.path.isdir(args.person_detection_path):
            from ml_serving.drivers import driver
            tools.print_fun("Load PERSON DETECTION model %s (driver %s)" %
                            (args.person_detection_path, args.person_detection_driver))
            drv = driver.load_driver(args.person_detection_driver)
            person_driver = drv()
            person_driver.load_model(args.person_detection_path)
        else:
            tools.print_fun("person-detection model is not found, skipped")

    multi_detect = None
    if args.multi_detect:
        multi_detect = [int(i) for i in args.multi_detect.split(',')]

    return Detector(
        face_driver=face_driver,
        facenet_driver=facenet_driver,
        head_pose_driver=head_pose_driver,
        person_driver=person_driver,
        classifiers_dir=None if args.without_classifiers else args.classifiers_dir,
        threshold=args.threshold,
        person_threshold=args.person_threshold,
        min_face_size=args.min_face_size,
        debug=args.debug,
        process_not_detected=args.process_not_detected,
        account_head_pose=not args.head_pose_not_account,
        multi_detect=multi_detect,
        normalization=args.normalization,
    )


class FaceInfo:
    __slots__ = [
        'bbox', 'person_bbox', 'state', 'label', 'overlay_label',
        'prob', 'face_prob', 'classes', 'classes_meta', 'meta',
        'looks_like', 'embedding', 'head_pose', 'last_seen'
    ]

    def __init__(
            self,
            bbox=None,
            person_bbox=None,
            state=NOT_DETECTED,
            label='',
            overlay_label='',
            prob=0,
            face_prob=0.0,
            classes=None,
            classes_meta=None,
            meta=None,
            looks_like=None,
            embedding=None,
            head_pose=None,
    ):
        self.bbox = bbox
        self.person_bbox = person_bbox
        self.state = state
        self.label = label
        self.overlay_label = overlay_label
        self.prob = prob
        self.face_prob = face_prob
        self.classes = classes
        self.classes_meta = classes_meta
        self.meta = meta
        self.looks_like = looks_like if looks_like else []
        self.embedding = embedding
        self.head_pose = head_pose
        self.last_seen = None

    def is_detected(self):
        return self.state == DETECTED


class PersonInfo:
    def __init__(
            self,
            bbox=None,
            prob=0,

    ):
        self.bbox = bbox
        self.prob = prob


class Detector(object):
    def __init__(
            self,
            face_driver=None,
            facenet_driver=None,
            classifiers_dir=defaults.CLASSIFIERS_DIR,
            head_pose_driver=None,
            head_pose_thresholds=defaults.HEAD_POSE_THRESHOLDS,
            threshold=defaults.THRESHOLD,
            person_threshold=defaults.PERSON_THRESHOLD,
            min_face_size=defaults.MIN_FACE_SIZE,
            person_driver=None,
            debug=defaults.DEBUG,
            process_not_detected=False,
            account_head_pose=True,
            multi_detect=None,
            normalization=defaults.NORMALIZATION,
            only_distance=False,
    ):
        self._initialized = False
        self.face_driver: driver.ServingDriver = face_driver
        self.facenet_driver: driver.ServingDriver = facenet_driver
        self.classifiers_dir = classifiers_dir

        self.head_pose_driver: driver.ServingDriver = head_pose_driver
        self.head_pose_thresholds = head_pose_thresholds
        self.head_pose_yaw = "angle_y_fc"
        self.head_pose_pitch = "angle_p_fc"
        self.head_pose_roll = "angle_r_fc"
        self.account_head_pose = account_head_pose

        self.person_driver: driver.ServingDriver = person_driver
        self.person_threshold = person_threshold

        self.use_classifiers = False
        self.only_distance = only_distance
        self.kd_tree = None
        self.normalization = normalization
        self.classifiers = DetectorClassifiers()
        self.threshold = threshold
        self.multi_detect = multi_detect
        # self.multi_detect = None
        self.min_face_size = min_face_size
        self.min_face_area = self.min_face_size ** 2
        self.debug = debug
        self.classes = []
        self.meta = {}
        self.classes_previews = {}
        self.process_not_detected = process_not_detected
        self.not_detected_embs = []
        self.detected_names = []

        self.current_frame_faces = []
        self.current_frame_persons = []

    def init(self):
        if self._initialized:
            return
        self._initialized = True

        self.load_classifiers()

    def load_classifiers(self):
        self.classes_previews = {}
        self.use_classifiers = False
        if self.facenet_driver is None or self.classifiers_dir is None:
            return

        loaded_classifiers = glob.glob(classifiers.classifier_filename(self.classifiers_dir, '*'))

        if len(loaded_classifiers) > 0:
            new = DetectorClassifiers()
            for clfi, clf in enumerate(loaded_classifiers):
                # Load classifier
                with open(clf, 'rb') as f:
                    tools.print_fun('Load CLASSIFIER %s' % clf)
                    opts = {'file': f}
                    if six.PY3:
                        opts['encoding'] = 'latin1'
                    (clf, class_names, class_stats) = pickle.load(**opts)
                    if isinstance(clf, svm.SVC):
                        embedding_size = clf.shape_fit_[1]
                        classifier_name = "SVM"
                        classifier_name_log = "SVM classifier"
                    elif isinstance(clf, neighbors.KNeighborsClassifier):
                        embedding_size = clf._fit_X.shape[1]
                        classifier_name = "kNN"
                        classifier_name_log = "kNN (neighbors %d) classifier" % clf.n_neighbors
                    else:
                        # try embedding_size = 512
                        embedding_size = 512
                        classifier_name = "%d" % clfi
                        classifier_name_log = type(clf)
                    tools.print_fun('Loaded %s, embedding size: %d' % (classifier_name_log, embedding_size))
                    if new.class_names is None:
                        new.class_names = class_names
                        self.classes = class_names
                    elif class_names != new.class_names:
                        raise RuntimeError("Different class names in classifiers")
                    if new.class_stats is None:
                        new.class_stats = class_stats
                    # elif class_stats != new.class_stats:
                    #     raise RuntimeError("Different class stats in classifiers")
                    new.classifier_names.append(classifier_name)
                    new.embedding_sizes.append(embedding_size)
                    new.classifiers.append(clf)

            self.classifiers = new
            self.use_classifiers = True

            embs_filename = classifiers.embeddings_filename(self.classifiers_dir)
            if os.path.isfile(embs_filename):
                with open(embs_filename, 'rb') as r:
                    opts = {'file': r}
                    if six.PY3:
                        opts['encoding'] = 'latin1'
                    new.embeddings = pickle.load(**opts)

                    size = 0
                    for i, cls in enumerate(self.classifiers.embeddings):
                        for embs in self.classifiers.embeddings[cls].values():
                            size += len(embs)

                    plain_embeddings = np.zeros([size, embedding_size])
                    class_index = []
                    emb_i = 0
                    for i, cls in enumerate(self.classifiers.embeddings):
                        for embs in self.classifiers.embeddings[cls].values():
                            for emb in embs:
                                plain_embeddings[emb_i] = emb
                                emb_i += 1
                                class_index.append(cls)
                                pass

                    self.classifiers.plain_embeddings = plain_embeddings
                    self.classifiers.class_index = class_index

        meta_file = os.path.join(self.classifiers_dir, classifiers.META_FILENAME)
        self.meta = {}
        if os.path.isfile(meta_file):
            tools.print_fun("Load metadata...")
            with open(meta_file, 'r') as mr:
                self.meta = json.load(mr)

    def detect_faces(self, frame, threshold=0.5, split_counts=None):
        boxes = self._detect_faces(frame, threshold)
        if not split_counts:
            return boxes

        def add_box(b):
            for b0 in boxes:
                if utils.box_intersection(b0, b) > 0.3:
                    return
            boxes.resize((boxes.shape[0] + 1, boxes.shape[1]), refcheck=False)
            boxes[-1] = b

        for split_count in split_counts:
            size_multiplier = 2. / (split_count + 1)
            xstep = int(frame.shape[1] / (split_count + 1))
            ystep = int(frame.shape[0] / (split_count + 1))

            xlimit = int(np.ceil(frame.shape[1] * (1 - size_multiplier)))
            ylimit = int(np.ceil(frame.shape[0] * (1 - size_multiplier)))
            for x in range(0, xlimit, xstep):
                for y in range(0, ylimit, ystep):
                    y_border = min(frame.shape[0], int(np.ceil(y + frame.shape[0] * size_multiplier)))
                    x_border = min(frame.shape[1], int(np.ceil(x + frame.shape[1] * size_multiplier)))
                    crop = frame[y:y_border, x:x_border, :]

                    box_candidates = self._detect_faces(crop, threshold, (x, y))

                    for b in box_candidates:
                        add_box(b)

        return boxes

    def _detect_faces(self, frame, threshold=0.5, offset=(0, 0)):
        boxes = self._detect_openvino(self.face_driver, frame, threshold, offset)
        if boxes is not None:
            boxes = boxes[(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) >= self.min_face_area]
        return boxes

    def detect_persons(self, frame, threshold=0.5):
        if self.person_driver is None:
            return None
        elif self.person_driver.driver_name == "openvino":
            return self._detect_openvino(self.person_driver, frame, threshold)
        elif self.person_driver.driver_name == "tensorflow":
            return self._detect_tensorflow(self.person_driver, frame, threshold)
        else:
            return None

    @staticmethod
    def _detect_openvino(drv, frame, threshold=0.5, offset=(0, 0)):
        if drv is None:
            return None
        # Get boxes shaped [N, 5]:
        # xmin, ymin, xmax, ymax, confidence
        input_name, input_shape = list(drv.inputs.items())[0]
        output_name = list(drv.outputs)[0]
        inference_frame = cv2.resize(frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
        inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
        outputs = drv.predict({input_name: inference_frame})
        output = outputs[output_name]
        output = output.reshape(-1, 7)
        bboxes_raw = output[output[:, 2] > threshold]
        # Extract 5 values
        boxes = bboxes_raw[:, 3:7]
        confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
        boxes = np.concatenate((boxes, confidence), axis=1)
        # Assign confidence to 4th
        # boxes[:, 4] = bboxes_raw[:, 2]
        boxes[:, 0] = boxes[:, 0] * frame.shape[1] + offset[0]
        boxes[:, 2] = boxes[:, 2] * frame.shape[1] + offset[0]
        boxes[:, 1] = boxes[:, 1] * frame.shape[0] + offset[1]
        boxes[:, 3] = boxes[:, 3] * frame.shape[0] + offset[1]
        return boxes

    @staticmethod
    def _detect_tensorflow(drv, frame, threshold=0.5):
        input_name, input_shape = list(drv.inputs.items())[0]
        inference_frame = np.expand_dims(frame, axis=0)
        outputs = drv.predict({input_name: inference_frame})
        boxes = outputs["detection_boxes"].copy().reshape([-1, 4])
        scores = outputs["detection_scores"].copy().reshape([-1])
        classes = np.int32((outputs["detection_classes"].copy())).reshape([-1])
        scores = scores[np.where(scores > threshold)]
        boxes = boxes[:len(scores)]
        classes = classes[:len(scores)]
        boxes = boxes[classes == 1]
        scores = scores[classes == 1]
        boxes[:, 0] *= frame.shape[0]
        boxes[:, 1] *= frame.shape[1]
        boxes[:, 2] *= frame.shape[0]
        boxes[:, 3] *= frame.shape[1]
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]].astype(int)

        confidence = np.expand_dims(scores, axis=0).transpose()
        boxes = np.concatenate((boxes, confidence), axis=1)

        return boxes

    def inference_facenet(self, img):
        outputs = self.facenet_driver.predict({list(self.facenet_driver.inputs)[0]: img})
        output = outputs[list(self.facenet_driver.outputs)[0]]
        return output

    def process_output(self, output, bbox, person_bbox=None, face_prob=None, label='', overlay_label='',
                       use_classifiers=True):

        if face_prob is None:
            face_prob = bbox[4]

        if not self.use_classifiers or not use_classifiers:
            return FaceInfo(
                bbox=bbox[:4].astype(int),
                person_bbox=person_bbox,
                state=NOT_DETECTED,
                label=label,
                overlay_label=overlay_label,
                prob=0,
                face_prob=face_prob,
                classes=[],
                classes_meta={},
                meta=None,
                looks_like=[],
                head_pose=None
            )

        looks_likes = []

        if not self.only_distance:
            out = self.recognize_classifier(output)
            overlay_label_str, summary_overlay_label, classes, stored_class_name, mean_prob, detected = out
        else:
            out = self.recognize_distance(output)
            overlay_label_str, summary_overlay_label, classes, stored_class_name, mean_prob, detected = out
        meta = self.meta[stored_class_name] if detected and stored_class_name in self.meta else None

        classes_meta = {}
        for cl in classes:
            cl_ = cl.replace(" ", "_")
            if cl_ in self.meta:
                classes_meta[cl_] = self.meta[cl_]

        return FaceInfo(
            bbox=bbox[:4].astype(int),
            person_bbox=person_bbox,
            state=DETECTED if detected else NOT_DETECTED,
            label=summary_overlay_label,
            overlay_label=overlay_label_str,
            prob=mean_prob,
            face_prob=face_prob,
            classes=classes,
            classes_meta=classes_meta,
            meta=meta,
            looks_like=[self.classifiers.class_names[ll] for ll in looks_likes],
            embedding=output,
        )

    def recognize_distance(self, output):
        output = output.reshape([-1, 512])
        # min_dist = 10e10
        threshold = 0.35
        if self.kd_tree is None:
            print('building tree...')
            # neighbors.DistanceMetric()
            # self.kd_tree = neighbors.BallTree(self.classifiers.plain_embeddings, metric=distance.cosine)
            embeddings = (self.classifiers.plain_embeddings + 1.) / 2.
            self.kd_tree = neighbors.KDTree(embeddings, metric='euclidean')
        # __import__('ipdb').set_trace()
        detected_class = None

        dist, idx = self.kd_tree.query((output + 1.) / 2., k=1)
        dist = dist[0][0]
        idx = idx[0][0]
        if dist < threshold:
            detected_class = self.classifiers.class_index[idx]
        # for cls in self.classifiers.embeddings:
        #     for embs in self.classifiers.embeddings[cls].values():
        #         for emb in embs:
        #             dist = distance.cosine(output, emb)
        #             if dist < min_dist and dist < threshold:
        #                 min_dist = dist
        #                 detected_class = cls

        if detected_class:
            prob = 1 - (max(dist, 0.2) - 0.2)
            summary_overlay_label = detected_class
            if self.debug:
                overlay_label_str = "%.1f%% %s" % (prob * 100, summary_overlay_label)
                overlay_label_str += "\ndistance: %.3f" % dist
            else:
                overlay_label_str = "%.1f%% %s" % (prob * 100, summary_overlay_label)
            classes = [detected_class]
            detected = True
        else:
            summary_overlay_label = ''
            if self.debug:
                overlay_label_str = "Summary: not detected; distance: %.3f" % dist
            else:
                overlay_label_str = ''
            classes = []
            detected = False
            prob = 0.

        return overlay_label_str, summary_overlay_label, classes, detected_class, prob, detected

    def recognize_classifier(self, output):
        detected_indices = []
        label_strings = []
        probs = []
        classes = []
        prob_detected = True
        summary_overlay_label = ""
        looks_likes = []
        for clfi, clf in enumerate(self.classifiers.classifiers):
            try:
                output = output.reshape(1, self.classifiers.embedding_sizes[clfi])
                predictions = clf.predict_proba(output)
            except ValueError as e:
                # Can not reshape
                tools.print_fun("ERROR: Output from graph doesn't consistent with classifier model: %s" % e)
                continue

            best_class_indices = np.argmax(predictions, axis=1)

            # print('??', predictions, best_class_indices)

            if isinstance(clf, neighbors.KNeighborsClassifier):
                def process_index(idx):
                    cnt = self.classifiers.class_stats[best_class_indices[idx]]['embeddings']
                    (closest_distances, neighbors_indices) = clf.kneighbors(output, n_neighbors=cnt)
                    eval_values = closest_distances[:, 0]

                    candidates = [clf._y[i] for i in neighbors_indices[0]]
                    counts = {cl: candidates.count(cl) for cl in set(candidates)}
                    max_candidate = sorted(counts.items(), reverse=True, key=lambda x: x[1])[0]

                    if best_class_indices[idx] != max_candidate[0] and max_candidate[1] > len(candidates) // 2:
                        # tools.print_fun(
                        #     "Changed candidate from %s to %s" % (
                        #         self.classifiers.class_names[best_class_indices[idx]],
                        #         self.classifiers.class_names[max_candidate[0]]
                        #     )
                        # )
                        best_class_indices[idx] = max_candidate[0]

                    min_distance = None
                    filename = None
                    if self.classifiers.embeddings is not None:
                        min_distance = 10e10
                        cls_name = self.classifiers.class_names[best_class_indices[idx]].replace(' ', '_')
                        if cls_name in self.classifiers.embeddings:
                            cls_embs = self.classifiers.embeddings[cls_name]
                            for img in cls_embs:
                                embs = cls_embs[img]
                                for emb in embs:
                                    d = distance.cosine(output, emb)
                                    if min_distance > d:
                                        min_distance = d
                                        filename = os.path.split(img)[1]
                                # print('!! len', img, len(embs))
                                # pass
                        # print('!!!', self.classifiers.class_names[best_class_indices[idx]])

                    ttl_cnt = counts[best_class_indices[idx]]

                    # probability:
                    # total matched embeddings
                    # less than 25% is 0%, more than 75% is 100%
                    # multiplied by distance coefficient:
                    # 0.5 and less is 100%, 1 and more is 0%
                    prob = max(0, min(1, 2 * ttl_cnt / cnt - .5)) * max(0, min(1, 2 - eval_values[idx] * 2))
                    looks_like = set(candidates)
                    looks_like.remove(best_class_indices[idx])
                    if not len(looks_like):
                        looks_like = []
                    debug_label = '%.3f %d/%d' % (
                        eval_values[idx],
                        ttl_cnt, cnt,
                    )
                    if min_distance is not None or filename is not None:
                        debug_label = '\n   {}'.format(debug_label)
                        if min_distance is not None:
                            debug_label += '\n   min cosine dist: {0:.3f}'.format(min_distance)
                        if filename is not None:
                            debug_label += '\n   from image {}'.format(filename)
                        debug_label = '{}\n'.format(debug_label)
                    return prob, debug_label, looks_like

            elif isinstance(clf, svm.SVC):
                def process_index(idx):
                    eval_values = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    return max(0, min(1, eval_values[idx] * 10)), '%.1f%%' % (eval_values[idx] * 100), []
            else:
                tools.print_fun("ERROR: Unsupported model type: %s" % type(clf))
                continue

            for i in range(len(best_class_indices)):
                prob, label_debug, looks_like = process_index(i)
                overlay_label = self.classifiers.class_names[best_class_indices[i]]
                detected_indices.append(best_class_indices[i])
                summary_overlay_label = overlay_label
                probs.append(prob)
                if prob <= 0:
                    prob_detected = False
                classifier_name = self.classifiers.classifier_names[clfi]
                label_debug_info = \
                    '%s: %.1f%% %s (%s)' % (classifier_name, prob * 100, overlay_label, label_debug)
                if self.debug:
                    label_strings.append(label_debug_info)
                elif len(label_strings) == 0:
                    label_strings.append(overlay_label)
                classes.append(overlay_label)
                looks_likes.extend(looks_like)
                if len(looks_likes):
                    looks_likes = list(set(looks_likes))

        # detected if all classes are the same, and all probs are more than 0
        detected = len(set(detected_indices)) == 1 and prob_detected
        mean_prob = sum(probs) / len(probs) if detected else 0
        if not detected:
            candidates = []
            new_probs = []
            for i in range(len(probs)):
                if probs[i] >= 0.35:
                    candidates.append(detected_indices[i])
                    new_probs.append(probs[i])

            detected = len(set(candidates)) == 1 and prob_detected
            mean_prob = sum(new_probs) / len(new_probs) if detected else 0
            if detected:
                if not self.debug:
                    label_strings[0] = self.classifiers.class_names[candidates[0]]
                summary_overlay_label = self.classifiers.class_names[candidates[0]]

        if self.debug:
            if detected:
                label_strings.append("Summary: %.1f%% %s" % (mean_prob * 100, summary_overlay_label))
            else:
                label_strings.append("Summary: not detected")

        if detected:
            classes = [summary_overlay_label]
        else:
            summary_overlay_label = ''

        overlay_label_str = ""
        if self.debug:
            if len(label_strings) > 0:
                overlay_label_str = "\n".join(label_strings)
        elif detected:
            overlay_label_str = label_strings[0]

        stored_class_name = self.classifiers.class_names[detected_indices[0]].replace(" ", "_")
        return overlay_label_str, summary_overlay_label, classes, stored_class_name, mean_prob, detected


    def wrong_pose_indices(self, bgr_frame, boxes):
        if self.head_pose_driver is None:
            return []
        if boxes is None or len(boxes) == 0:
            return []

        imgs = np.stack(images.get_images(bgr_frame, boxes, 60, 0, normalization=None))
        outputs = self.head_pose_driver.predict({'data': imgs.transpose([0, 3, 1, 2])})

        yaw = outputs[self.head_pose_yaw].reshape([-1])
        pitch = outputs[self.head_pose_pitch].reshape([-1])
        roll = outputs[self.head_pose_roll].reshape([-1])

        # Return shape [N, 3] as a result
        return np.array([yaw, pitch, roll]).transpose()

    def wrong_pose_skips(self, poses):
        skips = set()
        # print(yaw, pitch, roll)
        for i, [y, p, r] in enumerate(poses):
            if (np.abs(y) > self.head_pose_thresholds[0]
                    or np.abs(p) > self.head_pose_thresholds[1]
                    or np.abs(r) > self.head_pose_thresholds[2]):
                skips.add(i)

        return skips

    # def skip_wrong_pose_indices(self, bgr_frame, boxes):
    #     if self.head_pose_driver is None:
    #         return set(), []
    #
    #     if boxes is None or len(boxes) == 0:
    #         return set(), []
    #
    #     imgs = np.stack(images.get_images(bgr_frame, boxes, 60, 0, do_prewhiten=False))
    #     outputs = self.head_pose_driver.predict({'data': imgs.transpose([0, 3, 1, 2])})
    #
    #     yaw = outputs[self.head_pose_yaw].reshape([-1])
    #     pitch = outputs[self.head_pose_pitch].reshape([-1])
    #     roll = outputs[self.head_pose_roll].reshape([-1])
    #
    #     skips = set()
    #     # print(yaw, pitch, roll)
    #     for i, (y, p, r) in enumerate(zip(yaw, pitch, roll)):
    #         if (np.abs(y) > self.head_pose_thresholds[0]
    #                 or np.abs(p) > self.head_pose_thresholds[1]
    #                 or np.abs(r) > self.head_pose_thresholds[2]):
    #             skips.add(i)
    #
    #     # Return shape [N, 3] as a result
    #     return skips, np.array([yaw, pitch, roll]).transpose()

    def process_frame(self, frame, overlays=True,
                      stored_faces: [FaceInfo] = None,
                      stored_persons: [PersonInfo] = None):
        bboxes = []
        poses = []
        imgs = []
        embeddings = None
        face_probs = None
        # labels = None
        # overlay_labels = None
        persons_bboxes = []
        persons_probs = []

        person_bbox = None

        if stored_faces is not None:
            embeddings = []
            face_probs = []
            labels = []
            overlay_labels = []
            # todo add from h5
            for d in stored_faces:
                bboxes.append(d.bbox)
                # poses.append(d.head_pose)
                embeddings.append(d.embedding)
                face_probs.append(d.face_prob)
                labels.append(d.label)
                overlay_labels.append(d.overlay_label)
                imgs.append(None)
        else:
            bboxes = self.detect_faces(frame, self.threshold, self.multi_detect)
            poses = self.wrong_pose_indices(frame, bboxes)
            imgs = images.get_images(frame, bboxes, normalization=self.normalization)

        if stored_persons is not None:
            persons_bboxes = []
            persons_probs = []
            for d in stored_persons:
                persons_bboxes.append(d.bbox)
                persons_probs.append(d.prob)
        else:
            persons_bboxes_raw = self.detect_persons(frame, self.person_threshold)
            if persons_bboxes_raw is not None:
                persons_bboxes = persons_bboxes_raw[:, :4].astype(int)
                persons_probs = persons_bboxes_raw[:, 4]
                stored_persons = []
                for i, b in enumerate(persons_bboxes):
                    stored_persons.append(PersonInfo(bbox=b, prob=persons_probs[i]))

        skips = self.wrong_pose_skips(poses)
        # skips, poses = self.skip_wrong_pose_indices(frame, bboxes)

        faces = []
        not_detected_embs = []
        detected_names = []

        # if self.use_classifiers:

        for img_idx, img in enumerate(imgs):
            # set face probability from saved data
            face_prob = face_probs[img_idx] if face_probs else None

            # Infer
            # t = time.time()
            # Convert BGR to RGB
            if img_idx not in skips or not self.account_head_pose:

                if embeddings:
                    output = embeddings[img_idx]
                else:
                    img = img[:, :, ::-1]
                    img = img.transpose([2, 0, 1]).reshape([1, 3, 160, 160])
                    output = self.inference_facenet(img)
                # LOG.info('facenet: %.3fms' % ((time.time() - t) * 1000))
                # output = output[facenet_output]

                if stored_faces is not None:
                    face = stored_faces[img_idx]
                else:
                    person_bbox = None
                    if persons_bboxes is not None:
                        for pb in persons_bboxes:
                            if utils.box_includes(pb, bboxes[img_idx]):
                                person_bbox = pb[:4].astype(int)
                                break
                    face = self.process_output(
                        output,
                        bboxes[img_idx],
                        person_bbox=person_bbox,
                        # face_prob=face_prob,
                        # label=labels[img_idx] if labels is not None else '',
                        # overlay_label=overlay_labels[img_idx] if overlay_labels is not None else '',
                        # use_classifiers=data is None,
                    )
                    # face.person_bbox = persons_bboxes

                if poses is not None and len(poses) > img_idx:
                    face.head_pose = poses[img_idx]
                face.embedding = output.reshape([-1])

                if img_idx in skips:
                    face.state = WRONG_FACE_POS

                if self.process_not_detected:
                    if face.state == NOT_DETECTED:
                        not_detected_embs.append(output)

            else:
                face = FaceInfo(
                    bbox=bboxes[img_idx][:4].astype(int),
                    person_bbox=person_bbox,
                    state=WRONG_FACE_POS,
                    label='',
                    overlay_label='',
                    prob=0,
                    face_prob=face_prob if face_prob else bboxes[img_idx][4],
                    classes=[],
                    classes_meta={},
                    meta=None,
                    looks_like=[],
                    head_pose=poses[img_idx] if poses is not None and len(poses) > img_idx else None,
                )

            if face.state == DETECTED:
                if face.classes and len(face.classes) > 0:
                    detected_names.append(face.classes[0])

            faces.append(face)

        persons = []
        for i, pbbox in enumerate(persons_bboxes):
            persons.append(PersonInfo(bbox=pbbox, prob=persons_probs[i]))

        # else:
        #     for bbox in bboxes:
        #         face = FaceInfo(
        #             bbox=bbox[:4].astype(int),
        #             state=NOT_DETECTED,
        #             label='',
        #             overlay_label='',
        #             prob=0,
        #             face_prob=bbox[4],
        #             classes=[],
        #             classes_meta={},
        #             meta=None,
        #             looks_like=[],
        #             head_pose=None
        #         )
        #         faces.append(face)

        if overlays:
            self.add_overlays(frame, faces=faces, persons=persons)

        if self.process_not_detected:
            self.not_detected_embs.append(not_detected_embs)

        self.detected_names.append(detected_names)

        self.current_frame_faces = faces
        self.current_frame_persons = persons

        return faces

    def add_overlays(self, frame, faces: [FaceInfo] = None, persons: [PersonInfo] = None):
        if persons:
            for person in persons:
                self.add_person_overlay(frame, person)
        if faces:
            for face in faces:
                self.add_face_overlay(frame, face)

    def add_person_overlay(self, frame, person: PersonInfo):
        font_face, font_scale, thickness = Detector._get_text_props(frame)
        if person.bbox is not None:
            pbbox = person.bbox
            cv2.rectangle(
                frame,
                (pbbox[0], pbbox[1]), (pbbox[2], pbbox[3]),  # (left, top), (right, bottom)
                (0, 80, 0),
                thickness,
            )

    def add_face_overlay(self, frame, face: FaceInfo, align_to_right=True):
        """Add box and label overlays on frame
        :param frame: frame in BGR channels order
        :param face: Face info - box, label, embedding, pose etc...
        :param align_to_right: align multistring text block to right
        :return:
        """

        font_face, font_scale, thickness = Detector._get_text_props(frame)

        # bbox = face.bbox.astype(int)
        bbox, pbbox = face.bbox, face.person_bbox
        color = self._color(face.state)
        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]), (bbox[2], bbox[3]),  # (left, top), (right, bottom)
            color,
            thickness * (2 if face.state == DETECTED else 1),
        )

        if face.person_bbox is not None:
            cv2.rectangle(
                frame,
                (pbbox[0], pbbox[1]), (pbbox[2], pbbox[3]),  # (left, top), (right, bottom)
                (0, 100, 0),
                thickness,
            )

        font_inner_padding_w, font_inner_padding_h = 5, 5

        if face.overlay_label is not None and face.overlay_label != "":
            strs = face.overlay_label.split('\n')
            str_w, str_h = 0, 0
            widths = []
            for i, line in enumerate(strs):
                lw, lh = self._get_text_size(frame, line)
                str_w = max(str_w, lw)
                str_h = max(str_h, lh)
                widths.append(lw)
            str_h = int(str_h * 1.6)  # line height

            to_right = bbox[0] + str_w > frame.shape[1] - font_inner_padding_w
            top = max(str_h, bbox[1] - int((len(strs) - 0.5) * str_h))

            for i, line in enumerate(strs):
                if align_to_right:
                    # all align to right box border
                    if to_right:
                        left = (bbox[2] - widths[i] - font_inner_padding_w)
                    else:
                        left = bbox[0] + font_inner_padding_w
                else:
                    # move left each string if it's ending not places on the frame
                    if bbox[0] + widths[i] > frame.shape[1] - font_inner_padding_w:
                        left = frame.shape[1] - widths[i] - font_inner_padding_w
                    else:
                        left = bbox[0] + font_inner_padding_w

                self._put_text(frame, line, left, int(top + i * str_h), color=(0, 0, 0), thickness_mul=3)
                self._put_text(frame, line, left, int(top + i * str_h), color=color)

                if face.classes and len(face.classes) > 0:
                    classes_preview_size = min(
                        str_h * 3,  # size depends on row height
                        int((face.bbox[2] - face.bbox[0] - 10) / len(face.classes) / 1.2),
                        # size depends on bounding box size
                    )
                    i_left = max(0, face.bbox[0])
                    i_top = min(face.bbox[3] + int(classes_preview_size * .1),
                                frame.shape[0] - int(classes_preview_size * 1.1))
                    for cls in face.classes:
                        cv2.rectangle(
                            frame,
                            (i_left, i_top),
                            (i_left + classes_preview_size, i_top + classes_preview_size),
                            color=color,
                            thickness=thickness + 1,
                        )
                        if cls not in self.classes_previews:
                            # tools.print_fun('Init preview for class "%s"' % cls)
                            self.classes_previews[cls] = None
                            cls_img_path = os.path.join(self.classifiers_dir, "previews/%s.png" % cls.replace(" ", "_"))
                            if os.path.isfile(cls_img_path):
                                try:
                                    self.classes_previews[cls] = cv2.imread(cls_img_path)
                                except Exception as e:
                                    tools.print_fun('Error reading preview for "%s": %s' % (cls, e))
                            else:
                                tools.print_fun('Error reading preview for "%s": file not found' % cls)
                        cls_img = self.classes_previews[cls]
                        if cls_img is not None:
                            resized = images.image_resize(cls_img, classes_preview_size, classes_preview_size)
                            try:
                                frame[i_top:i_top + resized.shape[0], i_left:i_left + resized.shape[1]] = resized
                            except Exception as e:
                                tools.print_fun("ERROR: %s" % e)
                        i_left += int(classes_preview_size * 1.2)

    @staticmethod
    def _get_text_props(frame, thickness=None, thickness_mul=None, font_scale=None, font_face=None):
        if font_scale is None or thickness is None:
            frame_avg = (frame.shape[1] + frame.shape[0]) / 2
            if font_scale is None:
                font_scale = frame_avg / 1200
            if thickness is None:
                thickness = int(font_scale * 2)
            if thickness_mul is not None:
                thickness_m = int(thickness * thickness_mul)
                thickness = thickness + 1 if thickness == thickness_m else thickness_m
        if font_face is None:
            font_face = cv2.FONT_HERSHEY_SIMPLEX
        return font_face, font_scale, thickness

    @staticmethod
    def _get_text_size(frame, text, thickness=None, thickness_mul=None, font_scale=None, font_face=None):
        font_face, font_scale, thickness = Detector._get_text_props(
            frame, thickness, thickness_mul, font_scale, font_face
        )
        return cv2.getTextSize(text, font_face, font_scale, thickness)[0]

    @staticmethod
    def _put_text(frame, text, left, top, color, thickness=None, thickness_mul=None,
                  font_scale=None, font_face=None, line_type=cv2.LINE_AA):
        font_face, font_scale, thickness = Detector._get_text_props(
            frame, thickness, thickness_mul, font_scale, font_face
        )
        cv2.putText(frame, text, (left, top), font_face, font_scale, color, thickness=thickness, lineType=line_type)

    @staticmethod
    def _color(state):
        if state == DETECTED:
            return 0, 255, 0
        elif state == WRONG_FACE_POS:
            return 0, 0, 250
        else:
            return 250, 0, 250


def cosine_dist(embeddings1, embeddings2):
    dot = np.sum(np.multiply(embeddings1, embeddings2))
    norm = np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
    similarity = dot/norm
    d = np.arccos(similarity) / math.pi
    return d
