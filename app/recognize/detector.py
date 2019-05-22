import glob
import json
import os
import pickle

import cv2
import numpy as np
import six
from openvino import inference_engine as ie
from sklearn import neighbors
from sklearn import svm

from app.recognize import nets, defaults, classifiers
from app.tools import images, bg_remove, print_fun


DETECTED = 1
NOT_DETECTED = 2
WRONG_FACE_POS = 3


class DetectorClassifiers:
    def __init__(self):
        self.classifiers = []
        self.classifier_names = []
        self.embedding_sizes = []
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
        '--debug',
        help='Full debug output for each detected face.',
        action='store_true',
    )


def detector_args(args):
    serving = None
    if args.head_pose_path is not None:
        if os.path.isfile(args.head_pose_path):
            from ml_serving.drivers import driver
            print_fun("Load HEAD POSE ESTIMATION model")
            drv = driver.load_driver('openvino')
            serving = drv()
            serving.load_model(args.head_pose_path)
        else:
            print_fun("head-pose-estimation openvino model is not found, skipped")
    return Detector(
        device=args.device,
        face_detection_path=args.face_detection_path,
        head_pose_driver=serving,
        model_path=args.model_path,
        classifiers_dir=args.classifiers_dir,
        bg_remove_path=args.bg_remove_path,
        threshold=args.threshold,
        min_face_size=args.min_face_size,
        debug=args.debug,
        process_not_detected=args.process_not_detected,
    )


class Processed:
    def __init__(
            self,
            bbox=None,
            state=NOT_DETECTED,
            label='',
            overlay_label='',
            prob=0,
            classes=None,
            classes_meta=None,
            meta=None,
            looks_like=None,
    ):
        self.bbox = bbox
        self.state = state
        self.label = label
        self.overlay_label = overlay_label
        self.prob = prob
        self.classes = classes
        self.classes_meta = classes_meta
        self.meta = meta
        self.looks_like = looks_like if looks_like else []


class Detector(object):
    def __init__(
            self,
            device=defaults.DEVICE,
            face_detection_path=defaults.FACE_DETECTION_PATH,
            head_pose_path=defaults.HEAD_POSE_PATH,
            model_path=defaults.MODEL_PATH,
            classifiers_dir=defaults.CLASSIFIERS_DIR,
            bg_remove_path=None,
            head_pose_driver=None,
            head_pose_thresholds=defaults.HEAD_POSE_THRESHOLDS,
            threshold=defaults.THRESHOLD,
            min_face_size=defaults.MIN_FACE_SIZE,
            debug=defaults.DEBUG,
            loaded_plugin=None,
            facenet_exec_net=None,
            process_not_detected=False,
    ):
        self._initialized = False
        self.device = device
        self.face_detection_path = face_detection_path
        self.model_path = model_path
        self.classifiers_dir = classifiers_dir
        self.bg_remove_path = bg_remove_path
        self.bg_remove = None
        self.head_pose_driver = head_pose_driver
        self.head_pose_thresholds = head_pose_thresholds
        self.head_pose_yaw = "angle_y_fc"
        self.head_pose_pitch = "angle_p_fc"
        self.head_pose_roll = "angle_r_fc"
        self.use_classifiers = False
        self.classifiers = DetectorClassifiers()
        self.threshold = threshold
        self.min_face_size = min_face_size
        self.min_face_area = self.min_face_size ** 2
        self.debug = debug
        self.loaded_plugin = loaded_plugin
        self.classes = []
        self.meta = {}
        self.classes_previews = {}
        self.face_net = facenet_exec_net
        self.process_not_detected = process_not_detected
        self.not_detected_embs = []
        self.detected_names = []

    def init(self):
        if self._initialized:
            return
        self._initialized = True

        extensions = os.environ.get('INTEL_EXTENSIONS_PATH')
        if self.loaded_plugin is not None:
            plugin = self.loaded_plugin
        else:
            plugin = ie.IEPlugin(device=self.device)
            if extensions and "CPU" in self.device:
                for ext in extensions.split(':'):
                    print_fun("LOAD extension from {}".format(ext))
                    plugin.add_cpu_extension(ext)

        self.loaded_plugin = plugin

        print_fun('Load FACE DETECTION')
        weights_file = self.face_detection_path[:self.face_detection_path.rfind('.')] + '.bin'
        net = ie.IENetwork(self.face_detection_path, weights_file)
        self.face_detect = nets.FaceDetect(plugin, net)

        if self.model_path:
            print_fun('Load FACENET model')
            model_file = self.model_path
            weights_file = self.model_path[:self.model_path.rfind('.')] + '.bin'
            net = ie.IENetwork(model_file, weights_file)
            self.facenet_input = list(net.inputs.keys())[0]
            outputs = list(iter(net.outputs))
            self.facenet_output = outputs[0]
            if self.face_net is None:
                self.face_net = plugin.load(net)

        self.bg_remove = bg_remove.get_driver(self.bg_remove_path)

        self.load_classifiers()

    def load_classifiers(self):

        self.classes_previews = {}

        if not bool(self.model_path):
            return

        self.use_classifiers = False

        loaded_classifiers = glob.glob(os.path.join(self.classifiers_dir, "*.pkl"))

        if len(loaded_classifiers) > 0:
            new = DetectorClassifiers()
            for clfi, clf in enumerate(loaded_classifiers):
                # Load classifier
                with open(clf, 'rb') as f:
                    print_fun('Load CLASSIFIER %s' % clf)
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
                    print_fun('Loaded %s, embedding size: %d' % (classifier_name_log, embedding_size))
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

        meta_file = os.path.join(self.classifiers_dir, classifiers.META_FILENAME)
        self.meta = {}
        if os.path.isfile(meta_file):
            print_fun("Load metadata...")
            with open(meta_file, 'r') as mr:
                self.meta = json.load(mr)

    def detect_faces(self, frame, threshold=0.5):
        if self.bg_remove is not None:
            bounding_boxes_frame = self.bg_remove.apply_mask(frame)
        else:
            bounding_boxes_frame = frame
        detected = self._openvino_detect(self.face_detect, bounding_boxes_frame, threshold)

        return detected[(detected[:, 3] - detected[:, 1]) * (detected[:, 2] - detected[:, 0]) >= self.min_face_area]

    def inference_facenet(self, img):
        output = self.face_net.infer(inputs={self.facenet_input: img})
        return output[self.facenet_output]

    def process_output(self, output, bbox):
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
                print_fun("ERROR: Output from graph doesn't consistent with classifier model: %s" % e)
                continue

            best_class_indices = np.argmax(predictions, axis=1)

            if isinstance(clf, neighbors.KNeighborsClassifier):

                def process_index(idx):
                    cnt = self.classifiers.class_stats[best_class_indices[idx]]['embeddings']
                    (closest_distances, neighbors_indices) = clf.kneighbors(output, n_neighbors=cnt)
                    eval_values = closest_distances[:, 0]

                    candidates = [clf._y[i] for i in neighbors_indices[0]]
                    counts = {cl: candidates.count(cl) for cl in set(candidates)}
                    max_candidate = sorted(counts.items(), reverse=True, key=lambda x: x[1])[0]

                    if best_class_indices[idx] != max_candidate[0] and max_candidate[1] > len(candidates) // 2:
                        # print_fun(
                        #     "Changed candidate from %s to %s" % (
                        #         self.classifiers.class_names[best_class_indices[idx]],
                        #         self.classifiers.class_names[max_candidate[0]]
                        #     )
                        # )
                        best_class_indices[idx] = max_candidate[0]

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
                    return prob, '%.3f %d/%d' % (
                        eval_values[idx],
                        ttl_cnt, cnt,
                    ), looks_like

            elif isinstance(clf, svm.SVC):

                def process_index(idx):
                    eval_values = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    return max(0, min(1, eval_values[idx] * 10)), '%.1f%%' % (eval_values[idx] * 100), []

            else:

                print_fun("ERROR: Unsupported model type: %s" % type(clf))
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
        meta = self.meta[stored_class_name] if detected and stored_class_name in self.meta else None

        classes_meta = {}
        for cl in classes:
            cl_ = cl.replace(" ", "_")
            if cl_ in self.meta:
                classes_meta[cl_] = self.meta[cl_]

        return Processed(
            bbox=bbox.astype(int),
            state=DETECTED if detected else NOT_DETECTED,
            label=summary_overlay_label,
            overlay_label=overlay_label_str,
            prob=mean_prob,
            classes=classes,
            classes_meta=classes_meta,
            meta=meta,
            looks_like=[self.classifiers.class_names[ll] for ll in looks_likes],
        )

    def skip_wrong_pose_indices(self, bgr_frame, boxes):
        if self.head_pose_driver is None:
            return set()

        if boxes is None or len(boxes) == 0:
            return set()
        imgs = np.stack(images.get_images(bgr_frame, boxes, 60, 0, do_prewhiten=False))
        outputs = self.head_pose_driver.predict({'data': imgs.transpose([0, 3, 1, 2])})

        yaw = np.abs(outputs[self.head_pose_yaw].reshape([-1]))
        pitch = np.abs(outputs[self.head_pose_pitch].reshape([-1]))
        roll = np.abs(outputs[self.head_pose_roll].reshape([-1]))

        skips = set()
        # print(yaw, pitch, roll)
        for i, (y, p, r) in enumerate(zip(yaw, pitch, roll)):
            if (y > self.head_pose_thresholds[0]
                    or p > self.head_pose_thresholds[1]
                    or r > self.head_pose_thresholds[2]):
                skips.add(i)

        return skips

    def process_frame(self, frame, overlays=True):
        bounding_boxes_detected = self.detect_faces(frame, self.threshold)
        skips = self.skip_wrong_pose_indices(frame, bounding_boxes_detected)

        frame_processed = []
        not_detected_embs = []
        detected_names = []

        if self.use_classifiers:
            imgs = images.get_images(frame, bounding_boxes_detected)

            for img_idx, img in enumerate(imgs):
                # Infer
                # t = time.time()
                # Convert BGR to RGB
                if img_idx not in skips:
                    img = img[:, :, ::-1]
                    img = img.transpose([2, 0, 1]).reshape([1, 3, 160, 160])
                    output = self.inference_facenet(img)
                    # LOG.info('facenet: %.3fms' % ((time.time() - t) * 1000))
                    # output = output[facenet_output]

                    processed = self.process_output(output, bounding_boxes_detected[img_idx])
                    frame_processed.append(processed)

                    if self.process_not_detected:
                        if processed.state == NOT_DETECTED:
                            print('!! NOT_DETECTED')
                            not_detected_embs.append(output)

                    if processed.state == DETECTED:
                        print('!! DETECTED')
                        detected_names.append(processed.classes[0])

                else:
                    processed = Processed(
                        bbox=bounding_boxes_detected[img_idx].astype(int),
                        state=WRONG_FACE_POS,
                        label='',
                        overlay_label='',
                        prob=0,
                        classes=[],
                        classes_meta={},
                        meta=None,
                        looks_like=[],
                    )
                    frame_processed.append(processed)

        if overlays:
            self.add_overlays(frame, frame_processed)

        if self.process_not_detected:
            self.not_detected_embs.append(not_detected_embs)

        self.detected_names.append(detected_names)

        return frame_processed

    def add_overlays(self, frame, frame_processed):
        if frame_processed:
            for processed in frame_processed:
                self.add_overlay(frame, processed)

    def add_overlay(self, frame, processed, align_to_right=True):
        """Add box and label overlays on frame
        :param frame: frame in BGR channels order
        :param processed: processed info - box, label, etc...
        :param align_to_right: align multistring text block to right
        :return:
        """

        fontFace, fontScale, thickness = Detector._getTextProps(frame)

        bbox = processed.bbox.astype(int)
        color = self._color(processed.state)
        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]), (bbox[2], bbox[3]),  # (left, top), (right, bottom)
            color,
            thickness * (2 if processed.state == DETECTED else 1),
        )

        font_inner_padding_w, font_inner_padding_h = 5, 5

        if processed.overlay_label is not None and processed.overlay_label != "":
            strs = processed.overlay_label.split('\n')
            str_w, str_h = 0, 0
            widths = []
            for i, line in enumerate(strs):
                lw, lh = self._getTextSize(frame, line)
                str_w = max(str_w, lw)
                str_h = max(str_h, lh)
                widths.append(lw)
            str_h = int(str_h * 1.6) # line height

            to_right = bbox[0] + str_w > frame.shape[1] - font_inner_padding_w
            top = max(str_h, bbox[1] - int((len(strs) - 0.5) * str_h))

            for i, line in enumerate(strs):
                if align_to_right:
                    # all align to right box border
                    left = (bbox[2] - widths[i] - font_inner_padding_w) \
                        if to_right \
                        else bbox[0] + font_inner_padding_w
                else:
                    # move left each string if it's ending not places on the frame
                    left = frame.shape[1] - widths[i] - font_inner_padding_w \
                        if bbox[0] + widths[i] > frame.shape[1] - font_inner_padding_w \
                        else bbox[0] + font_inner_padding_w

                self._putText(frame, line, left, int(top + i * str_h), color=(0, 0, 0), thickness_mul=3)
                self._putText(frame, line, left, int(top + i * str_h), color=color)

                if len(processed.classes) > 0:
                    classes_preview_size = min(
                        str_h * 3,  # size depends on row height
                        int((processed.bbox[2] - processed.bbox[0] - 10) / len(processed.classes) / 1.2),
                        # size depends on bounding box size
                    )
                    i_left = max(0, processed.bbox[0])
                    i_top = min(processed.bbox[3] + int(classes_preview_size * .1),
                                frame.shape[0] - int(classes_preview_size * 1.1))
                    for cls in processed.classes:
                        cv2.rectangle(
                            frame,
                            (i_left, i_top),
                            (i_left + classes_preview_size, i_top + classes_preview_size),
                            color=color,
                            thickness=thickness + 1,
                        )
                        if cls not in self.classes_previews:
                            # print_fun('Init preview for class "%s"' % cls)
                            self.classes_previews[cls] = None
                            cls_img_path = os.path.join(self.classifiers_dir, "previews/%s.png" % cls.replace(" ", "_"))
                            if os.path.isfile(cls_img_path):
                                try:
                                    self.classes_previews[cls] = cv2.imread(cls_img_path)
                                except Exception as e:
                                    print_fun('Error reading preview for "%s": %s' % (cls, e))
                            else:
                                print_fun('Error reading preview for "%s": file not found' % cls)
                        cls_img = self.classes_previews[cls]
                        if cls_img is not None:
                            resized = images.image_resize(cls_img, classes_preview_size, classes_preview_size)
                            try:
                                frame[i_top:i_top + resized.shape[0], i_left:i_left + resized.shape[1]] = resized
                            except Exception as e:
                                print_fun("ERROR: %s" % e)
                        i_left += int(classes_preview_size * 1.2)

    @staticmethod
    def _getTextProps(frame, thickness=None, thickness_mul=None, fontScale=None, fontFace=None):
        if fontScale is None or thickness is None:
            frame_avg = (frame.shape[1] + frame.shape[0]) / 2
            if fontScale is None:
                fontScale = frame_avg / 1200
            if thickness is None:
                thickness = int(fontScale * 2)
            if thickness_mul is not None:
                thickness_m = int(thickness * thickness_mul)
                thickness = thickness + 1 if thickness == thickness_m else thickness_m
        if fontFace is None:
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
        return fontFace, fontScale, thickness

    @staticmethod
    def _getTextSize(frame, text, thickness=None, thickness_mul=None, fontScale=None, fontFace=None):
        fontFace, fontScale, thickness = Detector._getTextProps(frame, thickness, thickness_mul, fontScale, fontFace)
        return cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    @staticmethod
    def _putText(frame, text, left, top, color, thickness=None, thickness_mul=None,
                 fontScale=None, fontFace=None, lineType=cv2.LINE_AA):
        fontFace, fontScale, thickness = Detector._getTextProps(frame, thickness, thickness_mul, fontScale, fontFace)
        cv2.putText(frame, text, (left, top), fontFace, fontScale, color, thickness=thickness, lineType=lineType)

    @staticmethod
    def _color(state):
        if state == DETECTED:
            return 0, 255, 0
        elif state == WRONG_FACE_POS:
            return 0, 0, 250
        else:
            return 250, 0, 250

    @staticmethod
    def _openvino_detect(face_detect, frame, threshold):
        inference_frame = cv2.resize(frame, face_detect.input_size)  # , interpolation=cv2.INTER_AREA)
        inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(*face_detect.input_shape)
        outputs = face_detect(inference_frame)
        outputs = outputs.reshape(-1, 7)
        bboxes_raw = outputs[outputs[:, 2] > threshold]
        bounding_boxes = bboxes_raw[:, 3:7]
        bounding_boxes[:, 0] = bounding_boxes[:, 0] * frame.shape[1]
        bounding_boxes[:, 2] = bounding_boxes[:, 2] * frame.shape[1]
        bounding_boxes[:, 1] = bounding_boxes[:, 1] * frame.shape[0]
        bounding_boxes[:, 3] = bounding_boxes[:, 3] * frame.shape[0]
        return bounding_boxes
