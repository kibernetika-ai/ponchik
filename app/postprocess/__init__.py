import logging

import datetime

import numpy as np
from sklearn import cluster
from scipy.spatial import distance

from app.recognize import detector

LOG = logging.getLogger(__name__)


class PostProcessor:

    def __init__(self, data, detector):

        self.detector = detector

        self.head_poses = data['face_head_poses']
        self.frame_nums = data['face_frame_nums']
        self.bounding_boxes = data['face_bboxes']
        self.embeddings = data['face_embeddings']

        self.data_attrs = data.attrs

        self.correct_head_poses = None
        self.frame_sequences = None

        self.frame_sequences_faces = None
        self.frame_sequences_faces_not_recognized = None
        self.frame_sequence_recognition_result = None
        self.frame_sequence_cluster_result = None
        self.clt_seq = None
        self.clt_seq_counts = None

    def run(self):
        self.calculate_correct_head_poses()
        self.generate_sequences()
        self.exclude_sequences_without_correct_poses()
        self.get_good_faces_for_sequences()
        self.recognize_sequences_faces()
        self.clusterize_sequences()

    def calculate_correct_head_poses(self, head_poses_tresholds=None):
        if head_poses_tresholds is None:
            head_poses_tresholds = self.detector.head_pose_thresholds
            # head_poses_tresholds = [20., 20., 20.]  # [37., 35., 25.]
        LOG.info('calculating correct head poses with tresholds {}...'.format(head_poses_tresholds))
        correct_head_poses_bool = [all(abs(hp) < head_poses_tresholds) for hp in self.head_poses]
        self.correct_head_poses = [i for i, v in enumerate(correct_head_poses_bool) if v]
        LOG.info('calculating correct head poses DONE')

    @staticmethod
    def box_intersection(box_a, box_b):
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_area = max(0, xb - xa) * max(0, yb - ya)

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        d = float(box_a_area + box_b_area - inter_area)
        if d == 0:
            return 0
        iou = inter_area / d
        return iou

    def distance_between(self, idx1, idx2):
        return distance.cosine(self.embeddings[idx1], self.embeddings[idx2])

    def generate_sequences(self):
        LOG.info('generating frames sequences...')
        min_face_sequence_len = 3

        face_sequences = []
        face_sequences_process = []

        prev_frame = None
        current_frame_boxes = []
        for idx, frame_num in enumerate(self.frame_nums):
            if idx and idx % 5000 == 0:
                LOG.info('processed index {} of {}'.format(idx, len(self.frame_nums)))
            # do not skip wrong poses in all cases
            # if self.correct_head_poses is not None and idx not in self.correct_head_poses:
            #     continue
            if frame_num != prev_frame:
                # process previous frame
                if prev_frame is not None:
                    # check current sequences
                    for ifproc, fproc in enumerate(face_sequences_process):
                        # close "broken" sequences and store it if it has enough length
                        if fproc[1] < frame_num - self.data_attrs['each_frame']:
                            if len(fproc[2]) >= min_face_sequence_len:
                                face_sequences.append(fproc)
                            del face_sequences_process[ifproc]
                    for ifb, fb in enumerate(current_frame_boxes):
                        found = False
                        for ifproc, fproc in enumerate(face_sequences_process):
                            bi = self.box_intersection(fproc[0], fb[0])
                            if bi > .3:
                                # distance between current and last in sequence (previous) embeddings
                                dist = self.distance_between(fb[1], fproc[2][-1])
                                if dist < .3:
                                    fproc[2].append(fb[1])
                                    fproc[3].append(fb[0])
                                    face_sequences_process[ifproc] = (fb[0], frame_num, fproc[2], fproc[3])
                                    found = True
                                    break
                        if not found:
                            # skip wrong pose only for sequence beginning
                            # if idx in self.correct_head_poses:
                            face_sequences_process.append((fb[0], frame_num, [fb[1]], [fb[0]]))
                    current_frame_boxes = []
                # starts next frame
                prev_frame = frame_num
            current_frame_boxes.append((self.bounding_boxes[idx], idx))

        # close opened sequences
        for ifproc, fproc in enumerate(face_sequences_process):
            if len(fproc[2]) >= min_face_sequence_len:
                face_sequences.append(fproc)

        self.frame_sequences = [f[2] for f in face_sequences]
        LOG.info('generating frames sequences DONE, sequences count: {}'.format(len(self.frame_sequences)))

    def exclude_sequences_without_correct_poses(self):
        LOG.info('skipping sequences without any frame with correct face pose...')
        frame_sequences_with_correct_poses = []
        for frame_sequence in self.frame_sequences:
            for frame in frame_sequence:
                if frame in self.correct_head_poses:
                    frame_sequences_with_correct_poses.append(frame_sequence)
                    break
        LOG.info(
            'sequences with at least frame with correct head pose: {}'.format(len(frame_sequences_with_correct_poses)))
        self.frame_sequences = frame_sequences_with_correct_poses
        LOG.info('skipping sequences DONE, sequences count: {}'.format(len(self.frame_sequences)))

    def get_good_faces_for_sequences(self, n_seconds = 5):
        LOG.info('get good pose faces for sequences: first and each next {} seconds (but only with good poses)...'.
                 format(n_seconds))
        self.frame_sequences_faces = []
        for frame_sequence in self.frame_sequences:
            frame_sequence_prev_frame = None
            for frame in frame_sequence:
                frame_sequence_cur_frame = self.frame_nums[frame]
                if frame_sequence_prev_frame is None or frame_sequence_cur_frame - frame_sequence_prev_frame > \
                        n_seconds * self.data_attrs['fps']:
                    self.frame_sequences_faces.append(frame)
                    frame_sequence_prev_frame = frame_sequence_cur_frame
        self.frame_sequences_faces.sort()
        LOG.info('good pose faces for recognition and clusterization DONE: total {} faces'.
                 format(len(self.frame_sequences_faces)))

    def recognize_sequences_faces(self):

        LOG.info('recognizing frame sequences by good faces frames...')

        self.detector.init()

        # Recognize good pose faces from sequences
        frame_sequences_faces_recognized = {}
        # frame_sequences_faces_not_recognized = []
        for idx in self.frame_sequences_faces:
            processed = self.detector.process_output(self.embeddings[idx], np.array([0, 0, 0, 0, 0]))
            if processed.is_detected():
                frame_sequences_faces_recognized[idx] = processed

        LOG.info('Recognized {} faces of {}'.
                 format(len(frame_sequences_faces_recognized), len(self.frame_sequences_faces)))

        # If sequence has at least one recognized face and all recognized faces are more than 50% of all faces to recognition
        # all this sequence is recognized as this person
        frame_sequence_recognition_stats = []

        for i, frame_sequence in enumerate(self.frame_sequences):
            recognized_stats = {}
            for frame in frame_sequence:
                if frame in frame_sequences_faces_recognized:
                    recognized = frame_sequences_faces_recognized[frame]
                    recognized_class = recognized.classes[0]
                    if recognized_class not in recognized_stats:
                        recognized_stats[recognized_class] = (recognized, 0)
                    recognized_stats[recognized_class] = (
                        recognized_stats[recognized_class][0],
                        recognized_stats[recognized_class][1] + 1,
                    )
            frame_sequence_recognition_stats.append(recognized_stats)

        # frame_sequence_recognition_stats
        self.frame_sequence_recognition_result = []
        for i, stats in enumerate(frame_sequence_recognition_stats):
            max_info = None
            max_info_value = 0
            total_info_value = 0
            for cl in stats:
                info = stats[cl]
                total_info_value += info[1]
                if info[1] > max_info_value:
                    max_info_value = info[1]
                    max_info = info[0]
            face_class = max_info if max_info_value > total_info_value / 2 else None
            if face_class is not None:
                face_class.overlay_label = 'Seq.1 {}\n{}'.format(i, face_class.overlay_label)
            self.frame_sequence_recognition_result.append(face_class)
        LOG.info('recognized sequences: {}, not recognized: {}'.format(
            len([r for r in self.frame_sequence_recognition_result if r is not None]),
            len([r for r in self.frame_sequence_recognition_result if r is None]),
        ))

        self.frame_sequences_faces_not_recognized = []
        for idx in self.frame_sequences_faces:
            seq_idx = None
            for i, seq in enumerate(self.frame_sequences):
                if idx in seq:
                    seq_idx = i
                    break
            if seq_idx is None or self.frame_sequence_recognition_result[seq_idx] is None:
                self.frame_sequences_faces_not_recognized.append(idx)
        LOG.info('not recognized faces for clusterization: {} of {}'.
                 format(len(self.frame_sequences_faces_not_recognized), len(self.frame_sequences_faces)))

    def clusterize_sequences(self):
        LOG.info('clusterizing unrecognized...')
        self.frame_sequence_cluster_result = []
        if len(self.frame_sequences_faces_not_recognized) == 0:
            LOG.info('there\'re no unrecognized faces, nothing to clusterize.')
            return
        self.clt_seq = cluster.DBSCAN(metric="euclidean", n_jobs=-1, min_samples=5)
        self.clt_seq.fit(self.embeddings[self.frame_sequences_faces_not_recognized])
        LOG.info('clusterizing unrecognized DONE, different clusters (without unrecognized): {}'.
                 format(len(set(self.clt_seq.labels_))-1))
        self.clt_seq_counts = [len(self.clt_seq.labels_[self.clt_seq.labels_ == lbl])
                               for lbl in list(set(self.clt_seq.labels_))
                               if lbl >= 0]
        LOG.info('Unrecognized count: {}'.
                 format(len(self.clt_seq.labels_[self.clt_seq.labels_ == -1])))
        if len(self.clt_seq_counts) > 0:
            LOG.info('Min class length: {} for class {}'.
                     format(min(self.clt_seq_counts), self.clt_seq_counts.index(min(self.clt_seq_counts))))
            LOG.info('Max class length: {} for class {}'.
                     format(max(self.clt_seq_counts), self.clt_seq_counts.index(max(self.clt_seq_counts))))
            LOG.info('Median class length: {:1.2f}, average class length: {:1.2f}'.
                     format(np.median(self.clt_seq_counts), np.mean(self.clt_seq_counts)))

        frame_sequence_cluster_stats = []

        for i, frame_sequence in enumerate(self.frame_sequences):
            cluster_stats = {}
            for frame in frame_sequence:
                if frame in self.frame_sequences_faces_not_recognized:
                    clusterized = self.clt_seq.labels_[self.frame_sequences_faces_not_recognized.index(frame)]
                    if clusterized >= 0:
                        if clusterized not in cluster_stats:
                            cluster_stats[clusterized] = (clusterized, 0)
                        cluster_stats[clusterized] = (
                            cluster_stats[clusterized][0],
                            cluster_stats[clusterized][1] + 1,
                        )

            frame_sequence_cluster_stats.append(cluster_stats)

        for stats in frame_sequence_cluster_stats:
            max_info = None
            max_info_value = 0
            total_info_value = 0
            for cl in stats:
                info = stats[cl]
                total_info_value += info[1]
                if info[1] > max_info_value:
                    max_info_value = info[1]
                    max_info = info[0]
            clusterized_face = None
            if max_info_value > total_info_value / 2:
                clusterized_face = detector.FaceInfo()
                clusterized_face.state = detector.DETECTED
                clusterized_face.label = 'Person {}'.format(max_info)
                clusterized_face.overlay_label = clusterized_face.label
            self.frame_sequence_cluster_result.append(clusterized_face)
        LOG.info('Clusterized sequences: {}, not clusterized: {}'.format(
            len([r for r in self.frame_sequence_cluster_result if r is not None]),
            len([r for r in self.frame_sequence_cluster_result if r is None]),
        ))

    def get_sequence_idx(self, idx):
        for i, frame_sequence in enumerate(self.frame_sequences):
            if idx in frame_sequence:
                return i
        return None

    def get_sequence_recognized_face(self, idx):
        sequence_idx = self.get_sequence_idx(idx)
        if sequence_idx is not None:
            res = self.frame_sequence_recognition_result[sequence_idx]
            # if res is None:
            #     return detector.FaceInfo(overlay_label='Seq3 {}'.format(sequence_idx))
            if res is not None:
                return res
            return self.frame_sequence_cluster_result[sequence_idx]
        return None

    def get_face(self, idx):
        sequence_idx = self.get_sequence_idx(idx)
        if sequence_idx is not None:
            recognized = self.frame_sequence_recognition_result[sequence_idx]
            if recognized is not None:
                return recognized

        recognized = self.get_sequence_recognized_face(idx)
        if recognized is not None:
            return recognized

        sequence_idx = self.get_sequence_idx(idx)
        if sequence_idx is not None:
            res = self.frame_sequence_recognition_result[sequence_idx]
            if res is None:
                return detector.FaceInfo(overlay_label='Seq.3 {}'.format(sequence_idx))
            return res
        return detector.FaceInfo(overlay_label='Seq.2 {}'.format(sequence_idx))

    def export_srt(self, srt_file):
        import srt
        LOG.info('writing SRT to {}...'.format(srt_file))
        sequences_frames_not_recognized_labels = {}
        for f in self.frame_sequences_faces_not_recognized:
            sequences_frames_not_recognized_labels[f] = self.clt_seq.labels_[self.frame_sequences_faces_not_recognized.index(f)]
        frames_not_recognized_labels = {}
        for s in self.frame_sequences:
            found = None
            for ss in s:
                if ss in sequences_frames_not_recognized_labels:
                    found = sequences_frames_not_recognized_labels[ss]
                    break
            if found is not None:
                for ss in s:
                    frames_not_recognized_labels[ss] = found

        frames_recognized_labels = {}
        for s in self.frame_sequences:
            found = None
            for ss in s:
                if ss in self.sequences_frames_recognized:
                    found = self.sequences_frames_recognized[ss].label
                    break
            if found is not None:
                for ss in s:
                    frames_recognized_labels[ss] = found

        subs = []
        fps = self.data_attrs['fps']

        label = None
        prev_frame = None
        prev_label = None
        current_frame_not_recognized_labels = []
        current_frame_recognized_labels = []
        start = datetime.timedelta(milliseconds=0)

        for idx, fn in enumerate(self.frame_nums):
            #     if fn != prev_frame:
            #         print(fn, current_frame_recognized_labels, current_frame_not_recognized_labels)
            if fn != prev_frame and prev_frame is not None:
                end = datetime.timedelta(seconds=float(prev_frame) / fps)
                label_parts = []
                if len(current_frame_recognized_labels) > 0:
                    current_frame_recognized_labels.sort()
                    label_parts.append(', '.join(current_frame_recognized_labels))
                if len(current_frame_not_recognized_labels) > 0:
                    current_frame_not_recognized_labels.sort()
                    label_parts.append(
                        ', '.join(['Person {}'.format(p) for p in current_frame_not_recognized_labels if p >= 0]))
                label = ', '.join(label_parts)
                if label != prev_label:
                    if prev_label is not None and prev_label != "":
                        sub = srt.Subtitle(
                            index=len(subs) + 1,
                            start=start,
                            end=end,
                            content=prev_label
                        )
                        subs.append(sub)
                    prev_label = label
                    start = end
                current_frame_not_recognized_labels = []
                current_frame_recognized_labels = []

            if idx in frames_not_recognized_labels:
                current_frame_not_recognized_labels.append(frames_not_recognized_labels[idx])
            if idx in frames_recognized_labels:
                current_frame_recognized_labels.append(frames_recognized_labels[idx])
            prev_frame = fn

        end = datetime.timedelta(seconds=float(prev_frame) / fps)
        sub = srt.Subtitle(
            index=len(subs) + 1,
            start=start,
            end=end,
            content=label
        )
        subs.append(sub)

        with open(srt_file, 'w') as sw:
            sw.write(srt.compose(subs))

        LOG.info('writing SRT DONE, {} records'.format(len(subs)))
