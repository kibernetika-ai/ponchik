import logging
import srt
import datetime

import numpy as np
from sklearn import cluster

LOG = logging.getLogger(__name__)

class PostProcessor:

    def __init__(self, data, detector):

        self.detector = detector

        self.head_poses = data['head_poses']
        self.frame_nums = data['frame_nums']
        self.bounding_boxes = data['bounding_boxes']
        self.embeddings = data['embeddings']

        self.data_attrs = data.attrs

        self.correct_head_poses = None
        self.frame_sequences = None

        self.middle_sequence_frames = None
        self.sequences_frames_recognized = None
        self.sequences_frames_not_recognized = None
        self.clt_seq = None
        self.clt_seq_counts = None

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

    def generate_sequences(self):
        LOG.info('generating frames sequences...')
        min_face_sequence_len = 3

        face_sequences = []
        face_sequences_process = []

        prev_frame = None
        current_frame_boxes = []
        for idx, frame_num in enumerate(self.frame_nums):
            if self.correct_head_poses is not None and idx not in self.correct_head_poses:
                continue
            if frame_num != prev_frame:
                # process previous frame
                if prev_frame is not None:
                    # check current sequences
                    for ifproc, fproc in enumerate(face_sequences_process):
                        # close "breaked" sequences and store it if it has enouth length
                        if fproc[1] < frame_num - self.data_attrs['each_frame']:
                            if len(fproc[2]) >= min_face_sequence_len:
                                face_sequences.append(fproc)
                            del face_sequences_process[ifproc]
                    for ifb, fb in enumerate(current_frame_boxes):
                        found = False
                        for ifproc, fproc in enumerate(face_sequences_process):
                            if self.box_intersection(fproc[0], fb[0]) > 0:
                                fproc[2].append(fb[1])
                                fproc[3].append(fb[0])
                                face_sequences_process[ifproc] = (fb[0], frame_num, fproc[2], fproc[3])
                                found = True
                                break
                        if not found:
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

    def recognize_sequences(self):

        LOG.info('recognizing frame sequences by middle frames...')

        self.middle_sequence_frames = []
        for fs in self.frame_sequences:
            middle_frame = fs[len(fs) // 2]
            self.middle_sequence_frames.append(middle_frame)
        self.middle_sequence_frames.sort()

        self.detector.init()

        self.sequences_frames_recognized = {}
        self.sequences_frames_not_recognized = []
        for idx in self.middle_sequence_frames:
            processed = self.detector.process_output(self.embeddings[idx], np.array([0, 0, 0, 0, 0]))
            if processed.label != "":
                self.sequences_frames_recognized[idx] = processed
            else:
                self.sequences_frames_not_recognized.append(idx)

        LOG.info('recognizing sequences DONE, recognized: {}, not recognized: {}'.format(
            len(self.sequences_frames_recognized),
            len(self.sequences_frames_not_recognized),
        ))

    def clusterize(self):
        LOG.info('clusterizing unrecognized...')
        self.clt_seq = cluster.DBSCAN(metric="euclidean", n_jobs=-1, min_samples=5)
        self.clt_seq.fit(self.embeddings[self.sequences_frames_not_recognized])
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

    def face_info(self, idx):
        for s in self.frame_sequences:
            if idx in s:
                for ss in s:
                    if ss in self.sequences_frames_not_recognized:
                        return 'Person {}'.format(self.clt_seq.labels_[self.sequences_frames_not_recognized.index(ss)])
                    if ss in self.sequences_frames_recognized:
                        return self.sequences_frames_recognized[ss].overlay_label
        return ''

    def export_srt(self, srt_file):

        LOG.info('writing SRT to {}...'.format(srt_file))
        sequences_frames_not_recognized_labels = {}
        for f in self.sequences_frames_not_recognized:
            sequences_frames_not_recognized_labels[f] = self.clt_seq.labels_[self.sequences_frames_not_recognized.index(f)]
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

            if idx > 10000:
                break

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
