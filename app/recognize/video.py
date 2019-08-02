import os
import logging
import time
from threading import Thread

import cv2
import h5py
import numpy as np
import datetime
from scipy.spatial import distance

from app.postprocess import PostProcessor
from app.recognize import defaults
from app.recognize.clusterizator import Clusterizator
from app.tools import images
from app.tools import sound
from app.recognize.video_notify import InVideoDetected
from app.recognize import detector
from app.notify import notify

LOG = logging.getLogger(__name__)


def video_args(dtct: detector.Detector, listener, args):
    return Video(
        dtct,
        listener=listener,
        video_source=args.video_source,
        video_async=args.video_async,
        video_max_width=args.video_max_width,
        video_max_height=args.video_max_height,
        video_each_of_frame=args.video_each_of_frame,
        not_detected_store=args.video_not_detected_store,
        not_detected_check_period=args.video_not_detected_check_period,
        not_detected_dir=args.video_not_detected_dir,
        process_not_detected=args.process_not_detected,
        video_export_srt=args.video_export_srt,
        video_export_srt_file=args.video_export_srt_file,
        video_no_output=args.video_no_output,
        video_write_to=args.video_write_to,
        video_h5py=args.video_h5py,
        build_h5py=args.build_h5py,
        build_h5py_to=args.build_h5py_to,
        video_limit_sec=args.video_limit_sec,
        video_start_sec=args.video_start_sec,
        postprocess=args.postprocess,
        postprocess_clusterize_unrecognized=args.postprocess_clusterize_unrecognized,
        postprocess_export_srt=args.postprocess_export_srt,
        postprocess_export_srt_to=args.postprocess_export_srt_to,
    )


class Video:
    def __init__(self, detector: detector.Detector,
                 listener=None, video_source=None, video_async=False,
                 video_max_width=None, video_max_height=None, video_each_of_frame=1,
                 video_export_srt=False, video_export_srt_file=None,
                 not_detected_store=False, not_detected_check_period=defaults.NOT_DETECTED_CHECK_PERIOD,
                 not_detected_dir=defaults.NOT_DETECTED_DIR, process_not_detected=False,
                 video_no_output=False, build_h5py=False, build_h5py_to=None, video_h5py=None, video_limit_sec=None,
                 video_write_to=None, video_start_sec=None,
                 postprocess=False, postprocess_clusterize_unrecognized=False,
                 postprocess_export_srt=False, postprocess_export_srt_to=None,
                 ):
        self.detector = detector
        self.video_source = video_source
        self.video_source_is_file = self.check_video_source_is_file()
        self.fps = None
        self.width = None
        self.height = None
        self.video_limit_sec = video_limit_sec
        self.video_start_sec = video_start_sec
        self.video_write_to = video_write_to
        self.frame = None
        self.processed = None
        self.vs = None
        self.listener = listener
        self.video_async = video_async
        self.pipeline = None
        self.video_max_width = video_max_width
        self.video_max_height = video_max_height
        self.video_each_of_frame = max(1, video_each_of_frame)
        self.faces_detected = {}
        self.notify_started = False
        self.notifies_queue = []
        self.not_detected_store = not_detected_store
        self.not_detected_check_period = not_detected_check_period
        self.not_detected_dir = not_detected_dir
        if self.not_detected_store and not os.path.isdir(self.not_detected_dir):
            raise RuntimeError('directory %s is not exists' % self.not_detected_dir)
        self.not_detected_check_ts = time.time()
        self.process_not_detected = process_not_detected
        self.video_export_srt = video_export_srt
        self.video_export_srt_file = video_export_srt_file
        self.video_no_output = video_no_output
        self.h5 = None

        self.saved_faces = []
        self.save_faces = True
        self.postprocess_check = None

        if build_h5py or build_h5py_to:
            if video_h5py:
                raise ValueError('unable to read and write h5 data at once')
            if not build_h5py_to:
                if not self.video_source_is_file:
                    raise RuntimeError('h5 filename is not set')
                build_h5py_to = os.path.splitext(self.video_source)[0] + '.h5'
            self.h5 = h5py.File(build_h5py_to, 'w')
            self.h5.create_dataset(
                'face_embeddings',
                shape=(0, 512),
                dtype=np.float32,
                maxshape=(None, 512),
                chunks=True,
            )
            self.h5.create_dataset(
                'face_head_poses',
                shape=(0, 3),
                dtype=np.float32,
                maxshape=(None, 3),
                chunks=True,
            )
            self.h5.create_dataset(
                'face_bboxes',
                shape=(0, 4),
                dtype=np.int32,
                maxshape=(None, 4),
                chunks=True,
            )
            self.h5.create_dataset(
                'face_person_bboxes',
                shape=(0, 4),
                dtype=np.int32,
                maxshape=(None, 4),
                chunks=True,
            )
            self.h5.create_dataset(
                'face_probs',
                shape=(0,),
                dtype=np.float32,
                maxshape=(None,),
                chunks=True,
            )
            self.h5.create_dataset(
                'face_frame_nums',
                shape=(0,),
                dtype=np.int32,
                maxshape=(None,),
                chunks=True,
            )
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            self.h5.create_dataset(
                'face_images',
                shape=(0,),
                # dtype="|S10",
                dtype=dt,
                maxshape=(None,),
                chunks=True,
            )
            self.h5.create_dataset(
                'person_bboxes',
                shape=(0, 4),
                dtype=np.int32,
                maxshape=(None, 4),
                chunks=True,
            )
            self.h5.create_dataset(
                'person_probs',
                shape=(0,),
                dtype=np.float32,
                maxshape=(None,),
                chunks=True,
            )
            self.h5.create_dataset(
                'person_frame_nums',
                shape=(0,),
                dtype=np.int32,
                maxshape=(None,),
                chunks=True,
            )

        self.h5data = None
        self.h5data_skip = False
        if video_h5py:
            self.h5data = h5py.File(video_h5py, 'r')
            LOG.info('read data from {}'.format(video_h5py))
            self.fps = self.h5data.attrs['fps']
            self.width = self.h5data.attrs['width']
            self.height = self.h5data.attrs['height']
            self.video_each_of_frame = self.h5data.attrs['each_frame']
            self.detector.threshold = self.h5data.attrs['threshold']
            self.detector.min_face_size = self.h5data.attrs['min_face_size']
            LOG.info('set fps {}, width {}, height {}, each frame {}, threshold {}, min face size {}'.format(
                self.fps,
                self.width,
                self.height,
                self.video_each_of_frame,
                self.detector.threshold,
                self.detector.min_face_size,
            ))

        self.postprocess = None
        if postprocess:
            if self.h5data is None:
                raise ValueError('postprocess is available only with h5 stored data')
            self.postprocess = PostProcessor(self.h5data, self.detector)

            if postprocess_clusterize_unrecognized:
                self.postprocess.run()
                if postprocess_export_srt:
                    if postprocess_export_srt_to is None:
                        postprocess_export_srt_to = os.path.splitext(self.video_source)[0] + '.srt'
                    self.postprocess.export_srt(postprocess_export_srt_to)

    def start_notify(self):
        if self.notify_started:
            return
        notify_thread = Thread(target=self.notify, daemon=True)
        notify_thread.start()
        self.notify_started = True

    def check_video_source_is_file(self):
        return self.video_source and self.video_source != "realsense" and os.path.isfile(self.video_source)

    def init_video(self):
        if self.video_source is None:
            self.vs = cv2.VideoCapture(0)
        elif self.video_source == "realsense":
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()

            config = rs.config()
            # rs-enumerate-devices
            # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8,6)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 6)
            self.pipeline.start(config)
        else:
            self.vs = cv2.VideoCapture(self.video_source)
            if self.video_source_is_file:
                self.fourcc = int(self.vs.get(cv2.CAP_PROP_FOURCC))
                self.video_format = self.vs.get(cv2.CAP_PROP_FORMAT)
                self.fps = self.vs.get(cv2.CAP_PROP_FPS)
                self.width = int(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.video_export_srt:
                if not self.video_source_is_file:
                    raise ValueError('srt creation allowed only for video file')
                if not self.fps:
                    raise ValueError('unable to detect fps, srt creation is unavailable')

        if self.video_start_sec:
            skip_num = int(self.fps * self.video_start_sec)
            LOG.info('Skipping %s frames...' % skip_num)

            for i in range(skip_num):
                self.get_frame(decode=False)

            LOG.info('Skipped %s sec (%s frames).' % (self.video_start_sec, skip_num))

    def start(self):
        self.detector.init()
        if self.vs is None:
            self.init_video()
        if self.video_async:
            frame_thread = Thread(target=self.process_frame_async, daemon=True)
            frame_thread.start()
        if self.listener is not None:
            listen_thread = Thread(target=self.listen, daemon=True)
            listen_thread.start()

        self.start_notify()

        if self.h5:
            self.h5.attrs['fps'] = self.fps
            self.h5.attrs['width'] = self.width
            self.h5.attrs['height'] = self.height
            self.h5.attrs['each_frame'] = self.video_each_of_frame
            self.h5.attrs['threshold'] = self.detector.threshold
            self.h5.attrs['min_face_size'] = self.detector.min_face_size

        if self.video_write_to:
            if os.path.isfile(self.video_write_to):
                os.remove(self.video_write_to)
            video_writer = cv2.VideoWriter(
                self.video_write_to, self.fourcc, self.fps / self.video_each_of_frame,
                frameSize=(self.width, self.height)
            )

        try:
            processed_frame_idx = 0
            self.frame_idx = 0
            self.h5data_face_idx = 0
            self.h5data_person_idx = 0
            while True:
                # Capture frame-by-frame
                self.get_frame()
                self.frame_idx += 1
                if self.frame is None and self.video_source_is_file:
                    break
                if self.frame is not None and (self.frame_idx % self.video_each_of_frame == 0):
                    frame = self.frame.copy()
                    if self.video_async:
                        # Not works
                        self.detector.add_overlays(frame, self.processed)
                    else:
                        self.process_frame(frame=frame)
                    if not self.video_no_output:
                        cv2.imshow('Video', frame)
                    if self.video_write_to:
                        video_writer.write(frame)

                    processed_frame_idx += 1
                    t = datetime.timedelta(
                        milliseconds=self.frame_idx / self.fps * 1000) if self.fps else '-'

                    if processed_frame_idx % 100 == 0:
                        LOG.info("Processed %d frames, %s" % (processed_frame_idx, t))

                    if self.video_limit_sec and t.total_seconds() >= self.video_limit_sec:
                        LOG.info('Stop processing. It was limited to %s seconds.' % self.video_limit_sec)
                        break

                if not self.video_no_output:
                    key = cv2.waitKey(1)
                    # Wait 'q' or Esc or 'q' in russian layout
                    if key in [ord('q'), 202, 27]:
                        break

        except (KeyboardInterrupt, SystemExit) as e:
            LOG.info('Caught %s: %s' % (e.__class__.__name__, e))
        finally:
            if self.pipeline is not None:
                self.pipeline.stop()
            if self.h5:
                LOG.info('Written data for %s faces and %s persons.' % (
                    self.h5['face_bboxes'].shape[0], self.h5['person_bboxes'].shape[0]))
                self.h5.close()

            if self.video_write_to:
                LOG.info('Written video to %s.' % self.video_write_to)
                video_writer.release()

                # Add sound as well
                sound.merge_audio_with(self.video_source, self.video_write_to, self.video_limit_sec)

            cv2.destroyAllWindows()

        if self.video_export_srt:
            import srt

            detected_persons = self.detector.detected_names
            not_detected_persons = []

            if self.process_not_detected:
                LOG.info('Start unrecognized faces clustering')
                cl = Clusterizator()
                not_detected_persons = cl.clusterize_frame_faces(self.detector.not_detected_embs)
                LOG.info('Clustering done')

            # plain subtitles
            subs_strs = []
            for i in range(max(len(detected_persons), len(not_detected_persons))):
                frame_persons = []
                if len(detected_persons) > i:
                    names = detected_persons[i]
                    names.sort()
                    frame_persons.extend(names)
                if len(not_detected_persons) > i:
                    persons = [d for d in not_detected_persons[i] if d >= 0]
                    persons.sort()
                    frame_persons.extend(['Person %d' % d for d in persons])
                sub_str = None if len(frame_persons) == 0 else ", ".join(frame_persons)
                subs_strs.append(sub_str)

            # plain subtitles w/o one frame "holes"
            for i in range(len(subs_strs) - 2):
                if subs_strs[i] is not None and subs_strs[i + 1] != subs_strs[i] and subs_strs[i + 2] == subs_strs[i]:
                    subs_strs[i + 1] = subs_strs[i]

            subs = []
            cur_sub, cur_sub_start, cur_sub_i = None, None, None
            ts = None
            for i in range(len(subs_strs)):
                ts = datetime.timedelta(milliseconds=(i / self.fps) * 1000 * self.video_each_of_frame)
                sub = subs_strs[i]
                prev_sub = cur_sub
                if sub != cur_sub or (sub is not None and sub != prev_sub):
                    if cur_sub is not None:
                        # content = "%d-%d:\n%s\n%s" % (cur_sub_i, i, cur_sub, prev_sub)
                        content = cur_sub
                        subs.append(srt.Subtitle(
                            index=len(subs) + 1,
                            start=cur_sub_start,
                            end=ts,
                            content=content,
                        ))
                    cur_sub_start = ts
                    cur_sub_i = i
                    cur_sub = sub

            if cur_sub is not None:
                subs.append(
                    srt.Subtitle(index=len(subs) + 1, start=cur_sub_start, end=ts, content=cur_sub))

            video_export_srt_file = self.video_export_srt_file \
                if self.video_export_srt_file \
                else os.path.splitext(self.video_source)[0] + '.srt'
            with open(video_export_srt_file, 'w') as sw:
                sw.write(srt.compose(subs))

    def get_frame(self, decode=True):
        if self.pipeline is None:
            if decode:
                new_frame = self.vs.read()
            else:
                self.vs.grab()
                new_frame = 'not decoded'
        else:
            frames = self.pipeline.wait_for_frames()
            new_frame = frames.get_color_frame()
            new_frame = np.asanyarray(new_frame.get_data())
        if isinstance(new_frame, tuple):
            new_frame = new_frame[1]
        if new_frame is None:
            if not self.video_source_is_file:
                LOG.info("Oops frame is None. Possibly camera or display does not work")
            self.frame = None
            return None
        if self.video_max_width is not None and new_frame.shape[1] > self.video_max_width or \
                self.video_max_height is not None and new_frame.shape[0] > self.video_max_height:
            new_frame = images.image_resize(new_frame, width=self.video_max_width, height=self.video_max_height)
        self.frame = new_frame
        return self.frame

    def process_frame_async(self):
        while True:
            self.process_frame(self.frame, overlays=False)

    def process_frame(self, frame, overlays=True):
        original_copy = np.copy(frame)
        stored_faces = None
        stored_persons = None
        if self.h5data:
            stored_faces = []
            stored_persons = []
            if self.h5data_face_idx >= len(self.h5data['face_frame_nums']):
                if not self.h5data_skip:
                    LOG.warning(
                        'max h5 data index {} reached, skipped all next frames'.format(
                            len(self.h5data['face_frame_nums'])))
                    self.h5data_skip = True
                return
            while self.h5data['face_frame_nums'][self.h5data_face_idx] == self.frame_idx:
                stored_face = None
                if self.postprocess:
                    stored_face = self.postprocess.get_sequence_recognized_face(self.h5data_face_idx)
                if stored_face is None:
                    stored_face = detector.FaceInfo()
                stored_face.bbox = self.h5data['face_bboxes'][self.h5data_face_idx]
                stored_face.person_bbox = self.h5data['face_person_bboxes'][self.h5data_face_idx]
                if self._h5_box_is_none(stored_face.person_bbox):
                    stored_face.person_bbox = None
                stored_face.embedding = self.h5data['face_embeddings'][self.h5data_face_idx]
                stored_face.face_prob = self.h5data['face_probs'][self.h5data_face_idx]
                stored_face.head_pose = self.h5data['face_head_poses'][self.h5data_face_idx]
                stored_face.state = detector.DETECTED

                # stored_face = detector.FaceInfo(
                #     bbox=self.h5data['bounding_boxes'][self.h5data_idx],
                #     embedding=self.h5data['embeddings'][self.h5data_idx],
                #     face_prob=self.h5data['face_probs'][self.h5data_idx],
                #     head_pose=self.h5data['head_poses'][self.h5data_idx],
                # )
                # if self.postprocess:
                #     sequence_face = self.postprocess.get_sequence_recognized_face(self.h5data_idx)
                #     if sequence_face is not None:
                #         stored_face.label = sequence_face.label
                #         stored_face.overlay_label = sequence_face.overlay_label
                # lbl, dtctd = self.postprocess.face_info(self.h5data_idx)
                # stored_face.label = lbl
                # stored_face.overlay_label = lbl
                # stored_face.classes = [lbl.replace(' ', '_')]
                # if dtctd == self.postprocess.RECOGNIZED:
                #     stored_face.state = detector.DETECTED
                # elif dtctd == self.postprocess.DETECTED:

                # stored_face.detected =
                stored_faces.append(stored_face)
                self.h5data_face_idx += 1
                if self.h5data_face_idx >= len(self.h5data['face_frame_nums']):
                    break

            while self.h5data['person_frame_nums'][self.h5data_person_idx] == self.frame_idx:
                stored_person = detector.PersonInfo()
                stored_person.bbox = self.h5data['person_bboxes'][self.h5data_person_idx]
                stored_person.prob = self.h5data['person_probs'][self.h5data_person_idx]
                stored_persons.append(stored_person)
                self.h5data_person_idx += 1
                if self.h5data_person_idx >= len(self.h5data['person_frame_nums']):
                    break

        self.detector.process_frame(frame, overlays=overlays, stored_faces=stored_faces, stored_persons=stored_persons)

        for fi in self.detector.current_frame_faces:
            self.write_h5_if_needed(original_copy, face_info=fi)
        for pi in self.detector.current_frame_persons:
            self.write_h5_if_needed(original_copy, person_info=pi)

        self.research_processed(self.detector.current_frame_faces, frame=original_copy)

        self.postprocess_notify(self.detector.current_frame_faces)

        return self.detector.current_frame_faces

    def _h5_box_is_none(self, box):
        return all(box == self._h5_box_none())

    def _h5_box_none(self):
        return np.array([0, 0, 0, 0])

    def get_saved_faces(self):
        return self.saved_faces

    def save_face_infos(self, face_infos: [detector.FaceInfo]):
        self.saved_faces = face_infos

    def postprocess_notify(self, face_infos: [detector.FaceInfo], frame=None):
        now = time.time()
        if frame is None:
            frame = self.frame

        if self.postprocess_check:
            if now - self.postprocess_check < defaults.NOT_DETECTED_CHECK_PERIOD:
                return

        # Pick up not detected ones
        not_detected = [fi for fi in face_infos if fi.state == detector.NOT_DETECTED]

        # Check if there are embeddings like those which are already saved
        notify_list = []
        far_embeddings = []
        saved_faces = self.get_saved_faces()
        for nd in not_detected:

            if nd.embedding is None:
                continue

            found = False
            if saved_faces:
                for i in range(len(saved_faces)):
                    if distance.cosine(nd.embedding, saved_faces[i].embedding) <= defaults.DISTANCE_THRESHOLD:
                        if now - saved_faces[i].last_seen >= defaults.NOTIFY_THRESHOLD:
                            # If face was away for 1 min and then come back, generate notification.
                            saved_faces[i].overlay_label = (
                                'Person [id={}] came back after {} minutes'.format(
                                    saved_faces[i].label, int(round((now - saved_faces[i].last_seen) / 60))
                                )
                            )
                            notify_list.append(saved_faces[i])

                        # Update last seen
                        saved_faces[i].last_seen = now
                        saved_faces[i].bbox = nd.bbox
                        found = True
                        break

            if not saved_faces or not found:
                if not nd.label:
                    nd.label = '%s' % (len(far_embeddings) + 1)
                    nd.last_seen = now
                far_embeddings.append(nd)

        # Check previously saved faces for expiration
        i = 0
        while i < len(saved_faces):
            if now - saved_faces[i].last_seen >= defaults.EMBEDDING_EXPIRATION_TIME:
                LOG.info(
                    "Saved embedding [id={}] has been expired after {} sec".format(
                        saved_faces[i].label, defaults.EMBEDDING_EXPIRATION_TIME
                    )
                )
                saved_faces.remove(saved_faces[i])
            else:
                i += 1

        # Save unknown/far embeddings
        saved_faces += far_embeddings
        self.save_face_infos(saved_faces)
        # Notify about close embeddings
        for seen in notify_list:
            notify(text=seen.overlay_label, image=images.crop_by_box(frame, seen.bbox))

        self.postprocess_check = now

    def research_processed(self, face_infos: [detector.FaceInfo], frame=None):
        if frame is None:
            frame = self.frame

        for fd in self.faces_detected:
            self.faces_detected[fd].prepare()
        if face_infos is not None:
            store_not_detected = False
            now = time.time()
            if self.not_detected_store:
                if now - self.not_detected_check_ts > self.not_detected_check_period:
                    self.not_detected_check_ts = now
                    store_not_detected = True

            for fi in face_infos:
                if fi.state == detector.DETECTED:
                    if fi.classes and len(fi.classes) > 0:
                        name = fi.classes[0]
                        if name not in self.faces_detected:
                            self.faces_detected[name] = InVideoDetected()
                        self.faces_detected[name].exists_in_frame(face_info=fi, frame=frame)
                elif fi.state == detector.NOT_DETECTED and store_not_detected:
                    img = images.crop_by_box(frame, fi.bbox)
                    cv2.imwrite(os.path.join(self.not_detected_dir, '%s.jpg' % now), img)

        for fd in list(self.faces_detected):
            self.faces_detected[fd].exists_in_frame()
            if self.faces_detected[fd].not_detected_anymore and not self.faces_detected[fd].notified:
                del self.faces_detected[fd]
                continue
            if self.faces_detected[fd].make_notify():
                n = {'name': fd}
                fd_ = fd.replace(' ', '_')
                if fd_ in self.detector.meta:
                    meta = self.detector.meta[fd_]
                    if 'position' in meta:
                        n['position'] = meta['position']
                    if 'company' in meta:
                        n['company'] = meta['company']
                    if 'url' in meta:
                        n['url'] = meta['url']
                if self.faces_detected[fd].image is not None:
                    n['image'] = self.faces_detected[fd].image
                if len(self.faces_detected[fd].looks_like) > 0:
                    n['action_options'] = self.faces_detected[fd].looks_like.copy()
                self.notifies_queue.append(n)

    def write_h5_if_needed(self, frame, face_info: detector.FaceInfo = None, person_info: detector.PersonInfo = None):
        if not self.h5:
            return

        if face_info is not None:
            img = images.crop_by_box(frame, face_info.bbox)
            img_bytes = cv2.imencode('.jpg', img)[1].tostring()

            n = self.h5['face_bboxes'].shape[0]

            # resize +1
            self.h5['face_bboxes'].resize((n + 1, 4))
            self.h5['face_embeddings'].resize((n + 1, 512))  # todo set real shape size
            self.h5['face_person_bboxes'].resize((n + 1, 4))
            self.h5['face_probs'].resize((n + 1,))
            self.h5['face_head_poses'].resize((n + 1, 3))
            self.h5['face_frame_nums'].resize((n + 1,))
            self.h5['face_images'].resize((n + 1,))

            # Assign value
            self.h5['face_bboxes'][n] = face_info.bbox
            self.h5['face_embeddings'][n] = face_info.embedding
            self.h5['face_person_bboxes'][
                n] = self._h5_box_none() if face_info.person_bbox is None else face_info.person_bbox
            self.h5['face_probs'][n] = face_info.face_prob
            self.h5['face_head_poses'][n] = face_info.head_pose
            self.h5['face_frame_nums'][n] = self.frame_idx
            self.h5['face_images'][n] = np.fromstring(img_bytes, dtype='uint8')

        if person_info is not None:
            n = self.h5['person_bboxes'].shape[0]
            # resize +1
            self.h5['person_bboxes'].resize((n + 1, 4))
            self.h5['person_probs'].resize((n + 1,))
            self.h5['person_frame_nums'].resize((n + 1,))
            # Assign value
            self.h5['person_bboxes'][n] = person_info.bbox
            self.h5['person_probs'][n] = person_info.prob
            self.h5['person_frame_nums'][n] = self.frame_idx

    def listen(self):
        while True:
            err = None
            command, data = self.listener.listen()
            if command == 'reload_classifiers':
                LOG.info("reload classifiers")
                self.detector.load_classifiers()
            elif command == 'debug':
                deb = bool(data)
                LOG.info("set debug " + ("on" if deb else "off"))
                self.detector.debug = deb
            elif command == 'test':
                LOG.info("get test data:")
                LOG.info(data)
            else:
                err = ValueError('unknown command %s' % command)
            self.listener.result(err)

    def notify(self):
        while True:
            if len(self.notifies_queue) == 0:
                time.sleep(1)
            else:
                n = self.notifies_queue.pop()
                notify(**n)


def add_video_args(parser):
    parser.add_argument(
        '--video_source',
        help='Video source. If not set, current webcam is used (value 0).',
        default=None,
    )
    parser.add_argument(
        '--video_limit_sec',
        help='Limit video process by this time',
        default=0,
        type=int,
    )
    parser.add_argument(
        '--video_start_sec',
        help='Start processing video in the beginning of N second of video',
        default=0,
        type=int,
    )
    parser.add_argument(
        '--video_write_to',
        help='Write video to file',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--video_async',
        help='Asynchronous video (each frame does not wait for calculating boxes and labels).',
        action='store_true',
    )
    parser.add_argument(
        '--video_max_width',
        help='Resize video if width more than specified value.',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--video_max_height',
        help='Resize video if height more than specified value.',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--video_each_of_frame',
        help='Process every N frame (1 for not skipping).',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--build_h5py',
        help='Build h5py file with embeddings/data to file named as videofile '
             '(name han be changed with --build_h5py_to).',
        action='store_true',
    )
    parser.add_argument(
        '--build_h5py_to',
        help='Build h5py file with embeddings/data to file.',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--video_h5py',
        help='Get data from previously exported h5.',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--video_export_srt',
        help='Export SRT-file with detected/recognized persons',
        action='store_true',
    )
    parser.add_argument(
        '--video_export_srt_file',
        help='SRT file name (if not set - filename as for video)',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--video_no_output',
        help='Disable any output',
        action='store_true',
    )
    parser.add_argument(
        '--video_not_detected_store',
        help='Store not detected faces.',
        action='store_true',
    )
    parser.add_argument(
        '--video_not_detected_dir',
        help='Store not detected faces to specified directory.',
        type=str,
        default=defaults.NOT_DETECTED_DIR
    )
    parser.add_argument(
        '--video_not_detected_check_period',
        help='Check not detected faces for every N seconds.',
        type=int,
        default=defaults.NOT_DETECTED_CHECK_PERIOD,
    )
    parser.add_argument(
        '--postprocess',
        help='Make postprocess on saved h5 data.',
        action='store_true',
    )
    parser.add_argument(
        '--postprocess_clusterize_unrecognized',
        help='Clusterise unrecognized.',
        action='store_true',
    )
    parser.add_argument(
        '--postprocess_export_srt',
        help='Export SRT file.',
        action='store_true',
    )
    parser.add_argument(
        '--postprocess_export_srt_to',
        help='SRT file name.',
        type=str,
        default=None,
    )
