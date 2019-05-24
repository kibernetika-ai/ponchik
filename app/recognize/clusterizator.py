from sklearn.cluster import DBSCAN
import numpy as np


class Clusterizator:
    def __init__(self, eps=0.7):
        self.eps = eps

    def clusterize_frame_faces(self, frame_embs):
        embs_plain = []
        for e in frame_embs:
            embs_plain.extend(e)
        if len(embs_plain) == 0:
            return [[]] * len(frame_embs)

        embs = np.zeros((len(embs_plain), embs_plain[0].shape[1]))
        for i in range(len(embs_plain)):
            embs[i, :] = embs_plain[i]

        clt = DBSCAN(eps=self.eps)
        clt.fit(embs)

        faces_by_frames = []
        cur_label = 0
        for fe in frame_embs:
            if len(fe) == 0:
                faces_by_frames.append([])
            else:
                in_frame = clt.labels_[cur_label: cur_label+len(fe)]
                cur_label += len(fe)
                faces_by_frames.append(in_frame)

        return faces_by_frames
