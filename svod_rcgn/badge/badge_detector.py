import cv2
import numpy as np
import math
import threading
import queue
import sys
import logging
from itertools import permutations
import fuzzyset

MAX_DIM = 1024.0
ENGLISH_CHAR_MAP = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    "'",
    " ",
    '_'
]


def read_charset():
    charset = {}
    inv_charset = {}
    for i, v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset, inv_charset


def findRoot(point, group_mask):
    root = point
    update_parent = False
    stop_loss = 1000
    while group_mask[root] != -1:
        root = group_mask[root]
        update_parent = True
        stop_loss -= 1
        if stop_loss < 0:
            raise Exception('Stop loss')
    if update_parent:
        group_mask[point] = root
    return root


def join(p1, p2, group_mask):
    root1 = findRoot(p1, group_mask)
    root2 = findRoot(p2, group_mask)
    if root1 != root2:
        group_mask[root1] = root2


def get_all(points, w, h, group_mask):
    root_map = {}
    mask = np.zeros((h, w), np.int32)
    for i in range(len(points[0])):
        point_root = findRoot(points[1][i] + points[0][i] * w, group_mask)
        if root_map.get(point_root, None) is None:
            root_map[point_root] = len(root_map) + 1
        mask[points[0][i], points[1][i]] = root_map[point_root]
    return mask


def decodeImageByJoin(cls, links, cls_threshold, link_threshold):
    h = cls.shape[0]
    w = cls.shape[1]
    pixel_mask = cls >= cls_threshold
    link_mask = links >= link_threshold
    y, x = np.where(pixel_mask == True)
    group_mask = {}
    for i in range(len(x)):
        if pixel_mask[y[i], x[i]]:
            group_mask[y[i] * w + x[i]] = -1
    for i in range(len(x)):
        neighbour = 0
        for ny in range(y[i] - 1, y[i] + 2):
            for nx in range(x[i] - 1, x[i] + 2):
                if nx == x[i] and ny == y[i]:
                    continue
                if nx >= 0 and nx < w and ny >= 0 and ny < h:
                    pixel_value = pixel_mask[ny, nx]
                    link_value = link_mask[ny, nx, neighbour]
                    if pixel_value and link_value:
                        join(y[i] * w + x[i], ny * w + nx, group_mask)
                neighbour += 1
    return get_all((y, x), w, h, group_mask)


def maskToBoxes(mask, image_size, min_area=200, min_height=6):
    bboxes = []
    min_val, max_val, _, _ = cv2.minMaxLoc(mask)
    resized_mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    for i in range(int(max_val)):
        bbox_mask = resized_mask == (i + 1)
        bbox_mask = bbox_mask.astype(np.int32)
        contours, _ = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            continue
        r = cv2.minAreaRect(contours[0])
        maxarea = 0
        maxc = None
        for j in contours:
            area = cv2.contourArea(j)
            if area > maxarea:
                maxarea = area
                maxc = j
        if maxc is not None and maxarea > 36:
            r = cv2.minAreaRect(maxc)
            if min(r[1][0], r[1][1]) < min_height:
                continue
            bboxes.append(r)
    return bboxes


def norm_image_for_text_prediction(im, infer_height, infer_width):
    w = im.shape[1]
    h = im.shape[0]
    ratio = h / infer_height
    width = int(w / ratio)
    height = int(h / ratio)
    width = min(infer_width, width)
    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_CUBIC)
    pw = max(0, infer_width - im.shape[1])
    ph = max(0, infer_height - im.shape[0])
    im = np.pad(im, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
    return im


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


class BadgePorcessor(object):
    def __init__(self, people, text_detection, ocr, pixel_threshold=0.5, link_threshold=0.5):
        self.pixel_threshold = pixel_threshold
        self.link_threshold = link_threshold
        self.text_detection = text_detection
        self.ocr = ocr
        charset, _ = read_charset()
        self.chrset_index = charset
        self.names_db = fuzzyset.FuzzySet()
        self.data_db = {}
        for p in people:
            logging.info('Add: {}'.format(p))
            tokens = p.split(' ')
            for t in tokens:
                if len(t) > 1:
                    self.names_db.add(t)
            perm = permutations(tokens)
            for v in list(perm):
                v1 = ' '.join(v)
                v2 = ''.join(v)
                self.names_db.add(v1)
                self.names_db.add(v2)
                self.data_db[v1] = p
                self.data_db[v2] = p
        self.queue = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self.run)
        self.worker.start()

    def process(self, image, faces):
        self.queue.put_nowait({'image': image, 'faces': faces})

    def run(self):
        while True:
            try:
                logging.info('Start exec new task')
                data = self.queue.get()
                self._process(data)
                logging.info('End exec new task')
            except:
                logging.info('Failed to process: {}'.format(sys.exc_info()))

    def fix_length(self, l, b):
        return int(math.ceil(l / b) * b)

    def adjust_size(self, image):
        w = image.shape[1]
        h = image.shape[0]
        if w > h:
            if w > MAX_DIM:
                ratio = MAX_DIM / float(w)
                h = int(float(h) * ratio)
                w = MAX_DIM
        else:
            if h > MAX_DIM:
                ratio = MAX_DIM / float(h)
                w = int(float(w) * ratio)
                h = MAX_DIM
        w = self.fix_length(w, 32)
        h = self.fix_length(h, 32)
        return cv2.resize(image, (w, h))

    def get_text(self, predictions):
        line = []
        end_line = len(self.chrset_index) - 1
        for i in predictions:
            if i == end_line:
                break
            t = self.chrset_index.get(i, -1)
            if t == -1:
                continue
            line.append(t)
        return ''.join(line)

    def extract_text(self, image):
        image = np.expand_dims(image.astype(np.float32) / 255.0, 0)
        outputs = self.ocr.predict({'images': image})
        text = outputs['output'][0]
        text = self.get_text(text)
        return text

    def choose_one(self, candidates):
        a = []
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i == j:
                    continue
                full_name = candidates[i] + ' ' + candidates[j]
                e = self.names_db.get(full_name)
                if (e is not None) and len(e) > 0:
                    a.append(e[0])
        if len(a) < 1:
            return None
        a.sort(key=lambda tup: tup[0])
        a = a[len(a) - 1]
        if a[0] < 0.6:
            return None
        return a

    def badge_select(self, image, face):
        box_image = self.adjust_size(image)
        box_image = box_image.astype(np.float32) / 255.0
        box_image = np.expand_dims(box_image, 0)
        outputs = self.text_detection.predict({'image': box_image})
        cls = outputs['pixel_pos_scores'][0]
        links = outputs['link_pos_scores'][0]
        mask = decodeImageByJoin(cls, links, self.pixel_threshold, self.link_threshold)
        bboxes = maskToBoxes(mask, (image.shape[1], image.shape[0]))
        texts = []
        logging.info('BBoxes: {}'.format(bboxes))
        for i in range(len(bboxes)):
            box = np.int0(cv2.boxPoints(bboxes[i]))
            maxp = np.max(box, axis=0) + 2
            minp = np.min(box, axis=0) - 2
            y1 = max(0, minp[1])
            y2 = min(image.shape[0], maxp[1])
            x1 = max(0, minp[0])
            x2 = min(image.shape[1], maxp[0])
            text_img = image[y1:y2, x1:x2, :]
            if text_img.shape[0] < 1 or text_img.shape[1] < 1:
                continue
            if bboxes[i][1][0] > bboxes[i][1][1]:
                angle = -1 * bboxes[i][2]
            else:
                angle = -1 * (90 + bboxes[i][2])
            if angle != 0:
                text_img = rotate_bound(text_img, angle)

            text_img = norm_image_for_text_prediction(text_img, 32, 320)
            text = self.extract_text(text_img)
            if len(text) > 0:
                logging.info('Text: {}'.format(text))
                texts.append(text)
        candidates = []
        found_name = None
        for text in texts:
            if len(text) > 1:
                found = self.names_db.get(text)
                if (found is not None) and (len(found) > 0):
                    if found[0][0] > 0.7:
                        text = found[0][1]
                        if ' ' in text:
                            found_name = (found[0][0], text)
                            candidates = []
                            break
                        else:
                            candidates.append(text)
        if (found_name is None) and len(candidates) > 0:
            found_name = self.choose_one(candidates)
        if found_name is not None and (found_name[1] in self.data_db):
            name = self.data_db[found_name[1]]
            logging.info('Found name: {}'.format(name))
            # TODO Notify face

    def find_people(self, faces, image):
        h = image.shape[0]
        w = image.shape[1]
        if faces is not None:
            for box in faces:
                xmin0 = int(box[0])
                ymin0 = int(box[1])
                xmax0 = int(box[2])
                ymax0 = int(box[3])
                bw = xmax0 - xmin0
                bh = ymax0 - ymin0
                xmin = max(xmin0 - int(bw / 2), 0)
                xmax = min(xmax0 + int(bw / 2), w)
                ymax = min(ymax0 + bh * 3, h)
                ymin = ymin0
                box_image_original = image[ymin:ymax, xmin:xmax, :]
                face = image[ymin0:ymax0, xmin0:xmax0, :]
                logging.info('Process new face')
                self.badge_select(box_image_original, face)

    def _process(self, data):
        self.find_people(data['faces'], data['image'])
