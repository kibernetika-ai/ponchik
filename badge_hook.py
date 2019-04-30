import cv2
import numpy as np
import logging
import json
import base64
import math
import fuzzyset

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

ENGLISH_CHAR_MAP = [
    '#',
    # Alphabet normal
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    "'",
    " ",
    '_'
]

names_db = fuzzyset.FuzzySet()
data_db = {}

def read_charset():
    charset = {}
    inv_charset = {}
    for i, v in enumerate(ENGLISH_CHAR_MAP):
        charset[i] = v
        inv_charset[v] = i

    return charset, inv_charset


chrset_index = {}


def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    charset, _ = read_charset()
    global chrset_index
    chrset_index = charset

    names_db.add('stas khirman')
    names_db.add('staskhirman')
    names_db.add('khirman stas')
    names_db.add('khirmanstas')
    names_db.add('stas')
    names_db.add('khirman')

    data_db['stas khirman'] = 'Stas Khirman'
    data_db['khirman stas'] = 'Stas Khirman'
    data_db['staskhirman'] = 'Stas Khirman'
    data_db['khirmanstas'] = 'Stas Khirman'

    LOG.info("Init hooks")


def fix_length(l, b):
    return int(math.ceil(l / b) * b)


MAX_DIM = 1024.0


def adjust_size(image):
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
    w = fix_length(w, 32)
    h = fix_length(h, 32)
    return cv2.resize(image, (w, h))


def extract_text(image, ctx):
    image = np.expand_dims(image.astype(np.float32) / 255.0, 0)
    outputs = ctx.drivers[2].predict({'images': image})
    text = outputs['output'][0]
    text = get_text(text)
    logging.info('Found: {}'.format(text))
    return text


def choose_one(candidates):
    a = []
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i == j:
                continue
            full_name = candidates[i] + ' ' + candidates[j]
            e = names_db.get(full_name)
            if (e is not None) and len(e) > 0:
                a.append(e[0])
    if len(a) < 1:
        return None
    a.sort(key=lambda tup: tup[0])
    a = a[len(a) - 1]
    if a[0] < 0.6:
        return None
    return a

def badge_select(image,face, draw_image, offset, ctx, table):
    box_image = adjust_size(image)
    box_image = box_image.astype(np.float32) / 255.0
    box_image = np.expand_dims(box_image, 0)
    outputs = ctx.drivers[1].predict({'image': box_image})
    cls = outputs['pixel_pos_scores'][0]
    links = outputs['link_pos_scores'][0]
    mask = decodeImageByJoin(cls, links, ctx.pixel_threshold, ctx.link_threshold)
    bboxes = maskToBoxes(mask, (image.shape[1], image.shape[0]))
    texts = []
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
        text = extract_text(text_img, ctx)
        if len(text)>0:
            texts.append(text)
    candidates = []
    found_name = None
    for text in texts:
        if len(text) > 1:
            found = names_db.get(text)
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
        found_name = choose_one(candidates)
    if found_name is not None and (found_name[1] in data_db):
        if ctx.add_text_image == 'text':
            _, buf = cv2.imencode('.png', face[:, :, ::-1])
            buf = np.array(buf).tostring()
            face = base64.encodebytes(buf).decode()
        table.append({
            'type': 'text',
            'name': data_db[found_name[0]],
            'prob': float(found_name[1]),
            'image': face
        })
    for i in bboxes:
        box = cv2.boxPoints(i)
        box = np.int0(box)
        box = box + np.array([[offset[0], offset[1]]])
        draw_image = cv2.drawContours(draw_image, [box], 0, (255, 0, 0), 2)
    return table, draw_image


def find_people(image, draw_image, ctx, table):
    data = cv2.resize(image, (300, 300), cv2.INTER_LINEAR)
    data = np.array(data).transpose([2, 0, 1]).reshape(1, 3, 300, 300)
    # convert to BGR
    data = data[:, ::-1, :, :]
    data = ctx.drivers[0].predict({'data': data})
    data = list(data.values())[0].reshape([-1, 7])
    bboxes_raw = data[data[:, 2] > 0.25]

    w = image.shape[1]
    h = image.shape[0]
    if bboxes_raw is not None:
        for box in bboxes_raw:
            xmin0 = int(box[3] * w)
            ymin0 = int(box[4] * h)
            xmax0 = int(box[5] * w)
            ymax0 = int(box[6] * h)
            bw = xmax0 - xmin0
            bh = ymax0 - ymin0
            xmin = max(xmin0 - int(bw / 2), 0)
            xmax = min(xmax0 + int(bw / 2), w)
            ymax = min(ymax0 + bh * 3, h)
            ymin = ymin0
            box_image_original = image[ymin:ymax, xmin:xmax, :]
            xmin = max(xmin0 - int(bw / 2), 0)
            xmax = min(xmax0 + int(bw / 2), w)
            ymax = min(ymax0 + int(bh /2), h)
            ymin = max(ymin0 - int(bh / 2), 0)
            face = image[ymin:ymax, xmin:xmax, :]
            table, draw_image = badge_select(box_image_original,face, draw_image, (xmin, ymin), ctx, table)
            draw_image = cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

    return table, draw_image


def process_internal(inputs, ctx):
    image = inputs['image']
    ctx.pixel_threshold = float(inputs.get('pixel_threshold', 0.5))
    ctx.link_threshold = float(inputs.get('link_threshold', 0.5))
    ctx.add_text_image = 'original'
    table, image = find_people(image[:, :, ::-1].copy(), image, ctx, [])
    return {
        'output': image,
        'table_output': table,
    }


def process(inputs, ctx):
    ctx.add_text_image = 'text'
    image = inputs['image'][0]
    ctx.pixel_threshold = float(inputs.get('pixel_threshold', 0.5))
    ctx.link_threshold = float(inputs.get('link_threshold', 0.5))
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    table, image = find_people(image[:, :, ::-1].copy(), image, ctx, [])
    r_, buf = cv2.imencode('.png', image)
    image = np.array(buf).tostring()
    table = json.dumps(table)
    return {
        'output': image,
        'table_output': table,
    }


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
        # if min(r[1][0], r[1][1]) < min_height:
        #    logging.info('Skip size box {} {}'.format(r, i + 1))
        #    continue
        # if r[1][0] * r[1][1] < min_area:
        #    logging.info('Skip area box {} {}'.format(r, i + 1))
        #    continue
        # bboxes.append(r)
    return bboxes


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


def get_text(predictions):
    line = []
    end_line = len(chrset_index) - 1
    for i in predictions:
        if i == end_line:
            break
        t = chrset_index.get(i, -1)
        if t == -1:
            continue
        line.append(t)
    return ''.join(line)


def norm_image_for_text_prediction(im, infer_height, infer_width):
    w = im.shape[1]
    h = im.shape[0]
    # ration_w = max(w / infer_width, 1.0)
    # ration_h = max(h / infer_height, 1.0)
    # ratio = max(ration_h, ration_w)
    ratio = h / infer_height
    # if ratio > 1:
    width = int(w / ratio)
    height = int(h / ratio)
    width = min(infer_width, width)
    # im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)

    # im[np.greater(im,200)]=255
    # im[np.less(im,100)]=0

    # im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)

    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_CUBIC)

    pw = max(0, infer_width - im.shape[1])
    ph = max(0, infer_height - im.shape[0])
    im = np.pad(im, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
    return im
