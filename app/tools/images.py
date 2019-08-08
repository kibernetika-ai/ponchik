import cv2
import numpy as np


NORMALIZATION_STANDARD = "standard"
NORMALIZATION_FIXED = "fixed"
NORMALIZATION_PREWHITEN = "prewhiten"


def get_images(image, bounding_boxes, face_crop_size=112, face_crop_margin=32, normalization=None):
    images = []

    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(image.shape)[0:2]
        if nrof_faces > 1:
            for i in range(nrof_faces):
                det_arr.append(np.squeeze(det[i]))
        else:
            det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - face_crop_margin / 2, 0)
            bb[1] = np.maximum(det[1] - face_crop_margin / 2, 0)
            bb[2] = np.minimum(det[2] + face_crop_margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + face_crop_margin / 2, img_size[0])
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cv2.resize(cropped, (face_crop_size, face_crop_size), interpolation=cv2.INTER_AREA)
            #if normalization == NORMALIZATION_PREWHITEN:
            #    images.append(prewhiten(scaled))
            #elif normalization == NORMALIZATION_STANDARD:
            #    images.append(normalize(scaled))
            #elif normalization == NORMALIZATION_FIXED:
            #    images.append(fixed_normalize(scaled))
            #else:
            images.append(scaled)

    return images


def crop_by_boxes(img, boxes):
    crops = []
    for box in boxes:
        cropped = crop_by_box(img, box)
        crops.append(cropped)
    return crops


def crop_by_box(img, box):
    ymin = max([box[1], 0])
    ymax = min([box[3], img.shape[0]])
    xmin = max([box[0], 0])
    xmax = min([box[2], img.shape[1]])
    return img[ymin:ymax, xmin:xmax]


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def horizontal_flip(image):
    return np.fliplr(image)


def random_noise(image):
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss
    return noisy


def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def upscale(image):
    size = (image.shape[1], image.shape[0])
    image = cv2.resize(image, (30, 30))
    return cv2.resize(image, size)


# images normalization

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def fixed_normalize(x):
    return (x - 127.5) / 128.0


def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std
