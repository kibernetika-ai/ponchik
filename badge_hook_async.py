import cv2
import numpy as np
import logging
import json
import svod_rcgn.badge.badge_detector as badge
import requests

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

PARAMS = {
    'people_service': '',
    'people_service_key': '',
}

detector = None
people = []


def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    global PARAMS
    PARAMS.update(params)
    people_service = PARAMS['people_service']
    people_service_key = PARAMS['people_service_key']
    global people
    if len(people_service) > 0:
        multipart_form_data = {
            'raw_input': ('', 'yes'),
            'string_action': ('', 'classes')
        }
        resp = requests.request('post', people_service, files=multipart_form_data,
                                headers={
                                    'Authorization': 'Bearer ' + people_service_key
                                })
        response = resp.json()
        if 'classes' in response:
            people = response['classes']
        else:
            logging.info('Bad response from people_service {}'.format(response))


LOG.info("Init hooks")


def find_people(image, ctx):
    data = cv2.resize(image, (300, 300), cv2.INTER_LINEAR)
    data = np.array(data).transpose([2, 0, 1]).reshape(1, 3, 300, 300)
    # convert to BGR
    data = data[:, ::-1, :, :]
    data = ctx.drivers[0].predict({'data': data})
    data = list(data.values())[0].reshape([-1, 7])
    bboxes_raw = data[data[:, 2] > 0.5]
    bounding_boxes = bboxes_raw[:, 3:7]
    bounding_boxes[:, 0] = bounding_boxes[:, 0] * image.shape[1]
    bounding_boxes[:, 2] = bounding_boxes[:, 2] * image.shape[1]
    bounding_boxes[:, 1] = bounding_boxes[:, 1] * image.shape[0]
    bounding_boxes[:, 3] = bounding_boxes[:, 3] * image.shape[0]
    detector.process(image, bounding_boxes)


def process(inputs, ctx):
    global detector
    if detector is None:
        detector = badge.BadgePorcessor(people,ctx.drivers[1], ctx.drivers[2], 0.5, 0.5)

    image = inputs['image'][0]
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    image = image[:, :, ::-1].copy()
    find_people(image, ctx)
    r_, buf = cv2.imencode('.png', image)
    image = np.array(buf).tostring()
    table = json.dumps([])
    return {
        'output': image,
        'table_output': table,
    }
