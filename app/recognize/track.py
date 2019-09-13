from app.tools import images, utils

import logging
import time


class Beam:
    def __init__(self, id):
        self.positive = 0
        self.id = id
        self.embeding = None
        self.bbox = None
        self.fn = 0
        self.count = 0
        self.present = False
        self.present_since = 0
        self.last_present = 0
        self.meta = {}


class PearlDemoPresenter:
    def __init__(self,present_time,present_delay):
        self.beams = {}
        self.fn = 0
        self.count = 0
        self.present = ''
        self.present_meta = {}
        self.present_since = 0
        self.history = {}
        self.present_time = present_time
        self.present_delay = present_delay

    def process(self, faces):
        prev_fb = self.fn
        self.fn += 1
        for f in faces:
            if len(f.classes) > 0:
                current = f.classes[0]
                b = self.beams.get(current, Beam(current))
                if b.fn == 0:
                    self.beams[current] = b
                if b.positive < 5:
                    b.positive += 1
                b.meta = f.meta
                b.fn = self.fn
                b.embedding = f.embedding
                b.bbox = f.bbox
                b.negative = 0
            else:
                for k, v in self.beams.items():
                    if v.fn == prev_fb:
                        if utils.box_intersection(f.bbox, v.bbox) > 0.8:
                            if v.positive < 5:
                                v.positive += 1
                            v.fn = self.fn
                            v.embedding = f.embedding
                            v.bbox = f.bbox

        next = {}
        t = time.time()
        for k, v in self.beams.items():
            v.count += 1
            if v.count > 5:
                v.count = 5
            if v.fn != self.fn:
                v.fn = self.fn
                if v.positive > 0:
                    v.positive -= 1
            if v.count == 5 and v.positive == 0:
                if v.id == self.present:
                    # Active person
                    self.present = ''
                    self.present_meta = {}
                    self.present_since = 0
                    v.last_present = t
                    v.present = False
                if (t - v.last_present) > self.present_delay:
                    # remove
                    continue
            elif v.count == 5 and v.positive > 1:
                v.present = True
            next[k] = v
        self.beams = next
        if self.present != '':
            if (t - self.present_since) < self.present_time:
                #logging.info('Present {}'.format(self.present))
                return self.present_meta.get('name',None)
            else:
                self.beams[self.present].last_present=t
                self.present = ''
                self.present_meta = {}
                self.present_since = 0

        max_area = 0
        for k, v in self.beams.items():
            if not v.present:
                continue
            if (t - v.last_present) < self.present_delay:
                continue
            area = (v.bbox[2] - v.bbox[0]) * (v.bbox[3] - v.bbox[1])
            if area > max_area:
                if area>max_area*2:
                    max_area = area
                    self.present = k
                    self.present_meta = v.meta
                    self.present_since = t
                else:
                    max_area = area
                    self.present = ''
                    self.present_meta = {}
                    self.present_since = 0

        #logging.info('Present {}'.format(self.present))
        return self.present_meta.get('name',None)
