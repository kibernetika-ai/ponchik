from app.tools import images, utils


class Beam:
    def __init__(self, id):
        self.positive = 0
        self.id = id
        self.embeding = None
        self.bbox = None
        self.fn = 0


class PearlDemoPresenter:
    def __init__(self):
        self.beams = {}
        self.fn = 0
        self.count = 0

    def process(self, faces):
        prev_fb = self.fn
        self.fn += 1
        for f in faces:
            if len(f.classes) > 0:
                current = f.classes[0]
                b = self.beams.get(current, Beam(current))
                if b.fn == 0:
                    self.beams[current] = b
                b.positive += 1
                b.fn = self.fn
                b.embedding = f.embedding
                b.bbox = f.bbox
                b.negative = 0
            else:
                for k, v in self.beams.items():
                    if v.fn == prev_fb:
                        if utils.box_intersection(f.bbox, v.bbox) > 0.8:
                            v.positive += 1
                            v.fn = self.fn
                            v.embedding = f.embedding
                            v.bbox = f.bbox

        next = {}
        self.count += 1
        if self.count > 5:
            self.count = 5
        for k, v in self.beams.items():
            if v.fn != self.fn:
                v.fn = self.fn
            if v.positive>0:
                v.positive -= 1
            next[k] = v
        self.beams = next
