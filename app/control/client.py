import pickle

import zmq

from app.control import listener
from app.tools import utils


class Client:
    def __init__(self, port=listener.DEFAULT_LISTENER_PORT):
        self.port = port
        self.context = None
        self.socket = None

    def call(self, command, data=None):
        if self.socket is None:
            # https://stackoverflow.com/questions/44273941/python-script-not-terminating-after-timeout-in-zmq-recv
            nIOthreads = 2
            self.context = zmq.Context(nIOthreads)
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.AFFINITY, 1)
            self.socket.setsockopt(zmq.RCVTIMEO, 2000)
            self.socket.connect('tcp://127.0.0.1:%d' % self.port)
        self.socket.send(pickle.dumps((command, data)))
        try:
            err = pickle.loads(self.socket.recv())
        except zmq.error.Again:
            err = ValueError("no response, is server available?")
            utils.print_fun(err)
            self.socket.close()
            self.context.term()
            self.socket = None
            self.context = None
        return err
