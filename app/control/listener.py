import pickle
from app.tools import print_fun

import zmq

DEFAULT_LISTENER_PORT = 43210


class SVODListener:
    def __init__(self, port=DEFAULT_LISTENER_PORT):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind('tcp://127.0.0.1:%d' % port)

    def listen(self):
        while True:
            try:
                command, data = pickle.loads(self.socket.recv())
                # test, remove
                if command == "error":
                    raise ValueError("test error")
                return command, data
            except Exception as e:
                print_fun("listener exception: %s" % e)
                self.result(e)

    def result(self, err=None):
        self.socket.send(pickle.dumps(err))
