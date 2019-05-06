from svod_rcgn.tools.print import print_fun


class NotifyPrint:
    def __init__(self):
        pass

    def notify(self, message):
        print_fun("=" * len(message))
        print_fun(message)
        print_fun("=" * len(message))
