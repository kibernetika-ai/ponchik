from svod_rcgn.tools.print import print_fun


class NotifyPrint:

    @staticmethod
    def notify(**kwargs):
        msg_strings = ['%s has been detected' % kwargs['name']]
        if 'position' in kwargs:
            msg_strings.append('Position: %s' % kwargs['position'])
        if 'company' in kwargs:
            msg_strings.append('Company: %s' % kwargs['company'])
        if 'image' in kwargs:
            msg_strings.append('[IMAGE]')
        sl = max([len(s) for s in msg_strings])
        print_fun("=" * sl)
        print_fun('\r\n'.join(msg_strings))
        print_fun("=" * sl)
