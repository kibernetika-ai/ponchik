from app.tools import print_fun


class NotifyPrint:

    @staticmethod
    def notify(**kwargs):
        msg_strings = ['%s has been detected' % kwargs['name']]
        if 'position' in kwargs:
            msg_strings.append('Position: %s' % kwargs['position'])
        if 'company' in kwargs:
            msg_strings.append('Company: %s' % kwargs['company'])
        if 'url' in kwargs:
            msg_strings.append('URL: %s' % kwargs['url'])
        if 'image' in kwargs:
            msg_strings.append('[IMAGE]')
        if 'action_options' in kwargs:
            msg_strings.append('[OPTIONS]')
            msg_strings.extend([' - ' + o for o in kwargs['action_options']])
        if 'system' in kwargs:
            msg_strings.append('System: %s' % kwargs['system'])
        sl = max([len(s) for s in msg_strings])
        print_fun('=' * sl)
        print_fun('\r\n'.join(msg_strings))
        print_fun('=' * sl)