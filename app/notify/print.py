from app.tools import utils


class NotifyPrint:

    @staticmethod
    def notify(**kwargs):
        override_text = kwargs.get('text')
        if not override_text:
            msg_strings = ['%s has been detected' % kwargs['name']]
        else:
            msg_strings = ['%s' % kwargs['text']]
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
        utils.print_fun('=' * sl)
        utils.print_fun('\r\n'.join(msg_strings))
        utils.print_fun('=' * sl)
