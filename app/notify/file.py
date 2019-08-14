import datetime
import os


class NotifyFile:

    def __init__(self, filename):
        self.filename = filename
        if os.path.exists(self.filename):
            os.remove(self.filename)
        self.out = open(self.filename, 'w')

    def notify(self, **kwargs):
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
        self.out.write('{}: {}\n'.format(datetime.datetime.now(), ', '.join(msg_strings)))
        self.out.flush()
