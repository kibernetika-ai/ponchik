import datetime
import os


class NotifyFile:

    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(self.filename):
            open(self.filename, 'a').close()
        elif os.path.isdir(self.filename):
            raise ValueError('%s is dir, unable to log' % self.filename)

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
        with open(self.filename, 'a') as f:
            f.write('{}: {}'.format(datetime.datetime.now(), ', '.join(msg_strings)))
