import os
import time

import slack
import tempfile
import cv2


class NotifySlack:
    def __init__(self, token, channel):
        self.channel = channel
        self.client = slack.WebClient(token=token)
        self.tempdir = tempfile.TemporaryDirectory()

    def is_ok(self):
        try:
            resp = self.client.conversations_list()
        except Exception as e:
            print('Unable to connect to Slack, probably token is not correct: %s' % e)
            return False
        for ch in resp['channels']:
            if ch['name'] == self.channel:
                return True
        print('Slack channel "%s" is not exists or not available' % self.channel)
        return False

    def notify(self, name, position=None, company=None, image=None):
        msg_strings = [':exclamation: *%s* has been detected' % name]
        if position:
            msg_strings.append(':male-student: _position:_ %s' % position)
        if company:
            msg_strings.append(':classical_building: _company:_ %s' % company)
        msg = '\r\n'.join(msg_strings)

        ch = '#%s' % self.channel
        if image is None:
            self.client.chat_postMessage(channel=ch, text=msg)
        else:
            tmp_img = os.path.join(self.tempdir.name, str(time.time()) + '.jpg')
            cv2.imwrite(tmp_img, image)
            self.client.files_upload(
                channels=ch,
                file=tmp_img,
                title=name,
                initial_comment=msg,
            )
            os.remove(tmp_img)
