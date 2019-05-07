import os
import time

import slack
import tempfile
import cv2


class NotifySlack:
    def __init__(self, token, channel):
        self.channel = '#%s' % channel
        self.client = slack.WebClient(token=token)
        self.tempdir = tempfile.TemporaryDirectory()
        # todo check

    def notify(self, name, position=None, company=None, image=None, image_title=None):
        msg_strings = [':exclamation: *%s* has been detected' % name]
        if position:
            msg_strings.append(':male-student: _position:_ %s' % position)
        if company:
            msg_strings.append(':classical_building: _company:_ %s' % company)
        msg = '\r\n'.join(msg_strings)

        if image is None:
            self.client.chat_postMessage(channel=self.channel, text=msg)
        else:
            tmp_img = os.path.join(self.tempdir.name, str(time.time()) + '.jpg')
            cv2.imwrite(tmp_img, image)
            self.client.files_upload(
                channels=self.channel,
                file=tmp_img,
                title=image_title,
                initial_comment=msg,
            )
            os.remove(tmp_img)
