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

    def notify(self, **kwargs):

        ch = '#%s' % self.channel
        has_image = 'image' in kwargs
        image_id = None

        if has_image:
            tmp_img = os.path.join(self.tempdir.name, str(time.time()) + '.jpg')
            try:
                cv2.imwrite(tmp_img, kwargs['image'])
                r = self.client.files_upload(
                    channels=ch,
                    file=tmp_img,
                    title=kwargs['name'],
                )
                assert r['ok']
                image_id = r['file']['id']
            except:
                has_image = False
            finally:
                os.remove(tmp_img)

        message_txt = '%s has been detected' % kwargs['name']

        msg_strings = [':exclamation: *%s* has been detected' % kwargs['name']]
        if 'position' in kwargs:
            msg_strings.append(':male-student: _position:_ %s' % kwargs['position'])
        if 'company' in kwargs:
            msg_strings.append(':classical_building: _company:_ %s' % kwargs['company'])
        if 'url' in kwargs:
            msg_strings.append(':link: %s' % kwargs['url'])
        if 'system' in kwargs:
            msg_strings.append(':desktop_computer: _%s_' % kwargs['system'])

        actions = []
        if has_image and image_id:
            if 'action_confirm' not in kwargs or kwargs['action_confirm']:
                actions.append({
                    'type': 'button',
                    'action_id': 'confirm-%s' % image_id,
                    'style': 'primary',
                    'text': {
                        'type': 'plain_text',
                        'text': ':white_check_mark: Yes, it\'s %s' % kwargs['name'],
                    }
                })
            # if 'action_unknown' not in kwargs or kwargs['action_unknown']:
            #     actions.append({
            #         'type': 'button',
            #         'action_id': 'unknown-%s' % image_id,
            #         'style': 'danger',
            #         'text': {
            #             'type': 'plain_text',
            #             'text': 'Unknown',
            #         }
            #     })
            if 'action_options' in kwargs and len(kwargs['action_options']) > 0:
                options = [{
                    "text": {
                        "type": "plain_text",
                        "text": o
                    },
                    "value": o
                } for o in kwargs['action_options']]
                actions.append({
                    'type': 'static_select',
                    'action_id': 'confirmopt-%s' % image_id,
                    'options': options,
                })
            if 'action_dialog' not in kwargs or kwargs['action_dialog']:
                actions.append({
                    'type': 'button',
                    'action_id': 'dialog-%s' % image_id,
                    'style': 'danger',
                    'text': {
                        'type': 'plain_text',
                        'text': 'Not in list',
                    }
                })
            # if len(actions) > 0:
            #     actions.append({
            #         'type': 'button',
            #         'action_id': 'close',
            #         'text': {
            #             'type': 'plain_text',
            #             'text': 'Close',
            #         }
            #     })

        blocks = [
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': '\r\n'.join(msg_strings)
                }
            },

        ]
        if len(actions) > 0:
            blocks.append({
                'type': 'actions',
                'elements': actions
            })

        self.client.chat_postMessage(channel=ch, text=message_txt, blocks=blocks)
