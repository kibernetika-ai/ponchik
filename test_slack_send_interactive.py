import os
import slack

token, channel = os.environ.get('SLACK_TOKEN'), os.environ.get('SLACK_CHANNEL')
client = slack.WebClient(token=token)

client.chat_postMessage(channel=channel, text='test message')

# import cv2
# import os
#
# from svod_rcgn.notify import NotifySlack
#
# notifier = NotifySlack(
#     os.environ.get('SLACK_TOKEN'),
#     os.environ.get('SLACK_CHANNEL'),
# )
# if not notifier.is_ok():
#     print('Unable to connect to Slack')
#     exit(1)
#
# im = cv2.imread('/Users/vadim/Downloads/face_detection.png')
#
# notifier.notify(
#     name='John Dow',
#     position='CTO',
#     company='Horns and hooves',
#     image=im,
#     action_confirm=True,
#     action_unknown=True,
#     action_options=['Ivan Ivanov', 'Peter Petrov'],
# )

