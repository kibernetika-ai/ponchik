import os
import tempfile
import time

from flask import Flask, request, make_response
import requests
import json
import slack

slack_token = os.environ.get('SLACK_TOKEN')
if not slack_token:
    print('Env SLACK_TOKEN not set')
    exit(1)
serving_url = os.environ.get('SERVING_REQUEST_URL')
if not serving_url:
    print('Env SERVING_REQUEST_URL not set')
    exit(1)

app = Flask(__name__)
client = slack.WebClient(token=slack_token)


@app.route("/slack", methods=["POST"])
def message_options():

    data = json.loads(request.form["payload"])
    actor = data['user']
    res_string = ''
    res_comment = None

    action = data['actions'][0]
    atype = action['type']
    aid = action['action_id']
    aparts = aid.split('-')
    aaction = aparts[0]
    aimg_id = None
    aimg_title = None
    if atype == 'button':
        if aaction == 'unknown':
            res_string = ':question: @%s set this face as *Unknown*' % actor['name']
            aimg_id = aparts[1]
        elif aaction == 'confirm':
            res_string = ':white_check_mark: @%s confirmed this face as *__NAME__*' % actor['name']
            aimg_id = aparts[1]
        elif aaction == 'close':
            res_string = '@%s closed clarify options' % actor['name']

    if aimg_id:

        img_name = None

        try:
            file = client.api_call("files.info", http_verb="GET", params={'file': aimg_id})
            f = requests.get(
                file['file']['url_private_download'],
                headers={'Authorization': 'Bearer %s' % client.token},
                stream=True,
            )
            assert f.status_code == 200
            tf = tempfile.NamedTemporaryFile()
            for chunk in f:
                tf.write(chunk)
            img_name = tf.name
            time.sleep(1)
            if aaction == 'confirm':
                aimg_title = file['file']['title']
                res_string = res_string.replace('__NAME__', aimg_title)

        except Exception as e:

            res_string = ':exclamation: Confirmed image is absent'
            res_comment = 'Get Slack file info error: %s' % e

        if img_name:

            try:

                r = requests.post(
                    serving_url,
                    data={
                        'raw_input': 'true',
                        'string_action': 'clarify',
                        'string_name': aimg_title,
                    },
                    files={
                        'byte_face': open(img_name, 'rb')
                    }
                )
                assert r.status_code == 200

            except Exception as e:

                res_string = ':exclamation: Upload to serving failed'
                res_comment = 'Serving upload error: %s' % e

    msg = data['message']
    for b, bl in enumerate(msg['blocks']):
        if bl['type'] == 'actions':
            del msg['blocks'][b]
    msg['blocks'].append({
        'type': 'section',
        'text': {
            'type': 'mrkdwn',
            'text': res_string,
        }
    })
    if res_comment:
        msg['blocks'].append({
            'type': 'context',
            'text': {
                'type': 'mrkdwn',
                'text': res_comment,
            }
        })

    client.chat_update(channel=data['channel']['id'], **msg)
    return make_response('', 200)


@app.route("/probe", methods=["GET"])
def probe():
    return make_response('ok', 200)


def init_err(err):
    print(err)
    exit(1)


if __name__ == '__main__':
    try:
        client.api_test()
    except:
        init_err('Unable to connect to Slack')
    try:
        r = requests.post(
            serving_url,
            data={'raw_input': 'true'},
            files={'string_action': 'test'},
        )
        if r.status_code != 200:
            init_err('Serving test method should return 200, but it sends %d' % r.status_code)
    except Exception as e:
        init_err('Check serving request URL error: %s' % e)
    app.run(host='0.0.0.0', port=4242)
