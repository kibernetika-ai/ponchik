import os
import tempfile

from flask import Flask, request, make_response, Response
import requests
import json
import slack

app = Flask(__name__)
client = slack.WebClient(token=os.environ.get('SLACK_TOKEN'))


@app.route("/slack", methods=["POST"])
def message_options():

    data = json.loads(request.form["payload"])
    actor = data['user']
    res_string = ''

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
            if aaction == 'confirm':
                aimg_title = file['file']['title']
                res_string = res_string.replace('__NAME__', aimg_title)

        except:

            res_string = ':exclamation: Confirmed image is absent'

        # todo action to save just read image with specified name
        print("Store image as %s" % (aimg_title if aimg_title is not None else "UNKNOWN"))

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

    client.chat_update(channel=data['channel']['id'], **msg)
    return make_response('', 200)


@app.route("/probe", methods=["GET"])
def probe():
    return make_response('ok', 200)


if __name__ == '__main__':
    try:
        client.api_test()
    except:
        print('Unable to connect to Slack')
        exit(1)
    app.run(host='0.0.0.0', port=4242)
