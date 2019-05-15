import io
import os

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
def slack_interactive():

    data = json.loads(request.form["payload"])

    data_type = data['type']
    actor = data['user']
    channel = data['channel']['id']

    if data_type == 'block_actions':
        message = data['message']
        res_strings, res_comments = [], []
        for action in data['actions']:
            res_string, res_comment = process_block_action(action, data['trigger_id'], channel, message)
            if res_string:
                res_string = res_string.replace('__ACTOR__', '@%s' % actor['name'])
                res_strings.append(res_string)
            if res_comment:
                res_comments.append(res_comment)
        if len(res_strings) > 0:
            complete_message(
                message,
                channel,
                ', '.join(res_strings),
                ', '.join(res_comments),
            )
        return make_response('', 200)

    elif data_type == 'dialog_submission':
        message, channel, res, comment = process_dialog_submission(data)
        if res:
            res = res.replace('__ACTOR__', '@%s' % actor['name'])
        complete_message(
            message,
            channel,
            res,
            comment,
        )
        return make_response('', 200)

    else:
        return make_response('unhandled type "%s"' % data_type, 400)


@app.route("/probe", methods=["GET"])
def probe():
    return make_response('ok', 200)


def init_err(err):
    print(err)
    exit(1)


def process_block_action(action, trigger_id, channel, message):

    res_string = None
    res_comment = None

    aid = action['action_id']
    aparts = aid.split('-')
    aaction = aparts[0]
    aimg_id = None
    aimg_title = None

    if aaction == 'dialog':
        dialog = {
                "callback_id": "save-%s" % aparts[1],
                "title": "Hope you know who is it!",
                "submit_label": "Save",
                'state': json.dumps({'channel': channel, 'message': message}),
                "elements": [
                    {
                        "type": "text",
                        "label": "Name",
                        "name": "name",
                    },
                    {
                        "type": "text",
                        "label": "Position",
                        "name": "position",
                        "optional": True,
                    },
                    {
                        "type": "text",
                        "label": "Company",
                        "name": "company",
                        "optional": True,
                    },
                    {
                        "type": "text",
                        "label": "URL",
                        "name": "url",
                        "optional": True,
                    },
                ]
            }
        client.dialog_open(
            dialog=dialog,
            trigger_id=trigger_id
        )
        return None, None
    elif aaction == 'unknown':
        res_string = ':question: __ACTOR__ set this face as *Unknown*'
        aimg_id = aparts[1]
    elif aaction == 'confirm':
        res_string = ':white_check_mark: __ACTOR__ confirmed this face as *__NAME__*'
        aimg_id = aparts[1]
    elif aaction == 'confirmopt':
        res_string = ':white_check_mark: __ACTOR__ selected this face as *__NAME__*__META__'
        aimg_title = action['selected_option']['value']
        aimg_id = aparts[1]
    elif aaction == 'close':
        res_string = '__ACTOR__ closed clarify options'

    ok, res_string, res_comment = store_face(aimg_id, aimg_title, result=res_string, comment=res_comment)

    if not ok:
        res_string = ":exclamation: %s" % res_string

    return res_string, res_comment


def process_dialog_submission(submission):
    cid = submission['callback_id']
    cparts = cid.split('-')
    caction = cparts[0]
    msg_data = json.loads(submission['state'])

    message, channel, res, comment = msg_data['message'], msg_data['channel'], None, None

    if caction == 'save':
        aimg_id = cparts[1]
        s = submission['submission']

        ok, res, comment = store_face(
            aimg_id,
            s['name'],
            result=':white_check_mark: __ACTOR__ store this face as *__NAME__*',
            position=s['position'],
            company=s['company'],
            url=s['url'],
        )

    return message, channel, res, comment


def store_face(file_id, name, position=None, company=None, url=None, result=None, comment=None):

    ok = True
    file_bytes = None

    try:

        file = client.api_call("files.info", http_verb="GET", params={'file': file_id})
        f = requests.get(
            file['file']['url_private_download'],
            headers={'Authorization': 'Bearer %s' % client.token},
            stream=True,
        )
        assert f.status_code == 200

        file_bytes = io.BytesIO()
        for chunk in f.iter_content(chunk_size=8192):
            if chunk:
                file_bytes.write(chunk)
        file_bytes.seek(0)
        if name is None:
            name = file['file']['title']

    except Exception as e:

        result = "Confirmed image is absent"
        comment = 'Get Slack file info error: %s' % e
        file_bytes = None
        ok = False

    if file_bytes is not None:

        try:
            data = {
                'raw_input': 'true',
                'string_action': 'clarify',
                'string_name': name,
            }
            if position is not None:
                data['string_position'] = position
            if company is not None:
                data['string_company'] = company
            if url is not None:
                data['string_url'] = url
            r = requests.post(
                serving_url,
                data=data,
                files={
                    'byte_face': file_bytes
                }
            )
            assert r.status_code == 200
            result = result.replace('__NAME__', name)

        except Exception as e:

            result = "Upload to serving failed"
            comment = 'Serving upload error: %s' % e
            ok = False

    if '__META__' in result:
        meta_str = ''
        try:
            r = requests.post(
                serving_url,
                data={'raw_input': 'true'},
                files={'string_action': 'meta'},
            )
            assert r.status_code == 200
            full_meta = json.loads(r.text)
            full_meta_arr = json.loads(full_meta['meta'])
            meta = None
            for m in full_meta_arr:
                if 'name' in m and m['name'] == name:
                    meta = m
                    break
            if meta is not None:
                meta_strs = []
                if 'position' in meta:
                    meta_strs.append('position: %s' % meta['position'])
                if 'company' in meta:
                    meta_strs.append('company: %s' % meta['company'])
                meta_str = ', '.join(meta_strs)
        except Exception as e:
            print('Load metadata error: %s' % e)
            meta_str = ':exclamation: Unable to load metadata'

        if meta_str != '':
            meta_str = ' _(%s)_' % meta_str
        result = result.replace('__META__', meta_str)


    return ok, result, comment


def complete_message(message, channel, result, comment):
    update = False
    for b, bl in enumerate(message['blocks']):
        if bl['type'] == 'actions':
            del message['blocks'][b]
            update = True
    if len(result) > 0:
        message['blocks'].append({
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': result,
            }
        })
        update = True
    if update:
        client.chat_update(channel=channel, **message)


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
