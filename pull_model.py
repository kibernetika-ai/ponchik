import datetime
import io
import time

import croniter
import requests


versions = {}


def version_request(base, ws, name, token):
    url = '%s/workspace/%s/mlmodel/%s/versions' % (
        base, ws, name
    )
    resp = requests.get(url, headers={'Authorization': 'Bearer %s' % token})
    return resp


def download_version(url, token):
    resp = requests.get(url, headers={'Authorization': 'Bearer %s' % token}, stream=True)
    content = io.BytesIO()
    length = int(resp.headers['Content-Length'])
    read = 0
    max_len = 65
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if chunk:
            read += content.write(chunk)

            progress = '=' * int(float(read) / length * max_len - 1)
            progress += '>'
            progress += ' ' * (max_len - len(progress))
            print('Downloading... [%s]' % progress)

    # return cursor to 0
    content.seek(0)
    return content


def loop(pattern, base_url, ws, name, token, callback):
    cron = croniter.croniter(pattern, datetime.datetime.utcnow())
    next_time = cron.get_next(datetime.datetime)

    while True:
        current = datetime.datetime.utcnow()
        if current >= next_time:
            print('[%s] Check for new version... [model=%s/%s]' % (
                next_time.strftime('%Y-%m-%d %H:%M:%S'), ws, name
                )
            )
            changed, version, url = check(base_url, ws, name, token)
            if changed:
                # Download
                print('[%s] Downloading new version %s...' % (next_time.strftime('%Y-%m-%d %H:%M:%S'), version))
                fileobj = download_version(url, token)
                callback(version, fileobj)
            next_time = cron.get_next(datetime.datetime)

        time.sleep(1)


def check(base_url, ws, name, token):
    try:
        resp = version_request(base_url, ws, name, token)
    except Exception as e:
        print('Error: %s' % e)
        return None, None, None

    if resp.status_code >= 400:
        print('Response status %s.' % resp.status_code)
        print(resp.text)
        return None, None, None

    try:
        vs = resp.json()
    except (ValueError, TypeError) as e:
        print('not json: %s' % e)
        return None, None, None

    global versions
    if len(versions) == 0:

        for v in vs:
            versions[v['Version']] = v
        return None, None, None

    for v in vs:
        should_add = False
        if v['Status'] != 'ok':
            continue

        if v['Version'] not in versions:
            print('New version: %s' % v['Version'])
            should_add = True

        # Changed size
        for _, old in versions.items():
            if v['Version'] == old['Version'] and old['SizeBytes'] != v['SizeBytes']:
                print(
                    'Old version %s with new size %s'
                    % (v['Version'], v['SizeBytes'])
                )
                should_add = True
                break

        if should_add:
            versions[v['Version']] = v

            # report changed.
            return True, v['Version'], v['DownloadURL']
        else:
            return False, None, None
