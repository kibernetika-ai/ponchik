import datetime
import io
import json
import logging
import os
import shutil
import tarfile
import time

import croniter
import requests

VER_FILE = "__version"
LOG = logging.getLogger(__name__)


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
    max_len = 33
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if chunk:
            read += content.write(chunk)

            progress = '=' * int(float(read) / length * max_len - 1)
            progress += '>'
            progress += ' ' * (max_len - len(progress))
            LOG.info('Downloading... [%s]' % progress)

    # return cursor to 0
    content.seek(0)
    return content


def loop(pattern, base_url, ws, name, token, classifiers_dir, openvino_facenet):
    cron = croniter.croniter(pattern, datetime.datetime.utcnow())
    next_time = cron.get_next(datetime.datetime)

    curver_id = None
    curver_updated = None
    current_version_file = os.path.join(classifiers_dir, VER_FILE)
    if os.path.isfile(current_version_file):
        with open(current_version_file, 'r') as r:
            try:
                curver = json.load(r)
                curver_id = curver['version_id']
                curver_updated = curver['updated']
            except Exception as e:
                LOG.error('load current version data error: %s' % e)
                pass

    while True:
        current = datetime.datetime.utcnow()
        if current >= next_time:
            LOG.info('Check for new version... [model=%s/%s]' % (ws, name))
            changed, version, updated, url = check(
                base_url, ws, name, token, curver_id, curver_updated)
            if changed:
                # Download
                version_id = get_version_id(ws, name, version)
                LOG.info('Downloading new version %s...' % version_id)
                fileobj = download_version(url, token)
                curver_id, curver_updated = \
                    pull_version(version_id, updated, fileobj, classifiers_dir, openvino_facenet)
            next_time = cron.get_next(datetime.datetime)

        time.sleep(1)


def check(base_url, ws, name, token, current_version_id=None, current_version_updated=None):
    try:
        resp = version_request(base_url, ws, name, token)
    except Exception as e:
        LOG.info('Error: %s' % e)
        return None, None, None, None

    if resp.status_code >= 400:
        LOG.info('Response status %s.' % resp.status_code)
        LOG.info(resp.text)
        return None, None, None, None

    try:
        vs = resp.json()
    except (ValueError, TypeError) as e:
        LOG.info('not json: %s' % e)
        return None, None, None, None

    versions = []
    read_current_version = None

    for v in vs:
        if v['Status'] != 'ok':
            continue
        if get_version_id(ws, name, v['Version']) == current_version_id:
            read_current_version = v
            break
        versions.append(v)

    upd = None
    if len(versions) > 0:
        upd = versions[0]
    elif read_current_version is not None and current_version_updated is not None:
        f = '%Y-%m-%dT%H:%M:%SZ'
        try:
            prev = datetime.datetime.strptime(current_version_updated, f)
            read = datetime.datetime.strptime(read_current_version['Updated'], f)
            if prev < read:
                upd = read_current_version
        except Exception as e:
            LOG.info('Error getting update dates: %s' % e)

    if upd is not None:
        return True, upd['Version'], upd['Updated'], upd['DownloadURL']

    return False, None, None, None


def get_version_id(ws, name, version):
    return "{}/{}:{}".format(ws, name, version)


def pull_version(version_id, updated, fileobj, classifiers_dir, openvino_facenet):

    tar = tarfile.open(fileobj=fileobj)
    LOG.info('Extracting new version %s to %s...' % (version_id, classifiers_dir))

    shutil.rmtree(classifiers_dir, ignore_errors=True)
    os.mkdir(classifiers_dir)
    tar.extractall(classifiers_dir)

    LOG.info('Reloading classifiers...')
    if openvino_facenet is not None:
        openvino_facenet.load_classifiers()
    else:
        LOG.info('Classifiers not detected, skipped...')

    ver_file = os.path.join(classifiers_dir, VER_FILE)
    with open(ver_file, 'w') as v:
        json.dump(dict(
            version_id=version_id,
            updated=updated,
        ), v)

    return version_id, updated
