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
import zipfile

VER_FILE = "__version"
LOG = logging.getLogger(__name__)


def version_request(model_url, token):
    resp = requests.get(model_url, headers={'Authorization': 'Bearer %s' % token})
    return resp


def check(model_url, token, current_version_updated=None):
    try:
        resp = version_request(model_url, token)
    except Exception as e:
        LOG.info('Error: %s' % e)
        return False, current_version_updated

    if resp.status_code >= 400:
        LOG.info('Response status %s.' % resp.status_code)
        LOG.info(resp.text)
        return False, current_version_updated

    try:
        vs = resp.json()
        LOG.info("Event Meta: {}".format(vs))
    except (ValueError, TypeError) as e:
        LOG.info('not json: %s' % e)
        return False, current_version_updated

    read_current_version = vs.get("changed", None)

    if read_current_version == current_version_updated:
        return False, read_current_version

    if current_version_updated is None:
        return True, read_current_version

    if read_current_version is None:
        return False, current_version_updated

    f = '%Y-%m-%d %H:%M:%S'
    try:
        prev = datetime.datetime.strptime(current_version_updated, f)
        read = datetime.datetime.strptime(read_current_version, f)
        if prev < read:
            return True, read_current_version
    except Exception as e:
        LOG.info('Error getting update dates: %s' % e)
    return False, current_version_updated


def load_model(model_url, token, dst):
    download_url = os.path.join(model_url, 'export')
    request = requests.get(download_url, headers={'Authorization': 'Bearer %s' % token})
    model_zip = zipfile.ZipFile(io.BytesIO(request.content))
    shutil.rmtree(dst, ignore_errors=True)
    os.mkdir(dst)
    model_zip.extractall(dst)


def loop(pattern, model_url, token, classifiers_dir, openvino_facenet=None):
    cron = croniter.croniter(pattern, datetime.datetime.utcnow())
    next_time = cron.get_next(datetime.datetime)
    curver_updated = None
    current_version_file = os.path.join(classifiers_dir, VER_FILE)
    if os.path.isfile(current_version_file):
        with open(current_version_file, 'r') as r:
            try:
                curver = json.load(r)
                curver_updated = curver.get('updated', None)
            except Exception as e:
                LOG.error('load current version data error: %s' % e)
                pass

    while True:
        current = datetime.datetime.utcnow()
        if current >= next_time:
            LOG.info('Check for new version... [model=%s]' % (model_url))
            changed, updated = check(model_url, token, curver_updated)
            if changed:
                try:
                    load_model(model_url, token, classifiers_dir)
                    curver_updated = updated
                    ver_file = os.path.join(classifiers_dir, VER_FILE)
                    with open(ver_file, 'w') as v:
                        json.dump(dict(
                            updated=curver_updated
                        ), v)
                    LOG.info('Reloading classifiers...')
                    if openvino_facenet is not None:
                        openvino_facenet.load_classifiers()
                except Exception as e:
                    LOG.error('load current version data error: %s' % e)
                    pass

            next_time = cron.get_next(datetime.datetime)

        time.sleep(1)
