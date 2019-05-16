import os

from svod_rcgn.tools import print_fun

try:
    from mlboardclient.api import client
except ImportError:
    client = None


mlboard = None
mlboard_logging = False

if client:
    mlboard = client.Client()
    mlboard_logging = True
    try:
        mlboard.apps.get()
    except Exception:
        mlboard_logging = False
        print_fun('Do not use mlboard parameters logging.')
    else:
        print_fun('Using mlboard parameters logging.')


def update_task_info(data):
    if mlboard and mlboard_logging:
        mlboard.update_task_info(data)


def catalog_ref(name, ctype, version):
    return '#/{}/catalog/{}/{}/versions/{}'. \
        format(os.environ.get('WORKSPACE_NAME'), ctype, name, version)
