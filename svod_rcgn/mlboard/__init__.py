import os

from svod_rcgn.tools.print import print_fun

try:
    from mlboardclient.api import client
except ImportError:
    client = None


mlboard = None

if client:
    mlboard = client.Client()
    try:
        mlboard.apps.get()
    except Exception:
        mlboard = None
        print_fun('Do not use mlboard.')
    else:
        print_fun('Using mlboard.')


def catalog_ref(name, ctype, version):
    return '#/{}/catalog/{}/{}/versions/{}'. \
        format(os.environ.get('WORKSPACE_NAME'), ctype, name, version)
