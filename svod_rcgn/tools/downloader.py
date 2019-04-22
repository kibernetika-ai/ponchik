import io
import tarfile
from urllib.request import urlopen
from svod_rcgn.tools.print import print_fun


def downloader_args(args):
    return Downloader(args.download, args.input_dir)


class Downloader:
    def __init__(self, url, destination="."):
        self.url = url
        self.dest = destination

    def extract(self):
        try:
            print_fun("Downloading from %s" % self.url)
            arc = io.BytesIO(urlopen(self.url).read())
        except Exception as e:
            return RuntimeError("Download archive error: %s" % e)
        try:
            print_fun("Extracting...")
            tar = tarfile.open(fileobj=arc, mode="r:*")
            tar.extractall(path=self.dest)
            tar.close()
        except Exception as e:
            return RuntimeError("Extract archive error: %s" % e)
        return None
