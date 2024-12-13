import os
import zipfile
import urllib.request

from lib import *

data_dir = "./data"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

target_path = os.path.join(data_dir, "hymenoptera_data.zip")

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)

    # read file tar
    zip = zipfile.ZipFile(target_path)
    zip.extractall(data_dir)
    zip.close()


    os.remove(target_path)


