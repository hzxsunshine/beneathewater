import gdown
import pandas as pd
from os import subprocess

# Load data
def get_uieb_data():
    base_url = "https://drive.google.com/uc?id="
    id_ = "1Z5LKgmZmFRCQ6kZzHWTcV8gzeLcxHtXE"
    url = base_url + id_
    gdown.download(url)
    subprocess.run(["tar", "-xvf", "./uieb_dataset.tar.gz"])

def setup_colab_env():
    subprocess.run(["pip", "install", "ftfy", "regex", "tqdm"])
    subprocess.run(["pip", "install", "git+https://github.com/openai/CLIP.git"])