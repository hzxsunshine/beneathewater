import gdwon
import pandas as pd
from os import subprocess

# Load data
def get_uieb_data():
    base_url = "https://drive.google.com/uc?id="
    id_ = "1ylAgljMIMq8zIwTiU4w3SrUlGQ3sxs-U"
    url = base_url + id_
    gdown.download(url)
    subprocess.run(["tar", "-xvf", "./uieb.tar.gz"])

def setup_colab_env():
    subprocess.run(["pip", "install", "ftfy", "regex", "tqdm"])
    subprocess.run(["pip", "install", "git+https://github.com/openai/CLIP.git"])