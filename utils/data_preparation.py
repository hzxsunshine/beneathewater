import gdown
import pandas as pd
import subprocess

# Load data
def get_uieb_data():
    base_url = "https://drive.google.com/uc?id="
    id_ = "1Z5LKgmZmFRCQ6kZzHWTcV8gzeLcxHtXE"
    url = base_url + id_
    gdown.download(url)
    print("Downloaded data from Google Drive, start unzipping...")
    subprocess.run(["tar", "-xvf", "./uieb_dataset.tar.gz"])
    print("Unzipped data.")

def setup_colab_env():
    print("Installing dependencies...")
    subprocess.run(["pip", "install", "ftfy", "regex", "tqdm"])
    print("Installing CLIP...")
    subprocess.run(["pip", "install", "git+https://github.com/openai/CLIP.git"])
    print("Finished.")