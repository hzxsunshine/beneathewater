import os
import subprocess

import gdown
import pandas as pd

# Load data
def get_uieb_data():
    base_url = "https://drive.google.com/uc?id="
    id_ = "1Z5LKgmZmFRCQ6kZzHWTcV8gzeLcxHtXE"
    url = base_url + id_
    gdown.download(url)
    print("Downloaded data from Google Drive, start unzipping...")
    result = subprocess.run(["tar", "-xvf", "./uieb_dataset.tar.gz"], check=True)
    if result.returncode != 0:
        print("Failed to unzip data. Please try again.")
        return
    print("Unzipped data.")

def setup_colab_env():
    print("Installing dependencies...")
    result = subprocess.run(["pip", "install", "ftfy", "regex", "tqdm"], check=True)
    if result.returncode != 0:
        print("Failed to install dependencies. Please try again.")
        return
    print("Installing CLIP...")
    result = subprocess.run(["pip", "install", "git+https://github.com/openai/CLIP.git"], check=True)
    if result.returncode != 0:
        print("Failed to install CLIP. Please try again.")
        return
    print("Finished.")

def prepare_the_dataframe(path_to_data):
    '''
    Note: this will only be run once.
    '''
    pair = ["raw-890", "reference-890"]

    files_in_first_dir = set(file for file in os.listdir(os.path.join(path_to_data, pair[0])) if file.endswith('.png'))
    files_in_second_dir = set(file for file in os.listdir(os.path.join(path_to_data, pair[1])) if file.endswith('.png'))
    print("Number of files in the first directory: ", len(files_in_first_dir))
    print("Number of files in the second directory: ", len(files_in_second_dir))
    common_files = files_in_first_dir & files_in_second_dir
    print("Number of common files: ", len(common_files))
    print("Example of common files: ", list(common_files)[:5])

    # Create a dataframe
    df = pd.DataFrame(columns=["index", "raw", "reference"])
    df["index"] = list(range(len(common_files)))
    df["raw"] = [os.path.join(path_to_data, pair[0], file) for file in common_files]
    df["reference"] = [os.path.join(path_to_data, pair[1], file) for file in common_files]
    df.to_csv(os.path.join(path_to_data, "uieb.csv"), index=False)

    return df

if __name__ == "__main__":
    prepare_the_dataframe("YOUR_PATH_TO_DATA")