import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def process(input_dfs, contrastive_pairs, model=model, preprocess=preprocess):
    processed_dfs = input_dfs.copy()
    flatten_pairs = np.ravel(contrastive_pairs)
    processed_dfs[flatten_pairs] = 0.0
    cols = input_dfs.columns.tolist()
    paths = [col for col in cols if col.endswith("_path")]
    print(paths)
    if len(paths) > 1:
        print("Multiple paths found. Please check your dataframe. This process will only take the first path.")
    
    for index, row in tqdm(input_dfs.iterrows(), total=input_dfs.shape[0]):
        image = preprocess(Image.open(row[paths[0]])).unsqueeze(0).to(device)
        text = clip.tokenize(flatten_pairs).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            logits_per_image = logits_per_image.view(len(contrastive_pairs), 2)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            processed_dfs.loc[index, flatten_pairs] = probs.flatten()

    return processed_dfs

            
