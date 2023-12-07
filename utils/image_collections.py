import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

to_show_images = ["ScubaDiver", 4992, 3090, 3151, 4008, 3456] #1311， 1395, 1636, 3645, 1211, 2902, 3198, 3222, 3331, 3357, 3937,
img_path = "./notebooks/outputs/"
kinds = {
    "raw": [],
    "base": [],
    "vivid_mid": [],
    "color_cast": [],
    "exposure": [],
    "all0_last": [],
}

def load_image(path, target_height=360, aspect_ratio=13/9):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    current_aspect_ratio = w / h

    # Check if the image needs cropping
    if current_aspect_ratio != aspect_ratio:
        # Central cropping to 16:9
        if current_aspect_ratio > aspect_ratio:
            # Image is wider than 16:9, crop the width
            new_width = int(h * aspect_ratio)
            startx = w // 2 - new_width // 2
            img = img[:, startx:startx+new_width, :]
        else:
            # Image is taller than 16:9, crop the height
            new_height = int(w / aspect_ratio)
            starty = h // 2 - new_height // 2
            img = img[starty:starty+new_height, :, :]

    # Resize image to target dimensions
    target_width = int(target_height * aspect_ratio)
    img = cv2.resize(img, (target_width, target_height))

    return img

for kind in kinds.keys():
    for img_name in to_show_images:
        img = load_image(img_path + f"{kind}_{img_name}.jpg")
        kinds[kind].append(img)


def show_image_grid_horizontal(imgs, kinds, figsize=(30, 20)):
    rows = len(kinds)
    cols = len(imgs['raw'])  # Assuming all kinds have the same number of images

    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    for i, kind in enumerate(kinds.keys()):
        for j in range(cols):
            img = imgs[kind][j]
            axs[i, j].imshow(img)
            axs[i, j].axis("off")

            # Add kind name to the left of the first column
            if j == 0:
                # rotate the text by 90 degrees
                axs[i, j].text(-50, img.shape[0] // 2, kind, rotation=90,
                               size=10, ha="center", va="center",
                               bbox=dict(boxstyle="round", alpha=0.5,
                                         ec=(1., 0.5, 0.5),
                                         fc=(1., 0.8, 0.8),
                                         )
                               )


    plt.tight_layout()
    plt.savefig("./compare.png")
    plt.show()


def show_image_grid(imgs, titles, rows, cols, figsize=(24, 17)):
    fig, ax = plt.subplots(rows, cols, figsize=figsize)

    for i in range(rows):
        for j in range(cols):
            ax[i, j].imshow(imgs[j][i])
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(titles[j], size=20, ha="center", va="center")

    plt.tight_layout()
    plt.savefig("./compare.png")
    plt.show()

if __name__ == "__main__":
    # Preparing the images and titles for the grid
    grid_images = [kinds[kind] for kind in kinds]
    grid_titles = ["Raw", "Base", "Color Enhanced", "White Balance Enhanced", "Exposure Enhanced", "All Enhanced"]

    # Define the number of rows and columns
    num_rows = len(to_show_images)
    num_cols = len(kinds)

    # Show the image grid
    show_image_grid(grid_images, grid_titles, num_rows, num_cols)
