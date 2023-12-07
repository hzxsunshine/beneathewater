from typing import List, Final

path_to_data: Final[str] = "./uieb_dataset"
path_to_lsui: Final[str] = "./backup"

# references: https://www.nature.com/articles/s41598-023-44179-3
contrastive_pairs: Final[List[List[str]]] = [
    # White Balance: DONE
    ["Accurate Color Representation", "Blue/Green Color Cast"],
    # ["Accurate and Natural Color Representation", "Inaccurate, Artifact and uncomfortable Color Cast"], # WARM COLOR CAST result
    # ["Harmonious, Suitable, and Aesthetically Pleasing Color Representation", 
    #  "Distorted, Unfitting, and Visually Displeasing Color Cast"],

    # vibrant and vivid: DONE
    ["Vibrant and vivid", "Dull and washed-out"],
    ["Bright, Colorful Underwater Scene", "Dim, Desaturated Underwater Colors"],

    # Exposure and lit: DONE
    ["Crystal-clear and Unobstructed Scene", "Murky and Obscured View"],
    ["Balanced and Well-Lit view", "Underlit scene and Backlit Objects"],

    # Sharpness
    ["Richly detailed", " Blurry and indistinct"],
    ["Sharp Aquatic Details", "Blurred Movement"],

    # Lighting
    ["Natural Ambient Light", "Artificial Light"],

    # playground:
    # ["Studio Ghibli Style", "Realistic Underwater image"],
]

if __name__ == "__main__":
    print(contrastive_pairs)
