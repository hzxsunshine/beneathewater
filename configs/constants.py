from typing import List, Final

path_to_data: Final[str] = "./uieb_dataset"

# references: https://www.nature.com/articles/s41598-023-44179-3
contrastive_pairs: Final[List[List[str]]] = [
    ["Accurate Color Representation", "Blue/Green Color Cast"],
    ["Vibrant and vivid", "Dull and washed-out"],
    ["Richly detailed", " Blurry and indistinct"],
    ["Perfectly lit", "Poorly exposed"],
    ["Crystal-clear", "Obscured and murky"],
    ["Soft Focus", "Sharp Detail"],
    ["Natural Ambient Light", "Artificial Light"], 
]

def select_contrastive_pairs(indices=None):
    if indices is None:
        return contrastive_pairs
    else:
        selected_pairs = []
        for i in indices:
            if i < len(contrastive_pairs):
                selected_pairs.append(contrastive_pairs[i])
    return selected_pairs

if __name__ == "__main__":
    print(select_contrastive_pairs())
    print(contrastive_pairs)
