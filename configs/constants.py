from typing import List, Final

path_to_data: Final[str] = "./uieb_dataset"
contrastive_pairs: Final[List[List[str]]] = [
    ["Accurate Color Representation", "Blue/Green Color Cast"], # "Color representation is accurate"
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