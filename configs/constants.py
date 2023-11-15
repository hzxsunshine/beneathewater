contrastive_pairs = [
    ["Accurate Color Representation", "Blue/Green Color Cast"], # "Color representation is accurate"
]

def select_contrastive_pairs(indices=None):
    if indices is None:
        return contrastive_pairs
    else:
        contrastive_pairs = []
        for i in indices:
            contrastive_pairs.append(contrastive_pairs[i])
    return contrastive_pairs