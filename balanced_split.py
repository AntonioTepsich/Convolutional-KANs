from torch.utils.data import Subset
from collections import Counter
def train_val_split_balanced(train_dataset,proportion = 0.8):
    unique_classes = np.unique(train_dataset.targets.numpy())
    class_counts = Counter(train_dataset.targets.numpy())
    min_class_size = min(class_counts.values())

    balanced_indices = []
    for cls in unique_classes:
        indices_cls = np.where(train_dataset.targets.numpy() == cls)[0]
        np.random.shuffle(indices_cls)
        balanced_indices.extend(indices_cls[:min_class_size])
    np.random.shuffle(balanced_indices)
    train_size = int(proportion* len(balanced_indices))
    train_indices = balanced_indices[:train_size]
    val_indices = balanced_indices[train_size:]
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    return train_subset,val_subset