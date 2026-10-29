"""Task 6: accuracy metrics computed from saved predictions (plain numpy, no GPU needed).

Shared by src/evaluate.py (prints the summary on CARC) and src/plot_evaluation.py (report figures).
Convention: confusion matrix rows are the true class, columns the predicted class.
"""
import numpy as np


def confusion_matrix(labels, preds, num_classes):
    """cm[t, p] = number of images of true class t predicted as class p."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


def topk_accuracy(labels, topk, k):
    """Fraction of images whose true class is among the first k ranked predictions."""
    return float((topk[:, :k] == labels[:, None]).any(axis=1).mean())


def per_class_accuracy(cm):
    """Top-1 accuracy of each true class (recall); NaN for a class with no images."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.diag(cm) / cm.sum(axis=1)


def top_confusions(cm, k=20):
    """The k largest off-diagonal cells as (true, predicted, count, share of the true class), largest share first."""
    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)
    np.fill_diagonal(share, 0)
    flat = np.argsort(-share, axis=None, kind="stable")[:k]
    pairs = []
    for t, p in zip(*np.unravel_index(flat, cm.shape)):
        if share[t, p] == 0:  # no confusions left (diagonal cells were zeroed above)
            break
        pairs.append((int(t), int(p), int(cm[t, p]), float(share[t, p])))
    return pairs


def cluster_order(cm):
    """A class order that puts classes the model confuses with each other next to each other.

    Hierarchical clustering where the similarity of two classes is how often each is mistaken for
    the other (as a share of its images, averaged both ways). Sorting the confusion matrix this way
    turns groups of look-alike classes into visible blocks along the diagonal.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)
    similarity = (share + share.T) / 2
    np.fill_diagonal(similarity, 0)
    distance = 1 - similarity / max(similarity.max(), 1e-12)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average", optimal_ordering=True)
    return leaves_list(tree)
