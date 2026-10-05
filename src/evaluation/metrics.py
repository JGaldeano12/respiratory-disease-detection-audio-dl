import numpy as np

def icbhi_official_score(labels, predictions, normal_class=0):
    """
    Compute the official ICBHI 2017 challenge metrics.

    Specificity is the proportion of normal cycles correctly classified as
    normal. Sensitivity is the proportion of abnormal cycles (crackle, wheeze,
    or both) assigned exactly to their true class. The ICBHI score is the
    average of both (Rocha et al., 2019).

    Args:
        labels (np.ndarray): True class labels.
        predictions (np.ndarray): Predicted class labels.
        normal_class (int, optional): Label of the normal (healthy) class.
            Defaults to 0.

    Returns:
        tuple[float, float, float]: Sensitivity, specificity, and ICBHI score.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    # Boolean mask of the normal cycles.
    normal = labels == normal_class

    # Specificity: normal cycles predicted as normal.
    specificity = float(np.mean(predictions[normal] == normal_class))

    # Sensitivity: abnormal cycles predicted as their exact class.
    sensitivity = float(np.mean(predictions[~normal] == labels[~normal]))

    return sensitivity, specificity, (sensitivity + specificity) / 2