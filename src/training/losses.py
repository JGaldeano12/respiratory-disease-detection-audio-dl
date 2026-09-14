import tensorflow as tf

def focal_loss(gamma=2.0, alpha=None):
    """
    Create a focal loss function for multi-class classification.

    The focal loss reduces the contribution of well-classified samples
    and focuses training on misclassified samples. An optional class
    weighting factor can be applied through the alpha parameter.

    Args:
        gamma (float): Focusing parameter that controls the rate at which
            the contribution of well-classified samples is reduced.
        alpha (list or tuple, optional): Class-specific weighting factors
            used to adjust the contribution of each class to the loss.

    Returns:
        callable: A focal loss function that computes the loss between
            the true and predicted class probabilities.
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - pt, gamma)

        if alpha is not None:
            alpha_t = tf.reduce_sum(
                y_true * tf.constant(alpha, dtype=tf.float32),
                axis=-1, keepdims=True
            )
            focal_weight = alpha_t * focal_weight

        return tf.reduce_sum(focal_weight * ce, axis=-1)

    return loss