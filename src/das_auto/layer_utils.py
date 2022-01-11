import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss


def pairwise_euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects

    sum_square = tf.math.reduce_sum(tf.math.square(x - y[:, tf.newaxis]),
                                    axis=(-2, -1), keepdims=True)[..., 0]
    euc_dist = tf.math.sqrt(tf.math.maximum(sum_square, 0))
    return euc_dist


def cosine_dist(vects):
    x1, x2 = vects
    return 1 - tf.keras.layers.Dot(axes=1)([x1, x2])


class ContrastiveLossCOLA(Loss):
    def __init__(self, margin=1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        """[summary]

        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]

        Returns:
            [type]: [description]
        """
        if y_true.shape[0] is not None:
            y_true = tf.linalg.tensor_diag(tf.ones((y_true.shape[0],)))

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(self.margin - (y_pred), 0))
        loss = tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({'margin': self.margin})
        return config
