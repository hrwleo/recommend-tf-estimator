#-*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np


def focal_loss(pred, y, alpha=0.25, gamma=2):
    zeros = tf.zeros_like(pred, dtype=pred.dtype)
    pos_p_sub = tf.where(y > zeros, y - pred, zeros)
    neg_p_sub = tf.where(y > zeros, zeros, pred)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent)


def cross_entropy_loss(labels, preds):
    """calculate cross_entropy_loss

      loss = -labels*log(preds)-(1-labels)*log(1-preds)

      Args:
        labels, preds

      Returns:
         log loss
    """

    if len(labels) != len(preds):
        raise ValueError(
            "labels num should equal to the preds num,")

    z = np.array(labels)
    x = np.array(preds)
    res = -z * np.log(x) - (1 - z) * np.log(1 - x)
    return res.tolist()