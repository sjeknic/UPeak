import tensorflow as tf
import tensorflow.keras.backend as K

def channel_precision(channel, name):
    def precision_func(y_true, y_pred):
        y_pred_tmp = K.cast(tf.equal( K.argmax(y_pred, axis=-1), channel), "float32")
        true_positives = K.sum(K.round(K.clip(y_true[:,:,channel] * y_pred_tmp, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_tmp, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())

        return precision
    precision_func.__name__ = name
    return precision_func


def channel_recall(channel, name):
    def recall_func(y_true, y_pred):
        y_pred_tmp = K.cast(tf.equal( K.argmax(y_pred, axis=-1), channel), "float32")
        true_positives = K.sum(K.round(K.clip(y_true[:,:,channel] * y_pred_tmp, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:,:,channel], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())

        return recall
    recall_func.__name__ = name
    return recall_func