'''
We define some custom Keras metrics here that are specific to binary classification.
'''

from keras import backend as K


def precision(y_true, y_pred):
    '''
    Custom Keras metric that measures the precision of a binary classifier.
    '''
    # Assuming index 1 is positive.
    pred_indices = K.argmax(y_pred, axis=-1)
    true_indices = K.argmax(y_true, axis=-1)
    num_true_positives = K.sum(pred_indices * true_indices)
    num_positive_predictions = K.sum(pred_indices)
    return K.cast(num_true_positives / num_positive_predictions, K.floatx())


def recall(y_true, y_pred):
    '''
    Custom Keras metric that measures the recall of a binary classifier.
    '''
    # Assuming index 1 is positive.
    pred_indices = K.argmax(y_pred, axis=-1)
    true_indices = K.argmax(y_true, axis=-1)
    num_true_positives = K.sum(pred_indices * true_indices)
    num_positive_truths = K.sum(true_indices)
    return K.cast(num_true_positives / num_positive_truths, K.floatx())


def f1_score(y_true, y_pred):
    '''
    Custom Keras metric that measures F1 score of a binary classifier.
    '''
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return K.cast(2 * prec * rec / (prec + rec), K.floatx())
