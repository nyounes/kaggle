from keras.models import model_from_json
from keras.layers import Conv2D, Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
import keras.backend as K


def conv2d(previous_layer, filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu",
           batch_norm=True, before_activation=True):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(previous_layer)
    if batch_norm is True and before_activation is True:
        x = BatchNormalization()(x)
    if activation == "prelu":
        x = PReLU()(x)
    else:
        x = Activation(activation)(x)
    if batch_norm is True and before_activation is False:
        x = BatchNormalization()(x)
    return x


def dense(previous_layer, units, activation="relu", batch_norm=True, before_activation=True):
    x = Dense(units)(previous_layer)
    if batch_norm is True and before_activation is True:
        x = BatchNormalization()(x)
    if activation == "prelu":
        x = PReLU()(x)
    else:
        x = Activation(activation)(x)
    if batch_norm is True and before_activation is False:
        x = BatchNormalization()(x)
    return x


def save_model(model, json_filename, weights_filename):
    model_json = model.to_json()
    with open(json_filename, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_filename)
    # print("Saved model to disk")


def load_model(json_filename, weights_filename):
    try:
        json_file = open(json_filename, 'r')
    except:
        json_file = None
    if json_file is not None:
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_filename)
        print("Loaded model from disk")
        return model, True
    else:
        return None, False


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def beta_score(y_true, y_pred):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''

    treshold = K.constant(0.2)
    greater_mask = K.greater(y_pred, treshold)
    y_pred_final = K.cast(greater_mask, "float32")

    beta = 2
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred_final)
    r = recall(y_true, y_pred_final)
    bb = beta ** 2
    # fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r)
    return fbeta_score
