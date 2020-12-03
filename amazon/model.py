from keras.layers import Input, MaxPooling2D, AveragePooling2D, Dropout, Flatten, concatenate
from keras.models import Model
from keras_tools import conv2d, dense

def cnn32(image_size, n_colors):
    inputs = Input(shape=(image_size, image_size, n_colors))
    first_model = conv2d(inputs, 16, batch_norm=True)
    first_model = conv2d(first_model, 16, batch_norm=True)
    first_model = conv2d(first_model, 16, batch_norm=True)
    first_model = MaxPooling2D(pool_size=(2, 2))(first_model)

    first_model = conv2d(first_model, 32, batch_norm=True)
    first_model = conv2d(first_model, 32, batch_norm=True)
    first_model = conv2d(first_model, 32, batch_norm=True)
    first_model = MaxPooling2D(pool_size=(2, 2))(first_model)

    first_model = conv2d(first_model, 64, batch_norm=True)
    first_model = conv2d(first_model, 64, batch_norm=True)
    first_model = conv2d(first_model, 64, batch_norm=True)
    first_model = MaxPooling2D(pool_size=(2, 2))(first_model)

    first_model = conv2d(first_model, 128, batch_norm=True)
    first_model = conv2d(first_model, 128, batch_norm=True)
    first_model = conv2d(first_model, 128, batch_norm=True)
    first_model = MaxPooling2D(pool_size=(2, 2))(first_model)
    first_model = Dropout(0.25)(first_model)

    first_model = Flatten()(first_model)
    first_model = dense(first_model, 256, activation="relu", batch_norm=True)
    first_model = dense(first_model, 128, activation="relu", batch_norm=True)
    first_model = Dropout(0.5)(first_model)
    first_model = dense(first_model, 17, activation="sigmoid", batch_norm=False)

    model = Model(inputs=inputs, outputs=first_model)

    return model


def cnn64(image_size, n_colors):
    inputs = Input(shape=(image_size, image_size, n_colors))
    layer1 = conv2d(inputs, 32, activation="prelu", batch_norm=True)
    layer1 = conv2d(layer1, 32, activation="prelu", batch_norm=True)
    layer1 = MaxPooling2D(pool_size=(2, 2))(layer1)

    layer2 = conv2d(layer1, 64, activation="prelu", batch_norm=True)
    layer2 = conv2d(layer2, 64, activation="prelu", batch_norm=True)
    layer2 = MaxPooling2D(pool_size=(2, 2))(layer2)

    layer3 = conv2d(layer2, 128, activation="prelu", batch_norm=True)
    layer3 = conv2d(layer3, 128, activation="prelu", batch_norm=True)
    layer3 = MaxPooling2D(pool_size=(2, 2))(layer3)

    layer4 = conv2d(layer3, 256, activation="prelu", batch_norm=True)
    layer4 = conv2d(layer4, 256, activation="prelu", batch_norm=True)
    layer4 = MaxPooling2D(pool_size=(2, 2))(layer4)
    layer4 = Dropout(0.25)(layer4)

    final_model = Flatten()(layer4)
    final_model = dense(final_model, 512, activation="relu", batch_norm=True)
    final_model = Dropout(0.5)(final_model)
    final_model = dense(final_model, 17, activation="sigmoid", batch_norm=False)

    model = Model(inputs=inputs, outputs=final_model)

    return model


def cnn64_kaggle(image_size, n_colors):
    inputs = Input(shape=(image_size, image_size, n_colors))
    layer1 = conv2d(inputs, 16, activation="relu", batch_norm=True)
    layer1 = conv2d(layer1, 16, activation="relu", batch_norm=True)
    layer1 = MaxPooling2D(pool_size=(2, 2))(layer1)

    layer2 = conv2d(layer1, 32, activation="relu", batch_norm=True)
    layer2 = conv2d(layer2, 32, activation="relu", batch_norm=True)
    layer2 = MaxPooling2D(pool_size=(2, 2))(layer2)

    layer3 = conv2d(layer2, 64, activation="relu", batch_norm=True)
    layer3 = conv2d(layer3, 64, activation="relu", batch_norm=True)
    layer3 = MaxPooling2D(pool_size=(2, 2))(layer3)

    layer4 = conv2d(layer3, 128, activation="relu", batch_norm=True)
    layer4 = conv2d(layer4, 128, activation="relu", batch_norm=True)
    layer4 = MaxPooling2D(pool_size=(2, 2))(layer4)
    layer4 = Dropout(0.25)(layer4)

    layer5 = conv2d(layer4, 256, activation="relu", batch_norm=True)
    layer5 = conv2d(layer5, 256, activation="relu", batch_norm=True)
    layer5 = MaxPooling2D(pool_size=(2, 2))(layer5)

    model1 = AveragePooling2D(padding="same")(layer4)
    model1 = Flatten()(model1)
    model2 = AveragePooling2D(padding="same")(layer5)
    model2 = Flatten()(model2)
    final_model = concatenate([model1, model2])

    final_model = dense(final_model, 512, activation="relu", batch_norm=True)
    final_model = Dropout(0.5)(final_model)
    final_model = dense(final_model, 17, activation="sigmoid", batch_norm=False)

    model = Model(inputs=inputs, outputs=final_model)

    return model


def cnn128(image_size, n_colors):
    inputs = Input(shape=(image_size, image_size, n_colors))
    layer1 = conv2d(inputs, 16, batch_norm=True)
    layer1 = conv2d(layer1, 16, batch_norm=True)
    layer1 = conv2d(layer1, 16, batch_norm=True)
    layer1 = MaxPooling2D(pool_size=(2, 2))(layer1)

    layer2 = conv2d(layer1, 32, batch_norm=True)
    layer2 = conv2d(layer2, 32, batch_norm=True)
    layer2 = conv2d(layer2, 32, batch_norm=True)
    layer2 = MaxPooling2D(pool_size=(2, 2))(layer2)

    layer3 = conv2d(layer2, 64, batch_norm=True)
    layer3 = conv2d(layer3, 64, batch_norm=True)
    layer3 = conv2d(layer3, 64, batch_norm=True)
    layer3 = MaxPooling2D(pool_size=(2, 2))(layer3)

    layer4 = conv2d(layer3, 128, batch_norm=True)
    layer4 = conv2d(layer4, 128, batch_norm=True)
    layer4 = conv2d(layer4, 128, batch_norm=True)
    layer4 = MaxPooling2D(pool_size=(2, 2))(layer4)

    layer5 = conv2d(layer4, 256, batch_norm=True)
    layer5 = conv2d(layer5, 256, batch_norm=True)
    layer5 = conv2d(layer5, 256, batch_norm=True)
    layer5 = MaxPooling2D(pool_size=(2, 2))(layer5)
    layer5 = Dropout(0.25)(layer5)

    final_model = Flatten()(layer4)
    final_model = dense(final_model, 512, activation="relu", batch_norm=True)
    final_model = Dropout(0.5)(final_model)
    final_model = dense(final_model, 17, activation="sigmoid", batch_norm=False)

    model = Model(inputs=inputs, outputs=final_model)

    return model
