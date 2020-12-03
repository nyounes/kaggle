import os
import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD
from keras import callbacks
from tools import make_submissions, optimise_f2_thresholds
from keras_tools import load_model, beta_score
from callbacks import CustomCallbacks
from data_generator import ImageDataGenerator
from model import cnn32, cnn64, cnn128
from sklearn.metrics import fbeta_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAIN_SIZE, VAL_SIZE, TEST_SIZE, TEST_SIZE_ADD = 30000, 10479, 40669, 20522
IMAGE_FIRST_DIM, N_COLORS = 64, 3
IMAGE_SIZE = IMAGE_FIRST_DIM * IMAGE_FIRST_DIM * N_COLORS
LABEL_SIZE = 17
DROPOUT = 0.25
BATCH_SIZE = 96
N_EPOCHS = 200
CHECKPOINTS_FOLDER = "checkpoints/"
EPOCH_TO_LOAD = 0 
MODEL_JSON = "epoch-" + str(EPOCH_TO_LOAD) + ".json"
MODEL_H5 = "epoch-" + str(EPOCH_TO_LOAD) + ".h5"
TO_FIT = True

df_train_labels = pd.read_csv("datas/train_labels.csv")
label_dict = df_train_labels.set_index("image_name").T.to_dict("list")

if EPOCH_TO_LOAD > 0:
    model, is_loaded = load_model(CHECKPOINTS_FOLDER + MODEL_JSON, CHECKPOINTS_FOLDER + MODEL_H5)
else:
    if IMAGE_FIRST_DIM == 64:
        model = cnn64(IMAGE_FIRST_DIM, N_COLORS)
    if IMAGE_FIRST_DIM == 128:
        model = cnn128(IMAGE_FIRST_DIM, N_COLORS)

starting_lr = 0.1
adam = Adam(lr=starting_lr)
sgd = SGD(lr=0.05, momentum=0.9, decay=0.0005)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[beta_score])

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory("datas/train", target_size=(IMAGE_FIRST_DIM, IMAGE_FIRST_DIM),
                                              batch_size=BATCH_SIZE,
                                              class_mode="multilabel", multilabel_classes=label_dict)

val_generator = datagen.flow_from_directory("datas/validation", target_size=(IMAGE_FIRST_DIM, IMAGE_FIRST_DIM),
                                              batch_size=1, shuffle=False,
                                              class_mode="multilabel", multilabel_classes=label_dict)

test_generator = datagen.flow_from_directory("datas/test", target_size=(IMAGE_FIRST_DIM, IMAGE_FIRST_DIM),
                                            batch_size=BATCH_SIZE, shuffle=False,
                                            class_mode=None)

if TO_FIT:
    my_callbacks = CustomCallbacks(starting_lr, EPOCH_TO_LOAD, val_generator)
    lr_callback = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                              verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    model.fit_generator(train_generator, steps_per_epoch=TRAIN_SIZE/BATCH_SIZE, epochs=N_EPOCHS,
                        validation_data=val_generator, validation_steps=VAL_SIZE/BATCH_SIZE,
			            verbose=0, callbacks=[my_callbacks, lr_callback], pickle_safe=True)


print("\n")
val_predictions = model.predict_generator(val_generator, steps=VAL_SIZE, pickle_safe=True)
tresholds_val = optimise_f2_thresholds(val_generator.classes, val_predictions, verbose=False)
constant_score = fbeta_score(val_generator.classes, np.array(val_predictions) > 0.2,
                             beta=2, average="samples")
optimized_score = fbeta_score(val_generator.classes, np.array(val_predictions) > tresholds_val,
                              beta=2, average="samples")
print("validation score with constant treshold: {:4f}".format(constant_score))
print("validation score with optimized treshold: {:4f}".format(optimized_score))
tresholds = tresholds_val.copy()

print("calculating test predictions...")
test_predictions = model.predict_generator(test_generator, steps=(TEST_SIZE + TEST_SIZE_ADD) / BATCH_SIZE)
print("predictions done")
make_submissions(test_predictions, tresholds_val, test_generator.filenames, "submissions_opti.csv")

print(model.summary())

print("\n")
print("the end")
