from time import time
import numpy as np
from sklearn.metrics import fbeta_score
import keras.backend as k
from keras.callbacks import Callback
from keras_tools import save_model


class CustomCallbacks(Callback):
    def __init__(self, starting_lr, current_epoch, val_generator):
        super(Callback, self).__init__()
        self.start_time = 0
        self.end_time = 0
        self.current_epoch = current_epoch
        self.current_batch = 0
        self.val_generator = val_generator
        self.learning_rate = starting_lr
        self.n_samples = 0

    def on_train_begin(self, logs=None):
        self.n_samples = self.params["steps"] * 96
        print("\n")
        print("Training on {0} samples during {1} epochs".format(int(self.n_samples), self.params["epochs"]))
        self.start_time = time()

    def on_batch_begin(self, batch, logs=None):
        modulo = int(self.params["steps"] / 20)
        self.end_time = time()
        batch_time = self.end_time - self.start_time
        if self.current_batch % modulo == 0:
            print("Epoch: {0}/{1}, Batch: {2}/{3}, time: {4}".format(
                self.current_epoch, self.params["epochs"], self.current_batch, int(self.params["steps"]), batch_time),
                 end="\r", flush=True)
        self.current_batch += 1


    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time()
        self.current_epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        self.end_time = time()
        total_time = self.end_time - self.start_time
        current_loss = logs.get("loss")
        current_val_loss = logs.get("val_loss")
        lr_value = k.get_value(self.model.optimizer.lr)

        predictions = self.model.predict_generator(self.val_generator, steps=self.val_generator.samples,
                                                   pickle_safe=True)
        y = self.val_generator.classes
        optimized_predictions = np.array(predictions) > 0.2
        beta_score = fbeta_score(y, optimized_predictions, beta=2, average="samples")

        print("Epoch: {:1} -- time: {:4.2f}s -- loss: {:.4f} -- beta score: {:.4f} --(lr: {:.4f} -- val_loss: {:.4f})".format(
            self.current_epoch, total_time, current_loss, beta_score, lr_value, current_val_loss))

        """
        if self.current_epoch >= 5:
            self.learning_rate = 0.005
        if self.current_epoch >= 10:
            self.learning_rate = 0.001
        if self.current_epoch >= 14:
            self.learning_rate = 0.0005
        if self.current_epoch >= 18:
            self.learning_rate = 0.0001
        if self .current_epoch >= 22:
            self.learning_rate = 0.00005

        k.set_value(self.model.optimizer.lr, self.learning_rate)
        """

        json_filename = "checkpoints/epoch-" + str(self.current_epoch) + ".json"
        weights_filename = "checkpoints/epoch-" + str(self.current_epoch) + ".h5"
        save_model(self.model, json_filename, weights_filename)
        self.current_batch = 0
