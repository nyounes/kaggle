from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score


def load_train_datas(train_size, image_dim1, image_dim3, label_size, split):
    image_size = image_dim1 * image_dim1 * image_dim3
    t = time()
    with open("datas/train_x32.bin", 'rb') as f:
        x = np.fromfile(f, dtype=np.uint8, count=train_size * image_size)
    x = x.reshape([train_size, image_size])
    print("training images loaded in {:4.2f} seconds".format(time() - t))

    t = time()
    with open("datas/train_y32.bin", 'rb') as f:
        y = np.fromfile(f, dtype=np.uint8, count=train_size * label_size)
    y = y.reshape([train_size, 17])
    print("training labels loaded in {:4.2f} seconds".format(time() - t))
    print("\n")

    x = x / 255.0

    train_x, val_x, train_y, val_y = train_test_split(x, y, train_size=split)

    return train_x, val_x, train_y, val_y


def load_val_datas(val_size, image_dim1, image_dim3, label_size):
    image_size = image_dim1 * image_dim1 * image_dim3
    t = time()
    with open("datas/val_x32.bin", 'rb') as f:
        x = np.fromfile(f, dtype=np.uint8, count=val_size * image_size)
    x = x.reshape([val_size, image_size])
    print("validation images loaded in {:4.2f} seconds".format(time() - t))

    t = time()
    with open("datas/val_y32.bin", 'rb') as f:
        y = np.fromfile(f, dtype=np.uint8, count=val_size * label_size)
    y = y.reshape([val_size, 17])
    print("validation labels loaded in {:4.2f} seconds".format(time() - t))
    print("\n")

    x = x.reshape([val_size, image_dim1, image_dim1, image_dim3])
    x = x / 255.0

    return x, y


def load_test_datas(filename, test_size, image_dim1, image_dim3):
    image_size = image_dim1 * image_dim1 * image_dim3
    t = time()
    with open(filename, 'rb') as f:
        test_x = np.fromfile(f, dtype=np.uint8, count=test_size * image_size)
    test_x = test_x.reshape([test_size, image_size])
    print("testing images loaded in {:4.2f} seconds".format(time() - t))
    test_x = test_x / 255.0

    return test_x


def make_predictions(model, test_size, image_size, n_colors):
    size = test_size[0]
    size_add = test_size[1]

    test_x = load_test_datas("datas/test_x32.bin", size, image_size, n_colors)
    test_x = test_x.reshape([size, image_size, image_size, n_colors])
    predictions = model.predict(test_x, batch_size=128)

    test_x = load_test_datas("datas/test_x32-add.bin", size_add, image_size, n_colors)
    test_x = test_x.reshape([size_add, image_size, image_size, n_colors])
    predictions_add = model.predict(test_x, batch_size=128)

    final_predictions = np.concatenate([predictions, predictions_add], axis=0)

    return final_predictions


def make_submissions(predictions, tresholds, keras_filenames, output_filename):
    print("making submissions file")
    label_predictions = np.array(predictions) > tresholds
    mapper = dict(zip(keras_filenames, label_predictions))
    submissions = pd.read_csv("datas/sample_submission.csv")

    label_map = dict(pd.read_csv("datas/label_dict.csv").values)
    inv_label_map = {i: l for l, i in label_map.items()}

    for i in range(len(submissions)):
        current_filename = submissions["image_name"].values[i]
        current_filename = "test-jpg/" + current_filename + ".jpg"
        current_predictions = mapper[current_filename]
        s = ""
        index_positive = np.where(current_predictions == 1)[0]
        for j in index_positive:
            s += inv_label_map[j]
            s += " "
        submissions["tags"].set_value(i, s)
    print(submissions.head())
    submissions.to_csv(output_filename, index=False)
    print("submissions file saved")


def make_submissions_old(predictions, filename):
    label_predictions = np.array(predictions) > 0.2
    submissions = pd.read_csv("datas/sample_submission.csv")

    label_map = dict(pd.read_csv("datas/label_dict.csv").values)
    inv_label_map = {i: l for l, i in label_map.items()}

    for i in range(len(submissions)):
        s = ""
        index_positive = np.where(label_predictions[i] == 1)[0]
        for j in index_positive:
            s += inv_label_map[j]
            s += " "
        submissions["tags"].set_value(i, s)

    print(submissions.head())
    submissions.to_csv(filename, index=False)


def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.2]*17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)

    return x
