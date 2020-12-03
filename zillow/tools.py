import numpy as np
import pandas as pd
from time import time
import gc
from feature_eng import test_feature_engineering


def make_single_predictions(clf, train_columns, date_to_predict, filename):
    df_test = pd.read_csv("datas/" + filename)
    df_test = test_feature_engineering(df_test, train_columns, date_to_predict)
    # d_test = xgb.DMatrix(test_x)
    predictions = clf.predict(df_test)
    return predictions


def make_all_predictions(clf, train_columns, date_to_predict, filename_list):
    start_time = time()
    print("predicting", end="...", flush=True)
    predictions = make_single_predictions(clf, train_columns, date_to_predict, filename_list[0])
    for f in filename_list[1:]:
        predictions = np.concatenate([predictions, make_single_predictions(clf, train_columns, date_to_predict, f)])
        gc.collect()
    end_time = time()
    print(" done in {:.4} seconds".format(end_time - start_time), flush=True)
    print("\n")
    return predictions


def make_submission(predictions, filename):
    start_time = time()
    print("preparing submission file", end="...", flush=True)
    submission = pd.read_csv('datas/sample_submission.csv')
    for c in submission.columns[submission.columns != 'ParcelId']:
        submission[c] = predictions
    end_time = time()
    print(" done in {:.4} seconds".format(end_time - start_time), flush=True)
    print("\n")

    start_time = time()
    print("saving csv file", end="...", flush=True)
    submission.to_csv(filename, index=False, float_format='%.4f')
    end_time = time()
    print(" done in {:.4} seconds".format(end_time - start_time), flush=True)
    print("\n")


def make_advanced_predictions(clf, train_columns, filename_list, output_filename):
    n_files = len(filename_list)
    submission = pd.read_csv("datas/sample_submission.csv")
    dates_to_predict = submission.columns[1:]

    for date in dates_to_predict:
        month, year = date[4:], date[:4]
        print("predictions for {0}/{1} : file {2}/{3}".format(month, year, 1, n_files), end="\r", flush=True)
        predictions = make_single_predictions(clf, train_columns, date, filename_list[0])
        for i, f in enumerate(filename_list[1:]):
            print("predictions for {0}/{1} : file {2}/{3}".format(month, year, i+2, n_files), end="\r", flush=True)
            predictions = np.concatenate([predictions, make_single_predictions(clf, train_columns, date, f)])
            gc.collect()
        submission[date] = predictions
        print("\n")

    start_time = time()
    print("saving csv file", end="...", flush=True)
    submission.to_csv(output_filename, index=False, float_format='%.4f')
    end_time = time()
    print(" done in {:.4} seconds".format(end_time - start_time), flush=True)
    print("\n")
