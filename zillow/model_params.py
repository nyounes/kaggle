lgb_params = {"max_bin": 10,
              "learning_rate": 0.00233,
              "boosting_type": "gbdt",
              "objective": "regression",
              "metric": "mae",
              "sub_feature": 0.5,
              "bagging_fraction": 0.7,
              "bagging_freq": 20,
              "num_leaves": 60,
              "min_data": 500,
              "min_hessian": 0.05,
              "verbose": -1
              }


xgb_params = {"eta": 0.00233,
              "objective": "reg:linear",
              "eval_metric": "mae",
              "max_depth": 6,
              "silent": 1
              }



