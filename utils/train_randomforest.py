#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def train_randomforest(x_train_csv = "X_train.csv", 
                       y_train_csv = "Y_train.csv", 
                       x_test_csv = "X_test.csv", 
                       y_test_csv = "Y_test.csv",
                      hyperopt_iterations = 2):
    global best
    
        
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score
    import hyperopt
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    from random import seed
    
    X_train = pd.read_csv(x_train_csv)
    y_train = pd.read_csv(y_train_csv)
    X_test = pd.read_csv(x_test_csv)
    y_test = pd.read_csv(y_test_csv)

    seed(42)
    def model_accuracy(params):
        rf_reg = RandomForestRegressor(**params)
        return cross_val_score(rf_reg, X_train, y_train["BCResponse"].tolist()).mean()

    space = {
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'max_features': hp.choice('max_features', range(1, 70)),
        'n_estimators': hp.choice('n_estimators', range(100, 500)),
        'min_samples_split' : hp.choice('min_samples_split', range(2, 10)),
        'min_samples_leaf' : hp.choice('min_samples_leaf', range(1, 10)),
        'max_leaf_nodes' : hp.choice('max_leaf_nodes', range(2, 10, 2)),
        'criterion': hp.choice('criterion', ["mse", "mae"])}

    best=0
    def hyperparameter_tuning(space):
        global best
        acc = model_accuracy(space)
        
        if acc > best:
            best = acc
            print ('new best:', best, space)
       
        return {'loss': 1-acc, 'status': STATUS_OK}
    
    
    trials = Trials()
    best = fmin(hyperparameter_tuning, space, algo=tpe.suggest, max_evals=hyperopt_iterations, trials=trials)

    rf_hyperparams = space_eval(space, best)

    modfit_rf = RandomForestRegressor(**rf_hyperparams)
    modfit_rf.fit(X_train, y_train["BCResponse"].tolist())


    rf_mse = mean_squared_error(y_test.to_numpy(), modfit_rf.predict(X_test))
    rf_accuracies = cross_val_score(estimator = modfit_rf, X = X_test, y = y_test["BCResponse"].tolist(), 
                                    cv = 10)
    rf_r2 = rf_accuracies.mean()
    
    return {"MSE" : rf_mse, "R2" : rf_r2}

