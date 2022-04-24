import kfp.components as comp
from typing import NamedTuple

def train_xgb(input_x_train_csv : comp.InputPath('CSV'), 
              input_y_train_csv : comp.InputPath('CSV'), 
              input_x_test_csv : comp.InputPath('CSV'), 
              input_y_test_csv : comp.InputPath('CSV'),
              hyperopt_iterations : int
  )-> NamedTuple('Outputs', [('MSE', float), ('R2', float), ('hyperparams', dict)]):
                             
    global best
    
        
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.svm import SVR
    import hyperopt
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    from random import seed
    
    X_train = pd.read_csv(input_x_train_csv)
    y_train = pd.read_csv(input_y_train_csv)
    X_test = pd.read_csv(input_x_test_csv)
    y_test = pd.read_csv(input_y_test_csv)

    seed(42)
    def model_accuracy(params):
        xgb_reg = xgb.XGBRegressor(objective = "reg:squarederror",
                                   tree_method = 'hist',
                                   eval_metric = ["rmse"],
                                   **params)
        return cross_val_score(xgb_reg, X_train, y_train).mean()

    space = {
        'max_depth' : hp.choice('max_depth', range(1, 30, 1)),
        'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
        'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
        'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
        'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
        'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
        'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1.0, 0.01),
        'colsample_bynode' : hp.quniform('colsample_bynode', 0.1, 1.0, 0.01),
        'max_delta_step' : hp.choice('max_delta_step', range(0, 11, 1))
    }
    
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

    xgb_hyperparams = space_eval(space, best)

    modfit_xgb = xgb.XGBRegressor(objective = "reg:squarederror",
                                      tree_method = 'hist',
                                      eval_metric = ["rmse"],
                                      **xgb_hyperparams)
    
    modfit_xgb.fit(X_train, y_train)

    xgb_mse = mean_squared_error(y_test.to_numpy(), modfit_xgb.predict(X_test))
    xgb_accuracies = cross_val_score(estimator = modfit_xgb, X = X_test, y = y_test, 
                                    cv = 10)
    xgb_r2 = xgb_accuracies.mean()
    
    return(xgb_mse, xgb_r2, xgb_hyperparams)

if __name__ == '__main__':
    from kfp.components import create_component_from_func
    base_img = "public.ecr.aws/f6t4n1w1/poc_kf_pipeline:latest"

    train_xgb_op = create_component_from_func(
        train_xgb,
        output_component_file='train_xgb_op.yaml',
        base_image=base_img,
        annotations={
            "author": "Antoine Villatte"}
    )
