import kfp.components as comp
from typing import NamedTuple

def train_svm(input_x_train_csv : comp.InputPath('CSV'), 
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
        svm_reg = SVR(**params)
        return cross_val_score(svm_reg, X_train, y_train).mean()

    space = {
        'C': hp.quniform('C', 0.005, 1.0, 0.01),
        'degree': hp.choice('degree', range(2, 6)),
        'coef0' : hp.quniform('coef0', 0.5, 2, 0.2),
        'gamma' : hp.quniform('gamma', 0.005, 0.1, 0.01),
        'kernel': hp.choice('kernel', ["linear", "rbf", "sigmoid"])
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

    svm_hyperparams = space_eval(space, best)

    modfit_svm = SVR(**svm_hyperparams)
    
    modfit_svm.fit(X_train, y_train)

    svm_mse = mean_squared_error(y_test.to_numpy(), modfit_svm.predict(X_test))
    svm_accuracies = cross_val_score(estimator = modfit_svm, X = X_test, y = y_test, 
                                    cv = 10)
    svm_r2 = svm_accuracies.mean()
    
    return(svm_mse, svm_r2, svm_hyperparams)

if __name__ == '__main__':
    from kfp.components import create_component_from_func
    base_img = "public.ecr.aws/f6t4n1w1/poc_kf_pipeline:latest"

    train_svm_op = create_component_from_func(
        train_svm,
        output_component_file='train_svm_op.yaml',
        base_image=base_img,
        annotations={
            "author": "Antoine Villatte"}
    )
