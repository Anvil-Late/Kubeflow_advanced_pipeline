import kfp.components as comp

def train_best_model(best_model : str, 
                     best_hyperparams : dict, 
                     input_X_train_csv : comp.InputPath('CSV'), 
                     input_y_train_csv : comp.InputPath('CSV'),
                     output_pickle_model : comp.OutputPath('Pickle')):
    
    
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    import os
    import pickle
    
    if best_model == 'XGB':
        modfit = xgb.XGBRegressor(objective = "reg:squarederror",
                                   tree_method = 'hist',
                                   eval_metric = ["rmse"],
                                   **best_hyperparams)
    elif best_model == 'SVM':
        modfit = SVR(**best_hyperparams)
    
    elif best_model == "RandomForest":
        modfit = RandomForestRegressor(**best_hyperparams)
    
    else:
        raise ValueError("Model name not recognized : ".format(best_model))
    
    X_train = pd.read_csv(input_X_train_csv)
    y_train = pd.read_csv(input_y_train_csv)
    modfit.fit(X_train, y_train)
    
    with open(output_pickle_model, 'wb') as f:
        pickle.dump(modfit, f)
        
if __name__ == '__main__':
    from kfp.components import create_component_from_func
    base_img = "public.ecr.aws/f6t4n1w1/poc_kf_pipeline:latest"

    train_best_model_op = create_component_from_func(
        train_best_model,
        output_component_file='train_best_model_op.yaml',
        base_image=base_img,
        annotations={
            "author": "Antoine Villatte"}
    )