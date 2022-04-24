def train_best_model(best_model, best_hyperparams, X_train_csv, y_train_csv):
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
    
    X_train = pd.read_csv(X_train_csv)
    y_train = pd.read_csv(y_train_csv)
    modfit.fit(X_train, y_train)
    
    with open(os.path.join(os.getcwd(), 'best_model.pkl'), 'wb') as f:
        pickle.dump(modfit, f)