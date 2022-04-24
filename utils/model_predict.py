def model_predict(model_pickle, X_predict_csv):
    import pandas as pd
    import pickle
    with open(model_pickle, 'rb') as file:
        modfit = pickle.load(file)
    
    X_predict = pd.read_csv(X_predict_csv)
    predictions = modfit.predict(X_predict)
    
    predictions_df = pd.DataFrame()
    predictions_df["y_pred"] = predictions
    
    return predictions_df