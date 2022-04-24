import kfp.components as comp

def model_predict(input_model_pickle : comp.InputPath('Pickle'), 
                  input_X_predict_csv : comp.InputPath('CSV'),
                  output_prediction_csv : comp.OutputPath('CSV')):
    
    import pandas as pd
    import pickle
    import numpy as np
    
    with open(input_model_pickle, 'rb') as file:
        modfit = pickle.load(file)
    
    X_predict = pd.read_csv(input_X_predict_csv)
    predictions = modfit.predict(X_predict)
    predictions = np.exp(predictions) - 1
    
    predictions_df = pd.DataFrame()
    predictions_df["y_pred"] = predictions
    
    predictions_df.to_csv(output_prediction_csv)
    
if __name__ == '__main__':
    from kfp.components import create_component_from_func
    base_img = "public.ecr.aws/f6t4n1w1/poc_kf_pipeline:latest"

    model_predict_op = create_component_from_func(
        model_predict,
        output_component_file='model_predict_op.yaml',
        base_image=base_img,
        annotations={
            "author": "Antoine Villatte"}
    )