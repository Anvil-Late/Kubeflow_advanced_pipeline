#,
#                     model_performance_R2 : comp.OutputArtifact('PNG'),
#                     model_performance_MSE :comp.OutputArtifact('PNG') 
import kfp.components as comp
from typing import NamedTuple

def evaluate_models(bucket : str, 
                     subfolder : str,
                     svm_mse : float, 
                     svm_r2 : float, 
                     svm_hyperparams : dict, 
                     xgb_mse : float, 
                     xgb_r2 : float, 
                     xgb_hyperparams : dict, 
                     rf_mse : float, 
                     rf_r2 : float, 
                     rf_hyperparams : dict
                     )-> NamedTuple('Outputs', [
                         ('best_model', str), 
                         ('hyperparams', dict)
                         ]):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import boto3
    
    def easy_bar_plot(x, y, data, figname, order=None, xlab=None, ylab=None, title=None, grid=True,
                  values_over_bars=True, vob_round=0, vob_offset=None, vob_rot=None, x_tick_rot=None):
    
        fig, ax = plt.subplots(figsize = (18, 8))
        if order is None:
            order = np.sort(data[x].unique()) 
        sns.barplot(x=x, y=y, data=data, ax=ax, order=order)
        if xlab is not None:
            ax.set_xlabel(xlab, fontsize = 16, fontweight = "bold")
        if ylab is not None:
            ax.set_ylabel(ylab, fontsize = 16, fontweight = "bold")
        if title is not None:
            plt.suptitle(title, fontsize = 18, fontweight = "bold")

        if grid :
            plt.grid(b=True, which='major', axis='both', alpha = 0.3)

        if values_over_bars:
            if vob_offset is None:
                vob_offset = 0.015
            if vob_rot is None:
                vob_rot = 0
            if vob_rot > 0:
                ha="left"
            else:
                ha="center"
            pos=0
            for i, (q, val) in data.iterrows():
                if val != 0:
                    ax.text(pos, val + vob_offset*data[y].max(), "{}".format(round(val,vob_round)), 
                            ha=ha, fontsize = 12, fontweight = "bold", rotation=vob_rot, 
                           rotation_mode="anchor")
                pos += 1
        if x_tick_rot is not None:
            plt.xticks(rotation = x_tick_rot, ha="right")
        plt.savefig(figname)
        plt.show()
        
    
    performance_report = {}
    performance_report["SVM"] = {"MSE" : svm_mse, "R2" : svm_r2}
    performance_report["XGB"] = {"MSE" : xgb_mse, "R2" : xgb_r2}
    performance_report["RandomForest"] = {"MSE" : rf_mse, "R2" : rf_r2}
    
    performance_df = pd.DataFrame.from_dict(performance_report, orient="index")
    performance_df = performance_df.reset_index().rename(columns={"index" : "Model"})
    
    hyperparams_dict = {"SVM" : svm_hyperparams, "XGB" : xgb_hyperparams, "RandomForest" : rf_hyperparams}
    
    easy_bar_plot(x="Model", y="R2", data=performance_df[["Model", "R2"]],
                  figname = "./model_performance_R2.png",
                  order=performance_df["Model"], 
                  xlab="Model", 
                  ylab="R2 score", 
                  title="R2 score by Model", 
                  grid=True, 
                  values_over_bars=True, 
                  vob_round=3, 
                  vob_offset=None, 
                  vob_rot=None, 
                  x_tick_rot=None)
    
    easy_bar_plot(x="Model", y="MSE", data=performance_df[["Model", "MSE"]], 
                  figname = "./model_performance_MSE.png",
                  order=performance_df["Model"], 
                  xlab="Model", 
                  ylab="MSE", 
                  title="MSE by Model", 
                  grid=True, 
                  values_over_bars=True, 
                  vob_round=3, 
                  vob_offset=None, 
                  vob_rot=None, 
                  x_tick_rot=None)
    
    best_model = performance_df.loc[performance_df["R2"]==performance_df["R2"].max(), "Model"].values[0]
    best_models_hyperparams = hyperparams_dict[best_model]
    
    s3_resource = boto3.client('s3')
    s3_resource.upload_file("./model_performance_R2.png", 
                            bucket, 
                            subfolder + "/model_performance_R2.png")
    s3_resource.upload_file("./model_performance_MSE.png", 
                            bucket, 
                            subfolder + "/model_performance_MSE.png")
    
    return (best_model, best_models_hyperparams)

if __name__ == '__main__':
    from kfp.components import create_component_from_func
    base_img = "public.ecr.aws/f6t4n1w1/poc_kf_pipeline:latest"

    evaluate_models_op = create_component_from_func(
        evaluate_models,
        output_component_file='evaluate_models_op.yaml',
        base_image=base_img,
        annotations={
            "author": "Antoine Villatte"}
    )