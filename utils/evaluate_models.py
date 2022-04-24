def evaluate_metrics(svm_hyperparams, svm_mse, svm_r2, 
                     xgb_hyperparams, xgb_mse, xgb_r2, 
                     rf_hyperparams, rf_mse, rf_r2):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
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
                  figname="./plots/model_performance_R2.png",
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
                  figname="./plots/models_performance_MSE.png",
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
    return (best_model, best_models_hyperparams)