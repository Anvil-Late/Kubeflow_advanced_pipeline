
import kfp

merge_and_split_op = kfp.components.load_component_from_file("./kf_utils/merge_and_split_op.yaml")
preprocess_dataset_op = kfp.components.load_component_from_file("./kf_utils/preprocess_dataset_op.yaml")
prepare_data_op = kfp.components.load_component_from_file("./kf_utils/prepare_data_op.yaml")
train_svm_op = kfp.components.load_component_from_file("./kf_utils/train_svm_op.yaml")
train_randomforest_op = kfp.components.load_component_from_file("./kf_utils/train_randomforest_op.yaml")
train_xgb_op = kfp.components.load_component_from_file("./kf_utils/train_xgb_op.yaml")
evaluate_models_op = kfp.components.load_component_from_file("./kf_utils/evaluate_models_op.yaml")
train_best_model_op = kfp.components.load_component_from_file("./kf_utils/train_best_model_op.yaml")
model_predict_op = kfp.components.load_component_from_file("./kf_utils/model_predict_op.yaml")

@kfp.dsl.pipeline(
   name='Emission prediction pipeline',
   description='An example pipeline.'
)
def emission_pipeline(
    bucket,
    data_2015,
    data_2016,
    hyperopt_iterations,
    subfolder
):
    merge_and_split_task = merge_and_split_op(bucket, data_2015, data_2016)
    preprocess_task = preprocess_dataset_op(merge_and_split_task.outputs['output_edfcsv'])
    preparation_task = prepare_data_op(preprocess_task.outputs['output_cleandatacsv'])
    
    rf_train_task = train_randomforest_op(preparation_task.outputs['output_xtraincsv'],
                                         preparation_task.outputs['output_ytraincsv'],
                                         preparation_task.outputs['output_xtestcsv'],
                                         preparation_task.outputs['output_ytestcsv'],
                                         hyperopt_iterations)
    
    xgb_train_task = train_xgb_op(preparation_task.outputs['output_xtraincsv'],
                                 preparation_task.outputs['output_ytraincsv'],
                                 preparation_task.outputs['output_xtestcsv'],
                                 preparation_task.outputs['output_ytestcsv'],
                                 hyperopt_iterations)
    
    svm_train_task = train_svm_op(preparation_task.outputs['output_xtraincsv'],
                                 preparation_task.outputs['output_ytraincsv'],
                                 preparation_task.outputs['output_xtestcsv'],
                                 preparation_task.outputs['output_ytestcsv'],
                                 hyperopt_iterations)
    
    evaluate_models_task = evaluate_models_op(bucket,
                                              subfolder,
                                              svm_train_task.outputs['MSE'],
                                              svm_train_task.outputs['R2'],
                                              svm_train_task.outputs['hyperparams'],
                                              xgb_train_task.outputs['MSE'],
                                              xgb_train_task.outputs['R2'],
                                              xgb_train_task.outputs['hyperparams'],
                                              rf_train_task.outputs['MSE'],
                                              rf_train_task.outputs['R2'],
                                              rf_train_task.outputs['hyperparams']
                                             )
    
    train_best_model_task = train_best_model_op(evaluate_models_task.outputs['best_model'],
                                               evaluate_models_task.outputs['hyperparams'],
                                               preparation_task.outputs['output_xtraincsv'],
                                               preparation_task.outputs['output_ytraincsv'])
    
    model_predict_task = model_predict_op(train_best_model_task.outputs['output_pickle_model'],
                                          preparation_task.outputs['output_xtestcsv'])
