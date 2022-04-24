import kfp.components as comp

def prepare_data(input_cleandatacsv : comp.InputPath('CSV'),
                output_xtraincsv : comp.OutputPath('CSV'),
                output_ytraincsv : comp.OutputPath('CSV'),
                output_xtestcsv : comp.OutputPath('CSV'),
                output_ytestcsv : comp.OutputPath('CSV')):
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    clean_data = pd.read_csv(input_cleandatacsv)

    # Numeric / categorical separation
    numcols = clean_data.select_dtypes(include = np.number).columns
    catcols = list(set(clean_data.columns) - set(numcols))
    othercols = ["in_train", "SiteEnergyUse(kBtu)", "TotalGHGEmissions", "BCResponse"]
    numcols = [feature for feature in numcols.tolist() if feature not in othercols]

    num_data = clean_data.loc[:, numcols]
    cat_data = clean_data.loc[:, catcols]
    rest_data = clean_data.loc[:, othercols]

    # Normalize
    standardscaler = StandardScaler()
    num_data = standardscaler.fit_transform(num_data)
    num_data = pd.DataFrame(num_data, columns = numcols)

    # Delete outliers
    not_extreme = (num_data.lt(3).all(axis = 1)) & (num_data.gt(-3).all(axis = 1))
    for index in range(len(not_extreme)):
        if rest_data.loc[:, "in_train"].iloc[index] == 0:
            not_extreme[index] = True

    # Encode categorical variables
    ohencoder = OneHotEncoder()
    ohecat_data = ohencoder.fit_transform(cat_data).toarray()
    ohecat_colnames = []
    for feature in catcols:
        for value in clean_data[feature].unique().tolist():
            ohecat_colnames.append(feature + "_" + value)
    ohecat_data = pd.DataFrame(ohecat_data, columns = ohecat_colnames)

    # Remove sparse dummies 
    ohecat_data = ohecat_data.loc[:, ohecat_data.sum().ge(30)]

    # Concatenate
    preproc_data = pd.concat([rest_data, num_data, ohecat_data], axis = 1)
    preproc_data = preproc_data.loc[not_extreme, :]
    rest_data = rest_data.loc[not_extreme, :]

    # train-test split
    train = preproc_data.loc[preproc_data["in_train"] == 1, :].drop(columns = "in_train")
    test = preproc_data.loc[preproc_data["in_train"] == 0, :].drop(columns = "in_train")

    train = train.drop(columns = ["TotalGHGEmissions", "SiteEnergyUse(kBtu)"])
    test = test.drop(columns = ["TotalGHGEmissions", "SiteEnergyUse(kBtu)"])

    # Separate features from preds
    X_train = train.drop(columns = "BCResponse")
    Y_train = train["BCResponse"]

    X_test = test.drop(columns = "BCResponse")
    Y_test = test["BCResponse"]
    

    # Save files 
    output_artifact_list = [output_xtraincsv, output_ytraincsv, output_xtestcsv, output_ytestcsv]
        
    for index, dataset in enumerate([X_train, Y_train, X_test, Y_test]):
        artif = output_artifact_list[index]
        dataset.to_csv(artif, index = False, header = True)

if __name__ == '__main__':
    from kfp.components import create_component_from_func
    base_img = "public.ecr.aws/f6t4n1w1/poc_kf_pipeline:latest"

    prepare_data_op = create_component_from_func(
        prepare_data,
        output_component_file='prepare_data_op.yaml',
        base_image=base_img,
        annotations={
            "author": "Antoine Villatte"}
    )