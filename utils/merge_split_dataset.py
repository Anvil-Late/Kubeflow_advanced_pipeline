import pandas as pd
import re
import boto3
from io import StringIO
from sklearn.model_selection import GroupShuffleSplit

def get_location(text, method):
    """Retrieves data from the 2015 dataset to harmonize it with the 2016 dataset"""
    if method == "latitude":
        match = re.search('{\'latitude\': \'(.+?)\', \'longitude', text)
    elif method == "longitude":
        match = re.search('[0-9]\', \'longitude\': \'(.+?)\', \'human_address\'', text)
    elif method == "address":
        match = re.search('\'{"address": "(.+?)", "city":', text)      
    elif method == "city":
        match = re.search('", "city": "(.+?)", "state":', text)
    elif method == "state":
        match = re.search('"state": "(.+?)", "zip":', text)
    elif method == "zip":
        match = re.search('"zip": "(.+?)"}\'}', text)
    else:
        raise ValueError("Veuillez choisir une méthode (latitude, longitude, adress, city, state, zip)")
        
    if match:
        found = match.group(1)
        return(found)
    return("N/A")

def merge_and_split(bucket, data_2015, data_2016):

    # Load datasets
    csv_strings = {}
    encoding = 'utf-8'
    for source_data, year in zip([data_2015, data_2016], ["2015", "2016"]):
        csv_obj = boto3.client('s3').get_object(Bucket=bucket, Key=source_data)
        body = csv_obj['Body']
        csv_string = body.read().decode(encoding)
        csv_strings[year] = csv_string
        
    data15 = pd.read_csv(StringIO(csv_strings["2015"]))
    data16 = pd.read_csv(StringIO(csv_strings["2016"]))

    # Rename mismatched columns
    rename_cols = {
        "GHGEmissions(MetricTonsCO2e)" : "TotalGHGEmissions",
        "GHGEmissionsIntensity(kgCO2e/ft2)" : "GHGEmissionsIntensity",
        "Comment" : "Comments"
    }
    data15.rename(columns = rename_cols, inplace = True)

    # Extract location info from 2015 dataset to harmonize it
    data15["Latitude"] = data15["Location"].apply(get_location, method = "latitude")
    data15["Longitude"] = data15["Location"].apply(get_location, method = "longitude")
    data15["Address"] = data15["Location"].apply(get_location, method = "address")
    data15["City"] = data15["Location"].apply(get_location, method = "city")
    data15["State"] = data15["Location"].apply(get_location, method = "state")
    data15["ZipCode"] = data15["Location"].apply(get_location, method = "zip")
    data15["ZipCode"] = data15["ZipCode"].astype(int) # convert to numeric
    data15.drop(columns = "Location", inplace = True)

    # Delete columns from data15 that arent in data16
    data15.drop(columns = set(data15.columns.tolist()).difference(data16.columns.tolist()), inplace = True)

    # Harmonize column order
    cols_order = data16.columns.tolist()
    data15 = data15.loc[:, cols_order]

    # Concatenate
    emission_df = pd.concat([data15, data16], axis = 0, ignore_index = True)

    # Train/test split
    inTrain , inTest = next(GroupShuffleSplit(train_size = 0.7, random_state = 42).\
                        split(emission_df, groups = emission_df["OSEBuildingID"]))

    emission_df["in_train"] = 0
    emission_df.iloc[inTrain, 46] = 1

    emission_df.to_csv("emission_df.csv", index=False)


