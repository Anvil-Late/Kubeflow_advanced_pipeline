import pandas as pd
import numpy as np
from scipy.stats import boxcox
import re
import math

def count_commas(text):
    """Fonction text.count(",") sachant gérer les nan"""
    if type(text) is str :
        return(text.count(",") + 1)
    else :
        if np.isnan(text):
            return(np.nan)
        else :
            raise(TypeError("Valeur non textuelle et non NA observée"))

def fetch_noFloors(emission_df, neighborhood, property_type):
    neighb_prop_pivot_table = emission_df.dropna().loc[emission_df["in_train"] == 1, :].pivot_table("NumberofFloors", 
                                                               index = "Neighborhood", 
                                                               columns = "PrimaryPropertyType", 
                                                               aggfunc = lambda X: X.quantile(0.5))
    output = neighb_prop_pivot_table.loc[neighborhood, property_type]
    if math.isnan(output):
        print("Pas de données pour ce type de bâtiment dans ce voisinage, extraction de la médiane globale")
        output = round(neighb_prop_pivot_table.loc[:, property_type].quantile(0.5),0)
        
    return(output)

def preprocess_dataset(input_artifact="/Users/antoinevillatte/Documents/GitHub/Kubeflow_example_pipeline/emission_df.csv"):
    emission_df = pd.read_csv(input_artifact)

    # Log of response variable
    emission_df["BCResponse"] = boxcox(emission_df["TotalGHGEmissions"] + 1, -0.1)

    # Fill NAs
        ## Impute SecondLargestPropertyUseType
    emission_df.loc[(emission_df["SecondLargestPropertyUseType"].isna()) & (~emission_df["ListOfAllPropertyUseTypes"].isna()), 
                    "SecondLargestPropertyUseType"] = "None"

        ## Impute SecondLargestPropertyUseTypeGFA
    emission_df.loc[(emission_df["SecondLargestPropertyUseTypeGFA"].isna()) & (~emission_df["ListOfAllPropertyUseTypes"].isna()), 
                    "SecondLargestPropertyUseTypeGFA"] = 0

        ## Impute ThirdLargestPropertyUseType
    emission_df.loc[(emission_df["ThirdLargestPropertyUseType"].isna()) & (~emission_df["ListOfAllPropertyUseTypes"].isna()), 
                    "ThirdLargestPropertyUseType"] = "None"

        ## Impute ThirdLargestPropertyUseTypeGFA
    emission_df.loc[(emission_df["ThirdLargestPropertyUseTypeGFA"].isna()) & (~emission_df["ListOfAllPropertyUseTypes"].isna()), 
                    "ThirdLargestPropertyUseTypeGFA"] = 0

    # Drop columns with too many NaNs
    emission_df.drop(columns = ["YearsENERGYSTARCertified", "Comments", "Outlier"], inplace = True)

    # Only keep non residential buildings
    emission_df = emission_df.loc[emission_df["BuildingType"].apply(lambda X: bool(re.search("^[Nn]on[Rr]esidential", X))), :]
    # Drop Building Type
    emission_df.drop(columns = "BuildingType", inplace = True)

    # Drop low-impact location variables
    emission_df.drop(columns = ["Address", "City", "State", "ZipCode", "TaxParcelIdentificationNumber", "CouncilDistrictCode"], 
                    inplace = True)

    # Drop ID variables
    emission_df.drop(columns = ["OSEBuildingID", "PropertyName", "DefaultData"], inplace = True)

    # Reset index
    emission_df.reset_index(inplace = True, drop = True)

    # Drop variables too highly correlated to the reponse variable
    emission_df.drop(columns = ['SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)',
                                'SiteEnergyUseWN(kBtu)', 'Electricity(kWh)', 'NaturalGas(therms)', 
                                "SiteEUI(kBtu/sf)", "GHGEmissionsIntensity"], inplace = True)


    # Feature engineering : neighborhood type
    emission_df["Neighborhood_type_GHGE"] = "med-low"
    emission_df.loc[emission_df["Neighborhood"].isin(["GREATER DUWAMISH", "SOUTHEAST"]), "Neighborhood_type_GHGE"] = "low"
    emission_df.loc[emission_df["Neighborhood"].isin(["NORTHEAST", "MAGNOLIA / QUEEN ANNE", "SOUTHWEST", "LAKE UNION"]), 
                    "Neighborhood_type_GHGE"] = "med-high"
    emission_df.loc[emission_df["Neighborhood"].isin(["DOWNTOWN", "EAST"]), "Neighborhood_type_GHGE"] = "high"

    # Feature engineering : age
    emission_df["Age"] = emission_df["DataYear"] - emission_df["YearBuilt"]

    # FE : Number of use types
    emission_df["NumberOfUseTypes"] = emission_df["ListOfAllPropertyUseTypes"].apply(count_commas)

    # FE : Proportion occupied by each use
    three_uses_sum = emission_df["LargestPropertyUseTypeGFA"] + emission_df["SecondLargestPropertyUseTypeGFA"] + \
                    emission_df["ThirdLargestPropertyUseTypeGFA"]

    emission_df["PrimaryUseGFARatio"] = round(emission_df["LargestPropertyUseTypeGFA"] / three_uses_sum, 3)
    emission_df["SecondaryUseGFARatio"] = round(emission_df["SecondLargestPropertyUseTypeGFA"] / three_uses_sum, 3)
    emission_df["TerciaryUseGFARatio"] = round(emission_df["ThirdLargestPropertyUseTypeGFA"] / three_uses_sum, 3)

    # FE : proportion of each energy use
    emission_df["NG_ratio"] = round(emission_df["NaturalGas(kBtu)"] / emission_df["SiteEnergyUse(kBtu)"], 3)
    emission_df["Elec_ratio"] = round(emission_df["Electricity(kBtu)"] / emission_df["SiteEnergyUse(kBtu)"], 3)
    emission_df["Steam_ratio"] = round(emission_df["SteamUse(kBtu)"] / emission_df["SiteEnergyUse(kBtu)"], 3)
    emission_df.drop(index = [780, 2781, 2027, 498, 1677], inplace = True) # Impossible values
    emission_df.reset_index(drop = True, inplace = True)

    # Drop useless columns
    clean_data = emission_df.drop(columns = ["DataYear", "Neighborhood", "Longitude", "Latitude", "YearBuilt", "PropertyGFATotal", 
                                            "SteamUse(kBtu)", "Electricity(kBtu)", "NaturalGas(kBtu)", "PrimaryPropertyType"])

    # Drop NAs
    clean_data.dropna(subset = ["LargestPropertyUseType", "TotalGHGEmissions", "NG_ratio"], inplace = True)
    emission_df.dropna().loc[emission_df["in_train"] == 1, :].pivot_table("NumberofFloors", 
                                                                        index = "Neighborhood", 
                                                                        columns = "PrimaryPropertyType", 
                                                                        aggfunc = lambda X: X.quantile(0.5))

    # Fill NAs
    for index in (clean_data.loc[clean_data["NumberofFloors"].isna(), "NumberofFloors"]).index.tolist():
        
        row_neighborhood = emission_df.loc[emission_df.index == index, "Neighborhood"].at[index]
        row_property_type = emission_df.loc[emission_df.index == index, "PrimaryPropertyType"].at[index]
        
        clean_data.loc[clean_data.index ==  index, "NumberofFloors"] = \
            fetch_noFloors(emission_df, row_neighborhood, row_property_type)
        
    clean_data.loc[clean_data["ENERGYSTARScore"].isna(), "ENERGYSTARScore"] = 0
    clean_data.reset_index(inplace = True, drop = True)

    clean_data.to_csv("clean_data.csv", index=False)