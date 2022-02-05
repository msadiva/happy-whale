## this file creates validation split for the whole train data
import pandas as pd


train = pd.read_csv("train.csv", low_memory = False)
train["total_count"] = train.groupby(["individual_id"])["individual_id"].transform('count')
train_grouped = train.groupby(["individual_id"]).apply(lambda x: x.sample(frac = 0.2, random_state = 34))
train_merged = pd.merge(left = train, right = train_grouped, on = 'image', how = 'left', suffixes = ('', '_y'))
train_merged["is_valid"] = False
train_merged["is_valid"] = train_merged["individual_id_y"].isna() != True
train_merged.drop(["species_y", "individual_id_y", "total_count_y"], axis = 1, inplace = True)
train_merged.to_csv("train_validation.csv", index = False)