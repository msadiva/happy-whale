## this file creates a small sample of the data
import pandas as pd
from sklearn import model_selection 
import shutil


train = pd.read_csv("train.csv", low_memory = False)
train["kfold"] = -1

kf = model_selection.StratifiedKFold(n_splits = 10, random_state = 240, shuffle = True)

## fill the new kfold column

for f, (t_, v_) in enumerate(kf.split(X = train, y = train.species)):
    train.loc[v_, "kfold"] = f


## selection fold 0 for experimentation

train_0 = train[train.kfold == 0]
train_0.to_csv("train_sample.csv")

source = "train_images/"

destination = "sample/"

for img in train_0["image"].values :

    source_final = source + img 
    destination_final = destination + img 

    try:
        shutil.copy(source_final, destination_final)
    except:
        print ("Error occured while copying file.")