from xgboost import XGBRegressor
import pandas as pd
import pickle

import warnings
warnings.simplefilter(action='ignore')

df_train = pd.read_csv("twitch_old.csv")

X_train, y_train = df_train.drop(columns=['rating']), df_train['rating']

y_train /=100

with open("pca.pkl","rb") as file:
    pca = pickle.load(file)
    file.close()

X_train = pca.transform(X_train)

best = {'objective': 'reg:logistic',
        'n_estimators': 30,
        'min_child_weight': 20,
        'gamma': 0.2,
        'seed': 123}

model = XGBRegressor(**best)
model.fit(X_train, y_train)

with open("model.pkl","wb") as file:
    pickle.dump(model, file)
    file.close()