from xgboost import XGBRegressor
import pandas as pd

import warnings
warnings.simplefilter(action='ignore')

df_train = pd.read_csv("barbara_antigo.csv")
df_test = pd.read_csv("barbara_novo.csv")

X_train, y_train = df_train.drop(columns=['rating']), df_train['rating']
X_test, y_test = df_test.drop(columns=['rating']), df_test['rating']

y_train /=100
y_test /=100

import pickle
with open("pca.pkl","rb") as file:
    pca = pickle.load(file)
    file.close()

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

best = {'objective': 'reg:logistic',
        'n_estimators': 30,
        'min_child_weight': 20,
        'gamma': 0.2,
        'seed': 123}

model = XGBRegressor(**best)
model.fit(X_train, y_train)