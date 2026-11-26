"""
Reproduce four rock-property prediction models from master.csv.
Models: XGBoost-UCS, XGBoost-E, SparsePolynomial-UCS, SparsePolynomial-E.
Run: python train_rock_models.py master.csv
"""
import sys, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor

csv_path = sys.argv[1] if len(sys.argv)>1 else "master.csv"
raw = pd.read_csv(csv_path)
colmap = {
 "Plagioclase feldspar (%)":"Unnamed: 3", "Alkali feldspar (%)":"Unnamed: 4",
 "Quartz (%)":"Unnamed: 5", "Mica (%)":"Unnamed: 8", "Amphibole (%)":"Unnamed: 9",
 "Density (g/cm³)":"Unnamed: 10", "Porosity (%)":"Unnamed: 11",
 "UCS (MPa)":"Unnamed: 12", "E (GPa)":"Unnamed: 13"
}
features = ["Plagioclase feldspar (%)","Amphibole (%)","Quartz (%)",
 "Alkali feldspar (%)","Density (g/cm³)","Porosity (%)","Mica (%)"]
data = pd.DataFrame({k:pd.to_numeric(raw[v],errors="coerce") for k,v in colmap.items()})
data["Lithology"] = raw["Unnamed: 1"]
data = data.iloc[4:].replace([np.inf,-np.inf],np.nan)
data = data.dropna(subset=features+["UCS (MPa)","E (GPa)"]).copy()
X = data[features].astype(float)
train_idx,test_idx = train_test_split(data.index,test_size=.20,random_state=42)
X_train,X_test = X.loc[train_idx],X.loc[test_idx]
summary=[]
for target in ["UCS","E"]:
    y = data[f"{target} (MPa)" if target=="UCS" else "E (GPa)"].astype(float)
    y_train,y_test = y.loc[train_idx],y.loc[test_idx]
    xgb = XGBRegressor(objective="reg:squarederror",random_state=42,n_jobs=1,tree_method="hist")
    params = {
      "n_estimators":[100,200,300,500],"max_depth":[1,2,3,4],
      "learning_rate":[.01,.03,.05,.08],"subsample":[.7,.85,1.0],
      "colsample_bytree":[.7,.85,1.0],"min_child_weight":[1,3,5],
      "reg_lambda":[1,5,10,20],"reg_alpha":[0,.01,.1,1.0]
    }
    search = RandomizedSearchCV(xgb,param_distributions=params,n_iter=30,scoring="r2",
      cv=KFold(n_splits=5,shuffle=True,random_state=42),random_state=42,n_jobs=-1,refit=True)
    search.fit(X_train,y_train)
    xgb_model=search.best_estimator_
    joblib.dump(xgb_model,f"XGBoost_{target}.joblib")
    p=xgb_model.predict(X_test)
    summary.append((f"XGBoost-{target}",r2_score(y_test,p),search.best_score_))
    poly=Pipeline([
      ("input_scaler",StandardScaler()),
      ("poly",PolynomialFeatures(degree=2,include_bias=False)),
      ("feature_scaler",StandardScaler()),
      ("lasso",LassoCV(alphas=np.logspace(-4,2,100),
       cv=KFold(n_splits=5,shuffle=True,random_state=42),max_iter=100000,n_jobs=-1))
    ])
    poly.fit(X_train,y_train)
    joblib.dump(poly,f"Sparse_Polynomial_{target}.joblib")
    p=poly.predict(X_test)
    cv=cross_val_score(poly,X_train,y_train,cv=KFold(n_splits=5,shuffle=True,random_state=42),scoring="r2")
    summary.append((f"Sparse-Polynomial-{target}",r2_score(y_test,p),cv.mean()))
print("Model | Test R2 | Training-set 5-fold CV R2")
for row in summary: print(f"{row[0]} | {row[1]:.4f} | {row[2]:.4f}")
