import pandas as pd
from preprocessing import min_max_normalize
from rfe_selection import select_features
from cscso_optimizer import cscso_optimize
from urf_model import build_urf

# Dummy example workflow
X = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]]).values
y = [0,1,0]

X_norm = min_max_normalize(X)
X_sel = select_features(X_norm, y)

params = {"n_estimators": 100, "max_depth": 5}
opt_params = cscso_optimize(params)

model = build_urf(opt_params)
model.fit(X_sel, y)

print("CSCS-URF model trained successfully")