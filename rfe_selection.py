from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def select_features(X, y, n_features=10):
    model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=n_features)
    X_selected = rfe.fit_transform(X, y)
    return X_selected