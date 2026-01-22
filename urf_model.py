from sklearn.ensemble import RandomForestClassifier

def build_urf(params):
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", None),
        random_state=42
    )