import os
from sklearn.pipeline import Pipeline
from joblib import load


def get_model() -> Pipeline:
    model_path = os.environ.get('OLDPWD', 'model/model.pkl')
    return load(f"{model_path}/model/model.pkl")
