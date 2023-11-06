from .util import get_model

model = get_model()


def get_prediction(data_to_predict) -> str:
    prediction = model.predict([data_to_predict])[0]
    return prediction
