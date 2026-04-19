import joblib

class ModelService:
    def __init__(self, path="app/models/trained_model.pkl"):
        self.model = joblib.load(path)

    def predict(self, features):
        return self.model.predict([features])[0]