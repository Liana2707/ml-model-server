from sklearn.neighbors import KNeighborsClassifier as knn
from models.base_model import BaseModel


class KNNModel(BaseModel):
    def __init__(self, model_name, params):
        self.model = knn(**params)
        super().__init__(model_name, params)

    def train(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)