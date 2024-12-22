from sklearn.linear_model import LinearRegression as lin_reg
from models.base_model import BaseModel


class LinearRegression(BaseModel):
    def __init__(self, model_name, params):
        self.model = lin_reg(**params)
        super().__init__(model_name, params)

    def train(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)