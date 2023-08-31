import joblib
import numpy as np
import pandas as pd


class Estimater():
    def __init__(self, preprocessor, model, usecols=None, target="price"):
        self.preprocessor = preprocessor
        self.usecols = usecols
        self.model = model
        self.target = target

    def predict(self, x:pd.DataFrame) -> np.ndarray:
        if self.usecols==None:
            x = self.preprocessor.transform(x)[self.usecols]
        else:
            x = self.preprocessor.transform(x)
        if self.targe=="price":
            return self.model.predict(x.values)
        elif self.target=="log(price)":
            return np.exp(self.model.predict(x.values))
        elif self.target=="price/year":
            return np.exp(self.model.predict(x.values)) * x["year"]

    def dump(self, path):
        joblib.dump(self, path)
        return self
