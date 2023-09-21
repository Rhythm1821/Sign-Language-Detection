import os
import numpy as np
from keras.models import load_model
from training_pipeline import X_test


class PredictionPipeline:
    def __init__(self):
        self.model = load_model("./model/signdetection.h5")
        self.actions = np.array(['hello', 'thanks', 'iloveyou'])
        
    def predict(self,X_test):
        res = self.model.predict(X_test) if len(X_test.shape)==3 else self.model.predict(np.expand_dims(X_test,axis=1))

        return self.actions[np.argmax(res)]

PredictionPipeline().predict(X_test=X_test)