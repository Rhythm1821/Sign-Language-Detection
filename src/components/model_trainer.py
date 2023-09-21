
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard

class ModelTrainer:
    def create_model(self):
        LAYERS = [
            LSTM(64,return_sequences=True,activation="relu",input_shape=(30,1662)), # cos each video is 30 frames by 1662 keypoints (X.shape=(90, 30, 1662))
            LSTM(128,return_sequences=True,activation="relu"),
            LSTM(64,return_sequences=False,activation="relu"), # False cos next is a Dense layer which does not require sequences
            Dense(64,activation="relu"),
            Dense(32,activation="relu"),
            Dense(3,activation="softmax")
        ]

        model = Sequential(LAYERS)
        return model


    def train_model(self,model,X_train,y_train):
        model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["categorical_accuracy"])
        model.fit(X_train,y_train,epochs=1000)

        