
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard


class Model:
    def create_model(self):
        LAYERS = [
            LSTM(64,return_sequences=True,activation="relu",input_shape=(30,1662)), # cos each video is 30 frames by 1662 keypoints (X.shape=(90, 30, 1662))
            LSTM(128,return_sequences=True,activation="relu"),
            LSTM(64,return_sequences=False,activation="relu"), # False cos next is a Dense layer which does not require sequences
            Dense(64,activation="relu"),
            Dense(32,activation="relu"),
            Dense(actions.shape[0],activation="softmax")
        ]

        model = Sequential(LAYERS)

        model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["categorical_accuracy"])

        tb_call = TensorBoard(log_dir=log_dir)

        model.fit(X_train,y_train,epochs=1000,callbacks=[tb_call])

        