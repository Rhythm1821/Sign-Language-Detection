import sys
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append(".")
from src.components.data_ingestion import DataIngestion


class DataTransformation:
    def __init__(self,sequences,labels):
        self.sequences = sequences
        self.labels = labels
        
    def X_and_y_split(self):
        X = np.array(self.sequences)
        num_classes = len(np.unique(self.labels))
        y = np.eye(num_classes)[self.labels].astype(int)

        return X,y
    
    def data_split(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05)

        return X_train,X_test,y_train,y_test