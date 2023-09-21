import sys
sys.path.append(".")
from src.components.data_ingestion import DataIngestion

if __name__=="__main__":
    obj = DataIngestion()
    obj.make_dir()
    sequences, labels = obj.store_data()
    X,y = obj.X_and_y_split(sequences,labels)
    X_train,X_test,y_train,y_test = obj.data_split(X,y)