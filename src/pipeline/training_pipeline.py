import sys
sys.path.append(".")
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# if __name__=="__main__":
def train_model():
    # Data Ingestion
    data_ingestion = DataIngestion()
    data_ingestion.make_dir()
    # data_ingestion.collect_data()  #if data is not collected
    sequences, labels = data_ingestion.store_data()

    # Data transformation
    data_transformation = DataTransformation(sequences,labels)
    X,y = data_transformation.X_and_y_split()
    X_train,X_test,y_train,y_test = data_transformation.data_split(X,y)

    # Model Trainer
    model_trainer = ModelTrainer()
    model=model_trainer.create_model()
    model_trainer.train_model(model,X_train,y_train)

    # Save the model
    model.save("./model/signdetection.h5")

    return "<h1>Training done Successfully</h1>"
