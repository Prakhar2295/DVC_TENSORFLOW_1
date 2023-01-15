import argparse
import os
import logging
import io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,classification_report,mean_absolute_error,mean_squared_error,r2_score
from src.utils.data_management import train_valid_generator
from src.utils.all_utils import create_directory, read_yaml
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import json


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir= "logs"
os.makedirs(log_dir,exist_ok = True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),
level=logging.INFO, format = logging_str,filemode= "a")

def evaluate_metrics(actual_values, predicted_values):
    pass

def evaluate_model(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]

    artifacts_dir = config["artifacts"]["ARTIFACTS_DIR"]

    data_dir = config["artifacts"]["DATA_DIR"]
    trained_model_dir = config["artifacts"]["TRAINED_MODEL_DIR"]

    data_fir_path = os.path.join(artifacts_dir, data_dir)
    trained_model_dir_path = os.path.join(artifacts_dir,trained_model_dir)

    trained_model_filename = config["artifacts"]["TRAINED_MODEL_FILENAME"]

    trained_model_file_path = os.path.join(artifacts_dir,trained_model_dir,trained_model_filename)

    reports_dir = config["artifacts"]["reports_dir"]
    scores_file = config["artifacts"]["scores"]

    reports_dir_path = os.path.join(artifacts_dir,reports_dir)
    scores_file_path = os.path.join(artifacts_dir,reports_dir,scores_file)


    create_directory(dirs=[reports_dir_path])


    print(reports_dir_path)    

    print(trained_model_file_path)

    model_1 =load_model(trained_model_file_path)

    train_generator,valid_generator = train_valid_generator(
        data_dir= artifacts["DATA_DIR"],
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmentation=params["AUGMENTATION"]
    )



    Y_pred = model_1.predict(valid_generator,verbose=1)
    y_pred = np.argmax(Y_pred,axis = 1)
    print('Confusion Matrix')
    print(confusion_matrix(valid_generator.classes, y_pred))
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in y_pred]
    print(labels)
    print(predictions)
    filenames=valid_generator.filenames
    results=pd.DataFrame({"Filename":filenames,"Predictions":predictions})
    results.to_json(scores_file_path)

    







    

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c",default ="config/config.yaml")
    args.add_argument("--params","-p",default ="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>> Stage 05 started")
        evaluate_model(config_path = parsed_args.config, params_path= parsed_args.params)
        logging.info(">>> stage 05 completed! training is done >>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e









