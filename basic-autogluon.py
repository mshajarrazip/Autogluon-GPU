from os import makedirs as os_makedirs
from os.path import exists as osp_exists, join as osp_join
from datetime import datetime
from torch import cuda as t_cuda
import logging
from json import dump as j_dump

# Create output folder (all generated files will go here)
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
PATH_TO_OUT = osp_join("output", TIMESTAMP)
if not osp_exists(PATH_TO_OUT):
    os_makedirs(PATH_TO_OUT)

# configure logging before loading autogluon
logging.basicConfig(
    filename=osp_join(PATH_TO_OUT, f"{TIMESTAMP}.log"),
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
    filemode="w"
)

from autogluon.tabular import TabularDataset, TabularPredictor

if __name__ == "__main__":

    logging.info("START TRAINING ~")
    logging.info("Checking if GPU is accessible ...")
    logging.info(
        f"Is available ({t_cuda.is_available()}) "
        f"Device count ({t_cuda.device_count()}) "
        f"Device name ({t_cuda.current_device()}) "
        )
    
    # Load the dataset
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    
    # TabularPredictor: Setup the predictor
    save_path = osp_join(PATH_TO_OUT, "ag_model")
    label = 'class'
    predictor = TabularPredictor(
        label=label,
        path=save_path
    ).fit(
        train_data=train_data,
        time_limit=120, # time limit
        num_gpus=4, # set the number of GPUS
        excluded_model_types=["CAT"] # disable CAT as it is buggy with the GPU
    )
    del predictor # to test loading properly

    # Load predictor and generate predictions
    test_data_nolab = test_data.drop(columns=[label])
    y_test = test_data[label]
    predictor = TabularPredictor.load(save_path)
    y_pred = predictor.predict(test_data_nolab)
    perf = predictor.evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        auxiliary_metrics=True
    )
    with open(osp_join(PATH_TO_OUT, "perf.json"), "w") as f:
        j_dump(perf, f, indent=4)

    logging.info("DONE ~")
