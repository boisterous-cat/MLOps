import io
import json
import logging
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from typing import NoReturn, Any, Dict, List, Literal

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner" : "Anastasia Dovgal",
    "email" : "example@gmail.com",
    "email_on_failure" : False, 
    "email_on_retry" : False,
    "retry" : 3,
    "retry_delay" : timedelta(minutes=1)
}

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get("S3_BUCKET")

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"

models = dict(
    zip(["linReg", "RFReg", "KNReg"], [
        LinearRegression(),
        RandomForestRegressor(n_estimators=100, max_depth=5),
        KNeighborsRegressor(n_neighbors=10),
    ]))

dag = DAG(
    dag_id = "mlops_hw1_dag",
    schedule_interval = "0 1 * * *", #At 01:00 
    start_date = days_ago(2), 
    catchup = False, # чтобы даг не стал наверстывать упущенное: 
    tags = ["mlops"],
    default_args = DEFAULT_ARGS
)

def init() -> Dict[str, Any]: 
    """
    Step0: Pipeline initialisation.
    """
    info = {}
    info["start_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")
    info["dataset_end"] = datetime.now().strftime("%Y-%m-%d")
    # Так как используем библиотечный сет, то поставим дату прошлогоднюю.
    info["dataset_start"] = (datetime.now() -
                           timedelta(365)).strftime("%Y-%m-%d")
    return info

def download_data(**kwargs) -> Dict[str, Any]:
    """
    step1: DownloadData from sklearn.
    """
    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids="init")
    info["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    # Получим датасет California housing
    housing = fetch_california_housing(as_frame=True)
    # Объединим фичи и таргет в один np.array
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

    s3_hook = S3Hook('s3_connection')
    buffer = io.BytesIO()
    data.to_pickle(buffer)
    buffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=buffer, 
        bucket_name=BUCKET,
        key="AnastasiaDovgal/datasets/california_housing.pkl", 
        replace=True)
    _LOG.info("File loaded to s3")

    info["features_number"] = len(data.columns.tolist())
    info["shape"] = data.shape[0]
    info["data_download_end"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    return info

def prepare_data(**kwargs) -> Dict[str, Any]:
    """
    Step 2: Prepare data for training.
    """
    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids="download_data")
    info["data_preparation_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(
        key="AnastasiaDovgal/datasets/california_housing.pkl", 
        bucket_name=BUCKET)
    data = pd.read_pickle(file)

    # Сделать препроцессинг
    # Разделить на фичи и таргет
    X, y = data[FEATURES], data[TARGET]
    # Разделить данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Обучить стандартизатор на train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Сохранить готовые данные на S3
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train_scaled, X_test_scaled, y_train, y_test],
    ):
        filebuffer = io.BytesIO()
        pickle.dump(data, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"datasets/{name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

    _LOG.info("Data prepared.")
    info["features"] = X.columns.tolist()
    info["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    return info

def train_model(**kwargs) -> Dict[str, Any]:
    """
    Step 3: Train logistic regression.
    """

    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids="prepare_data")
    m_name = kwargs["model_name"]

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    # Загрузить готовые данные с S3
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f"datasets/{name}.pkl",
                                     bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    # Обучить модель
    model = models[m_name]
    info[f"{m_name}_train_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_test"])
    y_test = data["y_test"]
    info[f"{m_name}_train_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    # Посчитать метрики
    result = {}
    result[f"{m_name}_r2_score"] = r2_score(y_test, y_pred)
    result[f"{m_name}_mse"] = MSE(y_test, y_pred)
    
    info[f"{m_name}_metrics"] = result
    _LOG.info("Model trained")
    
    return info

def save_results(**kwargs) -> None:
    """
    Step 3: Save results to S3.
    """

    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids=["train_linReg", "train_RFReg", "train_KNReg"])

    result = {}
    for metric in info:
        result.update(metric)

    date = datetime.now().strftime("%Y_%m_%d_%H")
    s3_hook = S3Hook("s3_connection")
    json_byte_object = json.dumps(result).encode()
    s3_hook.load_bytes(json_byte_object, f"AnastasiaDovgal/results/{date}.json", 
                          bucket_name=BUCKET, replace=True)

#задание
task_init = PythonOperator(task_id = "init", python_callable = init, dag = dag)

task_download_data = PythonOperator(task_id = "download_data", python_callable = download_data, dag = dag)

task_prepare_data = PythonOperator(task_id = "prepare_data", python_callable = prepare_data, dag = dag)

task_train_models = [
    PythonOperator(task_id=f"train_{model_name}",
                   python_callable=train_model,
                   dag=dag,
                   provide_context=True,
                   op_kwargs={"model_name": model_name})
    for model_name in models.keys()
]

task_save_results = PythonOperator(task_id="save_results",
                                   python_callable=save_results,
                                   dag=dag,
                                   provide_context=True)

#architecture of tasks
task_init >> task_download_data >> task_prepare_data >> task_train_models >> task_save_results