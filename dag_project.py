import io
import json
import logging
import numpy as np
import pandas as pd
import pickle
import mlflow
import os

from mlflow.models import infer_signature

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
DAG_NAME = "anastasia_dovgal_project"
EXPERIMENT_NAME = "anastasia_dovgal_project"
PARENT_RUN_NAME = "boisterous_cat"

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

model_names = ["linReg", "RFReg", "KNReg"]

models = dict(
    zip(model_names, [
        LinearRegression(),
        RandomForestRegressor(n_estimators=100, max_depth=5),
        KNeighborsRegressor(n_neighbors=10),
    ]))

dag = DAG(
    dag_id = DAG_NAME,
    schedule_interval = "0 1 * * *", #At 01:00 
    start_date = days_ago(2), 
    catchup = False, # чтобы даг не стал наверстывать упущенное: 
    tags = ["mlops"],
    default_args = DEFAULT_ARGS
)

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

def get_experiment_id(experiment_name: str) -> int:
    """
    создание или установка(set) эксперимента
    """
    # Проверяем на наличие уже существующего
    configure_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is not None:
        exp_id = mlflow.set_experiment(experiment_name).experiment_id
    else:
        # Создаем новый
        exp_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=f"s3://{BUCKET}"
        )

    return exp_id
    
def init() -> Dict[str, Any]: 
    """
    Step0: Pipeline initialisation.
    """
    info = {}
    info["start_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")

    configure_mlflow()
    # Создаем эксперимент
    exp_id = get_experiment_id(EXPERIMENT_NAME)

    info["experiment_id"] = exp_id
    info["experiment_name"] = EXPERIMENT_NAME

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri) 

    with mlflow.start_run(run_name=PARENT_RUN_NAME, 
                          experiment_id = exp_id, 
                          description = "parent") as parent_run:
        info["run_id"] = parent_run.info.run_id

    _LOG.info("Started run")
    
    return info

def download_data(**kwargs) -> Dict[str, Any]:
    """
    step1: DownloadData from sklearn.
    """
    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids="init")
    info["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    owner = kwargs["owner"]
    owner_path = ''.join(owner.split(' '))
    
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
        key=f"{owner_path}/datasets/california_housing.pkl", 
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
    owner = kwargs["owner"]
    owner_path = ''.join(owner.split(' '))
    
    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(
        key=f"{owner_path}/datasets/california_housing.pkl", 
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
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X.columns)
    
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
            key=f"{owner_path}/datasets/{name}.pkl",
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
    configure_mlflow()
    ti = kwargs["ti"]
    info = ti.xcom_pull(task_ids="prepare_data")
    owner = kwargs["owner"]
    owner_path = ''.join(owner.split(' '))
    m_name = kwargs["model_name"]

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    # Загрузить готовые данные с S3
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f"{owner_path}/datasets/{name}.pkl",
            bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    # Обучить модель
    model = models[m_name]
    info[f"{m_name}_train_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    model.fit(data["X_train"], data["y_train"])
    
    prediction = model.predict(data["X_test"])

    # без этого возникает ошибка
    y_test = data["y_test"]
    y_test.reset_index(drop=True, inplace=True)
    

    # Создадим валидационный датасет.
    eval_df = data["X_test"].copy()
    eval_df["target"] = y_test
    
    
    info[f"{m_name}_train_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    # Посчитать метрики
    result = {}
    result[f"{m_name}_r2_score"] = r2_score(y_test, prediction)
    result[f"{m_name}_mse"] = MSE(y_test, prediction)

    _LOG.info(info["run_id"])

    with mlflow.start_run(run_id=info["run_id"]) as parent_run:
        with mlflow.start_run(run_name=m_name, 
                              experiment_id=info["experiment_id"], 
                              nested=True) as child_run:
            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(data["X_test"], prediction)
            model_info = mlflow.sklearn.log_model(
                model,
                m_name,
                signature=signature,
                registered_model_name=f"sk-learn-{m_name}-reg-model")
            
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )
    
    info[f"{m_name}_metrics"] = result
    _LOG.info("Model trained")
    
    return info

def save_results(**kwargs) -> None:
    """
    Step 3: Save results to S3.
    """

    ti = kwargs["ti"]
    result = {}
    for model_name in models.keys():
        info = ti.xcom_pull(task_ids=f"train_{model_name}")
        result.update(info)

    owner = kwargs["owner"]
    owner_path = ''.join(owner.split(' '))


    date = datetime.now().strftime("%Y_%m_%d_%H")
    s3_hook = S3Hook("s3_connection")
    json_byte_object = json.dumps(result).encode()
    s3_hook.load_bytes(json_byte_object, f"{owner_path}/results/{date}.json", 
                          bucket_name=BUCKET, replace=True)

#задание
task_init = PythonOperator(task_id = "init",
                           python_callable = init, 
                           dag = dag)

task_download_data = PythonOperator(task_id = "download_data", 
                                    python_callable = download_data,
                                    dag = dag,
                                    op_kwargs={'owner': DEFAULT_ARGS["owner"]},
                                    provide_context=True)

task_prepare_data = PythonOperator(task_id = "prepare_data", 
                                   python_callable = prepare_data,
                                   dag = dag,
                                   op_kwargs={'owner': DEFAULT_ARGS["owner"]},
                                   provide_context=True)

task_train_models = [
    PythonOperator(task_id=f"train_{model_name}",
                   python_callable=train_model,
                   dag=dag,
                   provide_context=True,
                   op_kwargs={"model_name": model_name, 'owner': DEFAULT_ARGS["owner"]})
    for model_name in models.keys()
]

task_save_results = PythonOperator(task_id="save_results", 
                                   python_callable=save_results, 
                                   dag=dag,
                                   op_kwargs={'owner': DEFAULT_ARGS["owner"]},
                                   provide_context=True)


#architecture of tasks
task_init >> task_download_data >> task_prepare_data >> task_train_models >> task_save_results