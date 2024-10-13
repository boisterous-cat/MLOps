import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BUCKET = "hse-mlops"
EXPERIMENT_NAME = "anastasia_dovgal"
PARENT_RUN_NAME = "boisterous_cat"

models = dict(
    zip(["linReg", "RFReg", "KNReg"], [
        LinearRegression(),
        RandomForestRegressor(n_estimators=100, max_depth=5),
        KNeighborsRegressor(n_neighbors=10),
    ]))

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

def get_experiment_id(experiment_name: str) -> int:
    """
    создание или установка(set) эксперимента
    """
    client = MlflowClient()

    # Проверяем на наличие уже существующего
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is not None:
        exp_id = mlflow.set_experiment(experiment_name).experiment_id
    else:
        # Создаем новый
        exp_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=f"s3://{BUCKET}"
        )

    return exp_id

def download_data() -> None:
    """
    чтение данных
    """
    # Получим датасет California housing
    housing = fetch_california_housing(as_frame=True)
    # Объединим фичи и таргет в один np.array
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

    return data

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    предобработка данных
    """
    
    # Сделать препроцессинг
    # Разделить на фичи и таргет
    X, y = data[FEATURES], data[TARGET]
    # Разделить данные на обучение и тест
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Обучить стандартизатор на train
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,)
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=X.columns,)

    # без этого возникает ошибка
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    
    return X_train_scaled, X_val_scaled, y_train, y_val

def train_model(model,
    model_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,) -> None:
    """
    обучение модели
    логирование (сохранение) обученных моделей в хранилище артефактов на s3
    сбор метрик по каждой модели
    """

    # Обучить модель
    model.fit(X_train, y_train)
    # Сделаем предсказание
    prediction = model.predict(X_val)

    # Создадим валидационный датасет.
    eval_df = X_val.copy()
    eval_df["target"] = y_val

    # Сохраним результаты обучения с помощью MLFlow.
    signature = infer_signature(X_val, prediction)
    model_info = mlflow.sklearn.log_model(
        model,
        model_name,
        signature=signature,
        registered_model_name=f"sk-learn-{model_name}-reg-model")
    
    mlflow.evaluate(
        model=model_info.model_uri,
        data=eval_df,
        targets="target",
        model_type="regressor",
        evaluators=["default"],
    )

if __name__ == "__main__":

    exp_id = get_experiment_id(EXPERIMENT_NAME)

    # Создадим parent run
    with mlflow.start_run(
        run_name=PARENT_RUN_NAME,
        experiment_id=exp_id,
        description="parent",
    ) as parent_run:
        # Запустим child run на каждую модель
        for model_name in models.keys():
            model = models[model_name]

            with mlflow.start_run(
                run_name=model_name,
                experiment_id=exp_id,
                nested=True,
            ) as child_run:

                # Чтение данных
                data = download_data()
                # Предобработка данных
                X_train, X_val, y_train, y_val = prepare_data(data)
                # Обучение и регистрация модели
                train_model(
                    model,
                    model_name,
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                )
                