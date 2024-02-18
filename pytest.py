# импортируем необходимые библиотеки
import pytest
import pandas as pd
import datetime

# импортируем класс FeatureEngineering из файла build_features.py
from src.features.build_features import FeatureEngineering

# импортируем функцию predict из файла predict_model.py
from src.models.predict_model import predict


# создаем фикстуру, которая читает данные из папки raw и возвращает объект DataFrame
@pytest.fixture
def data():
    data = pd.read_csv(r'C:\Users\yusup\OneDrive\Рабочий стол\Demand_ forecast\data\raw\sc2021_train_deals.csv', parse_dates=["month", "date"])
    return data

# создаем фикстуру, которая создает объект класса FeatureEngineering и возвращает его
@pytest.fixture
def fe():
    fe = FeatureEngineering()
    return fe

# создаем фикстуру, которая применяет метод transform к данным и возвращает объект DataFrame с фичами
@pytest.fixture
def df(data, fe):
    df = fe.transform(data)
    return df

# создаем фикстуру, которая возвращает дату в формате YYYY-MM
@pytest.fixture
def date():
    date = datetime.datetime(2020, 12, 1)
    return date

# создаем функцию-тест, которая проверяет, что данные подгрузились верно
def test_data_load(data):
    # проверяем, что data не пустой
    assert not data.empty
    # проверяем, что data имеет правильные столбцы
    assert list(data.columns) == ["month", "date", "deal_id", "product_id", "quantity", "price"]
    # проверяем, что data имеет правильные типы данных
    assert data.dtypes["month"] == "datetime64[ns]"
    assert data.dtypes["date"] == "datetime64[ns]"
    assert data.dtypes["deal_id"] == "int64"
    assert data.dtypes["product_id"] == "int64"
    assert data.dtypes["quantity"] == "int64"
    assert data.dtypes["price"] == "float64"

# создаем функцию-тест, которая проверяет, что трансформация прошла
def test_data_transform(df):
    # проверяем, что df не пустой
    assert not df.empty
    # проверяем, что df имеет правильные столбцы
    assert list(df.columns) == ["month", "date", "deal_id", "product_id", "quantity", "price", "category", "seasonality", "discount", "sales"]
    # проверяем, что df имеет правильные типы данных
    assert df.dtypes["month"] == "datetime64[ns]"
    assert df.dtypes["date"] == "datetime64[ns]"
    assert df.dtypes["deal_id"] == "int64"
    assert df.dtypes["product_id"] == "int64"
    assert df.dtypes["quantity"] == "int64"
    assert df.dtypes["price"] == "float64"
    assert df.dtypes["category"] == "object"
    assert df.dtypes["seasonality"] == "float64"
    assert df.dtypes["discount"] == "float64"
    assert df.dtypes["sales"] == "float64"

# создаем функцию-тест, которая проверяет, что предсказание работает
def test_data_predict(df, date):
    # применяем функцию predict к df и date, чтобы получить предсказания
    preds_df = predict(df, date)
    # проверяем, что preds_df не пустой
    assert not preds_df.empty
    # проверяем, что preds_df имеет правильные столбцы
    assert list(preds_df.columns) == ["product_id", "predicted_demand"]
    # проверяем, что preds_df имеет правильные типы данных
    assert preds_df.dtypes["product_id"] == "int64"
    assert preds_df.dtypes["predicted_demand"] == "float64"