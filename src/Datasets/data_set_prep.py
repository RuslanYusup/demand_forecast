
import pandas as pd

file_path = 'C:\\Users\\yusup\\OneDrive\\Рабочий стол\\Demand_ forecast\\data\\raw\\sc2021_train_deals.csv'
class PrepareTimeSeriesForModel:
    """ Класс для формирования датасета для обучения модели
    Аргументы: входной файл
    Возвращает: подготовленный датасет для обучения модели
    Функции: prepare_time_series_for_model - подготовка датасета для обучения модели
    - загрузка данных из файла
    - ресемплирование по неделям и подготовка данных для модели prophet
    - удаляет категориальные признаки
    - переименовывает столбцы
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def prepare_time_series_for_model(self):
        data = pd.read_csv(self.file_path, parse_dates=["month", "date"])
        train_plot = data.set_index('date')
        train_plot = train_plot.drop(columns=["material_code", "company_code", "country", "region", "manager_code", "month", "contract_type", 'material_lvl1_name', 'material_lvl2_name', 'material_lvl3_name'])
        train_plot = train_plot.resample(rule='w').mean().reset_index()
        train_plot = train_plot.rename(columns={'date': 'ds', 'volume': 'y'})
        return train_plot





#%%
