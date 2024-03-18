from prophet import Prophet

from src.Datasets.data_set_prep import PrepareTimeSeriesForModel
from src.features.clean_anomalies import remove
from src.features.clean_anomalies import ReplaceAnomalies


class TrainProphet:
    """ Класс для обучения модели prophet
    Аргументы: датасет
    Возвращает: обученную модель
    Функции: train_prophet - обучение модели
    """

    def __init__(self, data):
        self.data = data

    def train_prophet(self):
        model = Prophet(weekly_seasonality=6, yearly_seasonality=False, seasonality_mode='multiplicative',
                        seasonality_prior_scale=.01, holidays_prior_scale=5)
        model.add_country_holidays(country_name='RU')
        model.fit(self.data)
        return model




file_path = 'C:\\Users\\yusup\\OneDrive\\Рабочий стол\\Demand_ forecast\\data\\raw\\sc2021_train_deals.csv'
prepare = PrepareTimeSeriesForModel(file_path)
data = prepare.prepare_time_series_for_model()
replace = ReplaceAnomalies(data)
replace.replace_anomalies()
data = remove(data)
model = TrainProphet(data)
model.train_prophet()



#%%
