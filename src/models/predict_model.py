from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from prophet import Prophet

from src.Datasets.data_set_prep import PrepareTimeSeriesForModel
from src.features.clean_anomalies import remove
from src.features.clean_anomalies import ReplaceAnomalies
from src.models.train_model import TrainProphet


MAPE = mean_absolute_percentage_error()
MAE = mean_absolute_error()
MSE = mean_squared_error()


periods = 20


class PredictProphet:
    """ Класс для предсказания модели prophet
    Аргументы: обученная модель, датасет для предсказания, горизонт предсказания
    Возвращает: предсказанные значения
    Функции: predict_prophet - предсказание модели
    """

    def __init__(self, model, data, periods):
        self.model = model
        self.data = data
        self.periods = periods

    def predict_prophet(self):
        future = self.model.make_future_dataframe(periods=self.periods)
        forecast = self.model.predict(future)
        return forecast

    def evaluate(self, y_true, y_pred):
        mape = MAPE(y_true, y_pred)
        mae = MAE(y_true, y_pred)
        mse = MSE(y_true, y_pred)
        return mape, mae, mse

    def metrics_show(self, y_true, y_pred):
        mape, mae, mse = self.evaluate(y_true, y_pred)
        print(f'MAPE,%: {mape}' * 100)
        print(f'MAE: {mae}')
        print(f'MSE: {mse}')

    def plot(self, forecast):
        self.model.plot(forecast, figsize=(10, 6))
        plt.show()

file_path = 'C:\\Users\\yusup\\OneDrive\\Рабочий стол\\Demand_ forecast\\data\\raw\\sc2021_train_deals.csv'
prepare = PrepareTimeSeriesForModel(file_path)
data = prepare.prepare_time_series_for_model()
replace = ReplaceAnomalies(data)
replace.replace_anomalies()
data = remove(data)
model = TrainProphet(data)
model.train_prophet()
predict = PredictProphet(model, data, periods)
forecast = predict.predict_prophet()
metrics = predict.evaluate(data['y'], forecast['yhat'])
predict.metrics_show(data['y'], forecast['yhat'])
predict.plot(forecast)
#%%
