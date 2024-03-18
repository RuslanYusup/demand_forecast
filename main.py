from src.Datasets.data_set_prep import PrepareTimeSeriesForModel
from src.features.clean_anomalies import ReplaceAnomalies, remove
from src.models.predict_model import PredictProphet, metrics
from src.models.train_model import TrainProphet

file_path = 'C:\\Users\\yusup\\OneDrive\\Рабочий стол\\Demand_ forecast\\data\\raw\\sc2021_train_deals.csv'
periods = 20


class TakeForecast:
    """ Класс для запуска pipeline из следующих шагов: data_set_prep, clean_anomalies, train_model, predict_model
    Аргументы: дата на которую пользователь хочет сделать прогноз и горизонт прогноза
    Возвращает: предсказанные значения
    Функции: predict_prophet - предсказание модели
    """

    def __init__(self, file_path_forecat, period):
        self.file_path = file_path_forecat
        self.periods = period

    def predict_prophet(self):
        # Подготовка данных
        prepare = PrepareTimeSeriesForModel(self.file_path)
        data = prepare.prepare_time_series_for_model()
        # Замена аномальных значений
        replace = ReplaceAnomalies(data)
        replace.replace_anomalies()
        # Удаление лишних столбцов
        data = remove(data)
        # Обучение модели prophet
        model = TrainProphet(data)
        model.train_prophet()
        # Предсказание модели
        predict = PredictProphet(model, data, self.periods)
        forecast = predict.predict_prophet()
        # Вывод результатов
        metrics.show(data['y'], forecast['yhat'])
        return forecast

        # горизонт прогноза
        start_date = forecast["ds"].min()
        end_date = forecast["ds"].max()
        # Пользователь вводит дату на которую он хочет сделать прогноз
        input_date = input(f"Введите дату в диапазоне от {start_date} до {end_date}: ")
        input_date = pd.to_datetime(input_date)
        if input_date < start_date or input_date > end_date:
            print("Введенная дата не находится в диапазоне предсказанных значений.")
        # Получите предсказанное значение и доверительный интервал для введенной даты
        else:
            prediction = forecast[forecast["ds"] == input_date]["yhat"].values[0]
            lower_bound = forecast[forecast["ds"] == input_date]["yhat_lower"].values[0]
            upper_bound = forecast[forecast["ds"] == input_date]["yhat_upper"].values[0]
            print(f"Предсказанное значение для {input_date}: {prediction:.2f}")
            print(f"Доверительный интервал: [{lower_bound:.2f}, {upper_bound:.2f}]")

# %%
