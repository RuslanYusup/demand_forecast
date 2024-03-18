from src.Datasets.data_set_prep import PrepareTimeSeriesForModel


class ReplaceAnomalies:
    """ Класс для замены аномальных значений на среднее скользящее значение
    Аргументы: подготовленный на этапе data_set_prep датасет
    Возвращает: очищенный датасет
    Функции: replace_anomalies - замена аномальных значений на среднее скользящее значение
    """

    def __init__(self, data):
        self.data = data

    def replace_anomalies(self):
        self.data['moving_average'] = self.data.rolling(window=300, min_periods=1, center=True, on='ds')['y'].mean()
        self.data['std_dev'] = self.data.rolling(window=300, min_periods=1, center=True, on='ds')['y'].std()
        self.data['lower'] = self.data['moving_average'] - 1.65 * self.data['std_dev']
        self.data['upper'] = self.data['moving_average'] + 1.65 * self.data['std_dev']
        self.data = self.data[(self.data['y'] < self.data['upper']) & (self.data['y'] > self.data['lower'])]
        return self.data
    def plot(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.scatter(x=self.data['ds'], y=self.data['y'], c='r', alpha=.5, marker='x', s=30)
        plt.scatter(x=self.data['ds'], y=self.data['y'], c='#0072B2')
        plt.ticklabel_format(style='plain', axis='y')
        plt.xlabel('Дата')
        plt.ylabel('Количество продаж')
        plt.show()

def remove(data):
    data.drop(['moving_average', 'std_dev', 'lower', 'upper'], axis=1, inplace=True)
    data = data.reset_index(drop=True)
    data.columns = ['ds', 'y']
    return data


file_path = 'C:\\Users\\yusup\\OneDrive\\Рабочий стол\\Demand_ forecast\\data\\raw\\sc2021_train_deals.csv'
prepare = PrepareTimeSeriesForModel(file_path)
data = prepare.prepare_time_series_for_model()
replace = ReplaceAnomalies(data)
replace.replace_anomalies()
data = remove(data)
print(data.columns)




#%%

#%%
