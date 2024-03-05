
import statsmodels.api as sm
import matplotlib.pyplot as plt

def test_stationarity(timeseries):
    # Определяем статистики
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # Проводим тест Дики-Фуллера
    print('Results of Dickey-Fuller Test:')
    dftest = sm.tsa.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Определение стационарности на основе p-value и других метрик
    if dfoutput['p-value'] <= 0.05 and dfoutput['Test Statistic'] < dfoutput['Critical Value (1%)']:
        print("Ряд стационарен")
    else:
        print("Ряд нестационарен")