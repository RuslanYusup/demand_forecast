
import numpy as np


key_cols = ["material_code", "company_code", "country", "region", "manager_code", "contract_type"]


FOLDS = 5

class FeatureEngineering:
    """
    Создает новые признаки:
    1. Lag признаки;
    2. Скользящие средние;
    """

    def __init__(self, df):
        self.df = df

    # генерация случайного шума случайный шум
    def random_noise(self):
        return np.abs(np.random.normal(scale=0.1, size=(len(self.df),)))

        # lag признаки
    def lag_features(self):
        lags = [3, 6, 9, 12, 18]
        for lag in lags:
            self.df['sales_lag_' + str(lag)] = self.df.groupby(key_cols)['volume'].transform(
                lambda x: x.shift(lag)) + self.random_noise() # Adding random noise to each value.

        return self.df

    # скользящие аггрегированные признаки
    def roll_agg_features(self):
        windows = [3, 6, 9, 12, 18]
        for window in windows:
            self.df['sales_roll_mean_' + str(window)] = self.df.groupby(key_cols)['volume']. \
                                                            transform(
                lambda x: x.shift(1).rolling(window=window,
                                             min_periods=3,
                                             win_type="triang").mean()) + self.random_noise()

            self.df['sales_roll_max_' + str(window)] = self.df.groupby(key_cols)['volume']. \
                                                           transform(
                lambda x: x.shift(1).rolling(window=window,
                                             min_periods=3).max()) + self.random_noise()

            self.df['sales_roll_min_' + str(window)] = self.df.groupby(key_cols)['volume']. \
                                                           transform(
                lambda x: x.shift(1).rolling(window=window,
                                             min_periods=3).min()) + self.random_noise()

        return self.df


    # применение обоих методов
    def transform(self):
        self.df = self.lag_features()
        self.df = self.roll_agg_features()

        return self.df



#%%
