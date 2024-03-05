
key_cols = ["material_code", "company_code", "country", "region", "manager_code", "month", "contract_type", 'material_lvl1_name', 'material_lvl2_name', 'material_lvl3_name']
target_col = 'volume'

import make_datesets
import SARIMAFeatureEngineering_cat
import check_stationarity
import clean_anomalies
import train_sarima_cat
import predict_sarima_cat
import plot_sarima_cat
from statsmodels.tools.eval_measures import mean_absolute_percentage_error as MAPE


# читаем данные из папки raw
df = make_datesets.prepare_time_series_for_model(file_path = 'C:\\Users\\yusup\\OneDrive\\Рабочий стол\\Demand_ forecast\\data\\raw\\sc2021_train_deals.csv')

# смотрим на стационарность
check_stationarity.test_stationarity(df['volume'])

# кодируем категориальные признаки
fe = SARIMAFeatureEngineering_cat(data=df, key_cols=key_cols, target_col=target_col)
df_processed_cat = fe.transform()
df_resampled_cat = df_processed_cat.resample(rule='W').mean()


# выявляем и удаляем аномалии
df_cleaned = clean_anomalies.remove_detected_anomalies(df_resampled_cat, "volume")
df_cleaned.drop(columns=['IsAnomaly'], inplace=True)

# разбиваем на обучающую и тестовую выборки
train_eval_cat = df_cleaned[:-33]
test_eval_cat = df_cleaned[-33:]

# обучаем модель
model_sarima_cat = train_sarima_cat(train_eval_cat)

# оцениваем качество модели
pred_sarima_cat = predict_sarima_cat(model_sarima_cat, train_eval_cat, test_eval_cat)

print("MAPE:", MAPE(pred_sarima_cat, test_eval_cat['volume']))

pred_sarima_cat.index = test_eval_cat.index

# строим график
plot_sarima_cat(train_eval_cat, test_eval_cat, pred_sarima_cat)


#%%
