import pandas as pd
import datetime

# импортируем класс FeatureEngineering из файла build_features.py
from src.features.build_features import FeatureEngineering

# импортируем функцию predict из файла predict_model.py
from src.models.predict_model import predict

# читаем данные из папки raw
data = pd.read_csv(r'C:\Users\yusup\OneDrive\Рабочий стол\Demand_ forecast\data\raw\sc2021_train_deals.csv', parse_dates=["month", "date"])
df = data
# создаем объект класса FeatureEngineering
fe = FeatureEngineering(df)
df_processed = fe.transform()

# запрашиваем у пользователя ввести дату в формате YYYY-MM
date_month = input("Введите дату в формате YYYY-MM: ")

# проверяем, что дата валидная и соответствует формату
try:
    date = datetime.datetime.strptime(date_month, "%Y-%m-%dd")
except ValueError:
    print("Неверный формат даты. Попробуйте снова.")
    exit()

# применяем функцию predict к df и date, чтобы получить предсказания
preds_df = predict(df_processed, date_month)

# выводим предсказания на экран
print("Предсказания спроса на продукты в супермаркете:")
print(preds_df)


#%%
