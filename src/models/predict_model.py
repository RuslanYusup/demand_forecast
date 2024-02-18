import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

from src.features.build_features import FeatureEngineering, FOLDS, key_cols


def predict(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """
    Вычисление предсказаний.

    Параметры:
        df:
          датафрейм, содержащий все сделки с начала тренировочного периода до `month`,
        month:
          месяц, для которого вычисляются предсказания.

    Результат:
        Датафрейм предсказаний для каждой группы, содержащий колонки:
            - `material_code`, `company_code`, `country`, `region`, `manager_code`,
            - `prediction`.
        Предсказанные значения находятся в колонке `prediction`.
    """

    group_ts = df.groupby(key_cols + ["month"])["volume"].sum().unstack(fill_value=0)
    group_ts[month] = 0
    new_df = pd.DataFrame(group_ts.stack()).reset_index()
    new_df = new_df.rename(columns={0: 'volume'})
    new_df['volume'] = np.log1p(new_df['volume'])
    fe_data = FeatureEngineering(new_df).transform()
    fe_data['month'] = fe_data['month'].astype(str)

    predicting_data = fe_data[fe_data['month'] == str(month)[:10]].reset_index()

    model = CatBoostRegressor()
    predictions = pd.DataFrame()
    for i in range(FOLDS):
        model_path = rf'C:\Users\yusup\OneDrive\Рабочий стол\Demand_ forecast\data\processed\cv_model_{i}.cbm'
        model.load_model(model_path)
        prediction = model.predict(predicting_data[model.feature_names_])
        predictions[f'prediction_{i}'] = prediction

    preds_df = predicting_data[key_cols].copy()
    preds_df["prediction"] = np.expm1(np.mean(predictions, axis=1))

    return preds_df
#%%
