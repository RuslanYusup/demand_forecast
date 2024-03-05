from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima_cat(train_eval_cat):
    model_sarima_cat = SARIMAX(train_eval_cat['volume'],
                               exog=train_eval_cat[["material_code",
                                                    "company_code",
                                                    "country",
                                                    "region",
                                                    "manager_code",
                                                    "month",
                                                    "contract_type",
                                                    'material_lvl1_name',
                                                    'material_lvl2_name',
                                                    'material_lvl3_name']],
                               order=(1, 0, 0),
                               seasonal_order=(0, 0, 0, 52)).fit()

    return model_sarima_cat


