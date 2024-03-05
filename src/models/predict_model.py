
def predict_sarima_cat(model_sarima_cat, train_eval_cat, test_eval_cat):

    start_cat = len(train_eval_cat)
    end_cat = len(train_eval_cat) + len(test_eval_cat) - 1

    pred_sarima_cat = model_sarima_cat.predict(start=start_cat, end=end_cat,
                                               exog=test_eval_cat[["material_code", "company_code", "country", "region", "manager_code", "month", "contract_type", 'material_lvl1_name', 'material_lvl2_name', 'material_lvl3_name']],
                                               dynamic=False)

    return pred_sarima_cat
