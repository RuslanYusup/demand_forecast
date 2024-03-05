
import pandas as pd
import matplotlib.pyplot as plt


## тестовая и обучающая выборка с прогнозами

def plot_sarima_predictions(train_eval_cat, test_eval_cat, pred_sarima_cat):
    """
    Plots SARIMA predictions.

    Args:
        train_eval_cat (pd.DataFrame): Training data for evaluation.
        test_eval_cat (pd.DataFrame): Test data for evaluation.
        pred_sarima_cat (pd.Series): Predicted values from SARIMA model.

    Returns:
        None
    """
    plt.figure(figsize=(15, 6))
    plt.plot(train_eval_cat['volume'], label='Train')
    plt.plot(test_eval_cat['volume'], label='Test')
    plt.plot(pred_sarima_cat, label='Prediction')
    plt.legend()
    plt.show()
