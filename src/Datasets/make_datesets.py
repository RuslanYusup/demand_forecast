
## подготовка ряда для обучения модели
import pandas as pd
import numpy as np

np.random.seed(42) # фиксируем воспроизводимость


def prepare_time_series_for_model(file_path: str) -> pd.DataFrame:
    """
    Prepares a time series for model training.

    Args:
        file_path (str): Path to the CSV file containing the raw data.

    Returns:
        pd.DataFrame: Processed time series data.
    """
    np.random.seed(42)  # Fixing reproducibility

    # Read the raw data
    data = pd.read_csv(file_path, parse_dates=["month", "date"])

    # Set the index to 'date'
    train_plot = data.set_index('date')

    return train_plot


#%%
