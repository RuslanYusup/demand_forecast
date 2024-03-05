
from sklearn.ensemble import IsolationForest

def remove_detected_anomalies(df, column_name):

    model = IsolationForest(contamination=0.05)
    model.fit(df[[column_name]])

    df["IsAnomaly"] = model.predict(df[[column_name]])

    df_cleaned = df[df["IsAnomaly"] != -1]

    return df_cleaned