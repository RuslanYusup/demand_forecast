from sklearn.preprocessing import LabelEncoder

class SARIMAFeatureEngineering_cat:
    """"
    Encodes categorical features using LabelEncoder
    """
    def __init__(self, data, key_cols, target_col):
        self.data = data
        self.key_cols = key_cols
        self.target_col = target_col

    def encode_categorical_features(self):
        """
        Encodes categorical features using LabelEncoder.
        """

        for col in self.key_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])

    def transform(self):

        self.encode_categorical_features()


        return self.data


