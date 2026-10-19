from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"


def load_data():
    return fetch_california_housing(as_frame=True).frame


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)