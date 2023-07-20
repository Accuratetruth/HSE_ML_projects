# import libraries
from pickle import dump, load
import pandas as pd
import sklearn


def split_data(df: pd.DataFrame):
    y = df['selling_price']
    X = df[["max_power", "year", "torque", "km_driven", "mileage", "engine", "transmission",
            "seats", "seller_type"]]

    return X, y

def open_data(path="cars_dataset.csv"):
    df = pd.read_csv(path)
    df = df[['selling_price', "max_power", "year", "torque", "km_driven", "mileage", "engine", "transmission",
            "seats", "seller_type"]]

    return df

def load_model_and_predict(df, path="rf_model1.pickle"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    
    return prediction
