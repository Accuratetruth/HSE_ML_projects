# import libraries
from pickle import dump, load
import pandas as pd


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

def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    to_encode = ['owner']
    for col in to_encode:
        dummy = pd.get_dummies(X_df[col], prefix=col)
        X_df = pd.concat([X_df, dummy], axis=1)
        X_df.drop(col, axis=1, inplace=True)

    if test:
        return X_df, y_df
    else:
        return X_df



def load_model_and_predict(df, path="rf_model.pickle"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    #
    # encode_prediction = {
    #     0: "Сожалеем, вам не повезло",
    #     1: "Ура! Вы будете жить"
    # }
    #
    # prediction = encode_prediction[prediction]

    return prediction
