import pandas as pd
import streamlit as st
from PIL import Image
from Model_cars import open_data, split_data, load_model_and_predict

def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('used_car_lot.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Used cars-best cars",
        page_icon=image,

    )

    st.write(
        """
        # Прогноз стоимости подержанного автомобиля
        Определяем стоимость вашего авто
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction):
    st.write("## Стоимость")
    st.write(prediction)

def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    # preprocessed_X_df = preprocess_data(full_X_df, test=False)

    # user_X_df = preprocessed_X_df[:1]
    user_X_df = full_X_df[:1]
    write_user_data(user_X_df)

    prediction = load_model_and_predict(user_X_df)
    write_prediction(prediction)


def sidebar_input_features():
    year = st.sidebar.slider("Год выпуска", min_value=1994, max_value=2023, value=2019,
                            step=1)
    km_driven = st.sidebar.slider("Пробег на дату продажи", min_value=1, max_value=600000, value=300,
                             step=10)
    seller_type = st.sidebar.selectbox("Продавец", ("Частник", "Дилер"))
    transmission = st.sidebar.selectbox("Тип трансмиссии", ("Механика", "Автомат"))
    # owner = st.sidebar.selectbox("Какой по счету хозяин", ("Первый", "Второй", "Третий", "Четвертый и более",
    #                                                        "Машина с тест-драйва"))
    mileage = st.sidebar.slider("Расход топлива", min_value=0, max_value=60, value=20,
                            step=1)
    engine = st.sidebar.slider("Рабочий объем двигателя", min_value=600, max_value=3700, value=800,
                                step=10)
    max_power = st.sidebar.slider("Пиковая мощность двигателя",
        min_value=32, max_value=400, value=100, step=2)
    torque = st.sidebar.slider("Крутящий момент", min_value=48, max_value=2000, value=70, step=2)
    seats = st.sidebar.slider("Количество сидений", min_value=2, max_value=14, value=5, step=1)
    translation = {
        "Частник": 1,
        "Дилер": 0,
        "Механика": 1,
        "Автомат": 0,
        # "Первый": "First Owner",
        # "Второй": "Second Owner",
        # "Третий": "Third Owner",
        # "Четвертый и более": "Fourth & Above Owner",
        # "Машина с тест-драйва": "Test Drive Car",
    }

    data = {
        "year": year,
        "km_driven": km_driven,
        "seller_type": translation[seller_type],
        "transmission": translation[transmission],
        # "owner": translation[owner],
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "torque": torque,
        "seats": seats
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
