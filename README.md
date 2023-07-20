# Used Cars Price Prediction

### Task

To build predictive model with help of ML to predict used cars price. 
    
### Libraries used:
pandas, scikit-learn, matplotlib, plotly, numpy, seaborn

### Data

**The target variable**
- `selling_price`(int): selling price

**Features**
- `name` (string): car model
- `year` (numeric, int): year of manufacture
- `km_driven` (numeric, int): пробег на дату продажи
- `fuel` (categorical: _Diesel_ or _Petrol_, or _CNG_, or _LPG_, or _electric_): fuel type
- `seller_type` (categorical: _Individual_ or _Dealer_, or _Trustmark Dealer_): seller
- `transmission` (categorical: _Manual_ or _Automatic_): transmission type
- `owner` (categorical: _First Owner_ or _Second Owner_, or _Third Owner_, or _Fourth & Above Owner_): owner
- `mileage` (string): mileage
- `engine` (string): engine capacity (СС)
- `max_power` (string): maximum engine power
- `torque` (string): torque (RPM,NM) 
- `seats` (numeric, float): seats number

*CC - Cubic Centimeters

*RPM is a measure of rotational speed and NM is a measure of linear distance, concentration or torque depending on capitalization.

### Framework

Machine learning solution was presented as a web application using Streamlit framework. App here

### Files
 - Аpp.py: streamlit app file
 - Model_cars.py : script for generating the Random Forest Regressor model
 - cars_dataset.csv and rf_model1.pickle: data file and pre-trained model
 - requirements.txt: package requirements files
 - ML_price_prediction_cars.ipynb: Jupyter notebook with created model

