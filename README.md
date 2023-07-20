# Прогноз стоимости подержанного автомобиля

### Задача

Построить предсказательную модель при помощи ML для прогнозирования стоимости подержанного автомобиля. 
    
### Использованные библиотеки:
pandas, scikit-learn, matplotlib, plotly, numpy, seaborn

### Данные

**Целевая переменная**
- `selling_price`: цена продажи, числовая

**Признаки**
- `name` (string): модель автомобиля
- `year` (numeric, int): год выпуска с завода-изготовителя
- `km_driven` (numeric, int): пробег на дату продажи
- `fuel` (categorical: _Diesel_ или _Petrol_, или _CNG_, или _LPG_, или _electric_): тип топлива
- `seller_type` (categorical: _Individual_ или _Dealer_, или _Trustmark Dealer_): продавец
- `transmission` (categorical: _Manual_ или _Automatic_): тип трансмиссии
- `owner` (categorical: _First Owner_ или _Second Owner_, или _Third Owner_, или _Fourth & Above Owner_): какой по счёту хозяин?
- `mileage` (string, по смыслу числовой): пробег, требует предобработки
- `engine` (string, по смыслу числовой): рабочий объем двигателя, требует предобработки (СС)
- `max_power` (string, по смыслу числовой): пиковая мощность двигателя, требует предобработки
- `torque` (string, по смыслу числовой, а то и 2): крутящий момент, требует предобработки (RPM,NM) 
- `seats` (numeric, float; по смыслу categorical, int)

*CC - Cubic Centimeters

*RPM is a measure of rotational speed and NM is a measure of linear distance, concentration or torque depending on capitalization.
