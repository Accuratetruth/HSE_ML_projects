#!/usr/bin/env python
# coding: utf-8

# # Прогноз стоимости подержанного автомобиля

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error


# In[2]:


DATASET_PATH = "https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/cars.csv"


# ## Описание датасета

# **Целевая переменная**
# - `selling_price`: цена продажи, числовая
# 
# **Признаки**
# - `name` (string): модель автомобиля
# - `year` (numeric, int): год выпуска с завода-изготовителя
# - `km_driven` (numeric, int): пробег на дату продажи
# - `fuel` (categorical: _Diesel_ или _Petrol_, или _CNG_, или _LPG_, или _electric_): тип топлива
# - `seller_type` (categorical: _Individual_ или _Dealer_, или _Trustmark Dealer_): продавец
# - `transmission` (categorical: _Manual_ или _Automatic_): тип трансмиссии
# - `owner` (categorical: _First Owner_ или _Second Owner_, или _Third Owner_, или _Fourth & Above Owner_): какой по счёту хозяин?
# - `mileage` (string, по смыслу числовой): пробег, требует предобработки
# - `engine` (string, по смыслу числовой): рабочий объем двигателя, требует предобработки
# - `max_power` (string, по смыслу числовой): пиковая мощность двигателя, требует предобработки
# - `torque` (string, по смыслу числовой, а то и 2): крутящий момент, требует предобработки
# - `seats` (numeric, float; по смыслу categorical, int)
# 
# *CC - Cubic Centimeters
# * They are unrelated units so there is no conversion possible. RPM is a measure of rotational speed and NM is a measure of linear distance, concentration or torque depending on capitalization.

# ## Предобработка данных

# In[3]:


cars = pd.read_csv(DATASET_PATH)


# In[4]:


cars.head(5)


# ### Размер датасета

# In[5]:


cars.shape


# ### Тип данных

# In[6]:


cars.info()


# ### Пропуски

# In[7]:


mis_items = cars.isna().sum().to_frame()
mis_items = mis_items.rename(columns = {0: 'missing_values'})
mis_items['%_of_total'] = ((mis_items['missing_values'] / cars.shape[0])*100).round(2)
mis_items.sort_values(by = 'missing_values', ascending = False)


# В столбцах : torque, mileage, engine, seats, max_power присутствуют пропуски - ~ 3% от общего количества

# #### Пропуски относятся к одним и тем же моделям либо нет?

# In[8]:


cars[cars['torque'].isna()]


# In[9]:


# временно заменим пропуски в столбце torque на "no"
cars['torque'] = cars['torque'].fillna('no')
cars.query('torque == "no"').isna().sum().to_frame()


# In[10]:


cars.query('torque == "no"')['name'].value_counts()


# In[11]:


cars.query('torque == "no" ')['fuel'].value_counts()


# Пропуски отнoсятся к одним и тем же моделям авто и плюс затрагивают основные признаки из которых строится цена - удаляем строки с пропусками

# In[12]:


cars['torque'] = cars['torque'].replace('no', np.nan)


# In[13]:


cars.dropna(inplace = True)


# ### Дубли

# In[14]:


cars.duplicated().sum()


# In[15]:


cars[cars.duplicated()].sort_values('name', ascending = True)


# В данном случае дубли возможны, тк машины с одинаковыми характеристиками могут продаваться по одной цене, но чтобы исключить риск переобучения модели - уберем их

# In[16]:


cars = cars.drop_duplicates().reset_index(drop = True)


# In[17]:


cars.info()


# ### Данные по столбцам

# Столбец mileage

# In[18]:


cars['mileage'].str.slice(-5).value_counts()


# In[19]:


# сделаем временную таблицу и посморим распределение по топливу
cars_fuel = cars[['mileage', 'fuel']].reset_index(drop = True)


# In[20]:


cars_fuel['check'] = cars_fuel['mileage'].str.slice(-5)
cars_fuel.groupby('fuel')['check'].value_counts()


# * Для LPG: 1кг = 1,96 литров
# https://www.elgas.com.au/blog/389-lpg-conversions-kg-litres-mj-kwh-and-m3/ 
# * Для CNG: 1кг = 1,7 литров
# https://www.autogasitalia.it/en/faq/metano/how-much-range-do-you-get-with-an-autogas-cng-conversion/
# * Среднее 1 кг = 1,67 литров - берем это значение

# In[21]:


# функция перевода из km/kg в kmpl
def converter(mileage):
    if mileage[-5:] == 'km/kg':
        i = mileage[:-5]
        i = float(i)*1.67
        return round(i,2)
    if mileage[-4:] == 'kmpl':
        return float(mileage[:-5])


# In[22]:


cars['mileage'] = cars['mileage'].apply(converter)


# Столбец engine

# In[23]:


cars['engine'].str.slice(-2).value_counts()


# In[24]:


cars['engine'] = cars['engine'].str.rstrip('CC').astype('int')


# Столбец max_power

# In[25]:


cars['max_power'].str.slice(-3).value_counts()


# In[26]:


cars['max_power'] = cars['max_power'].str.rstrip('bhp').astype('float')


# Столбец seats

# In[27]:


# заменим float на int
cars['seats'] = cars['seats'].astype('int')


# Столбец torque

# In[28]:


cars['torque'].value_counts()


# In[29]:


cars['torque'].unique()


# In[30]:


cars2 = cars['torque'].str.split(r"N|n|@|k|K|/", expand=True)
cars2.columns = ['zero', 'one', 'two', 'three', 'four']
cars2


# In[31]:


cars2.info()


# In[32]:


cars2['zero'] = cars2['zero'].astype('float')


# 1 кг = 9.80665 Н 

# In[33]:


cars2['one'].str.find("g").value_counts()


# In[34]:


cars2['two'].str.find("g").value_counts()


# In[35]:


cars2['one'].str.find("G").value_counts()


# In[36]:


mask = cars2[cars2['one'].str.find("g") == 0].index
mask1 = cars2[cars2['two'].str.find("g") == 0].index
mask2 = cars2[cars2['one'].str.find("G") == 0].index


# In[37]:


cars2.loc[mask, 'zero'] = cars2['zero']*9.80665
cars2.loc[mask1, 'zero'] = cars2['zero']*9.80665
cars2.loc[mask2, 'zero'] = cars2['zero']*9.80665


# In[38]:


cars = cars.join(cars2)
cars.drop(['one', 'two', 'three', 'four', 'torque'], axis = 1, inplace = True)
cars = cars.rename(columns={'zero': 'torque'})


# In[39]:


cars.head()


# Столбец seller_type

# In[40]:


cars['seller_type'].value_counts()


# Заменим Trustmark Dealer на Dealer - тк по сути 2 категории

# In[41]:


cars['seller_type'] = cars['seller_type'].replace('Trustmark Dealer', 'Dealer')


# In[42]:


cars['seller_type'].value_counts()


# Столбец transmission

# In[43]:


cars['transmission'].value_counts()


# Столбец owner

# In[44]:


cars['owner'].value_counts()


# Столбец name

# In[45]:


cars['name'].unique()


# In[46]:


cars.describe()


# In[47]:


cars.describe(include='object')


# ## Разведочный анализ данных

# Есть ли связь между параметрами автомобиля?

# In[48]:


corr = cars.corr()
plt.figure(figsize=(16,16))
sns.heatmap(corr, cmap="crest", annot = True, square = True)
plt.show()


# Матрица корреляций показывает, что:
#    * Присутствует положительная корреляция между мощностью двигателя и ценой (коэффициент = 0.69). Однако, так же стоит обратить внимание на корреляцию между объемом двигателя, крутящим моментом, годом и ценой (коэффициент 0.45-0.42)
#    * Присутствует положительная корреляция между обьемом двигателя и мощностью двигателя (коэффициент = 0.68), а так же между объемом двигателя и количеством сидений (коэффициент = 0.65). Этот момент интересен, тк есть исключения в виде спортивных автомобилей, у которых количество сидений от 2 до 4х.
#    * Присутствует положительная корреляция между крутящим моментом и максимальной мощностью (коэффициент = 0.62)
#    
# Тк нет сильно коррелированных признаков (коэффициент >= 0.9), то удалять ничего не будем

# Посмотрим отдельно пробег и год

# In[49]:


px.scatter(cars,x = 'year',y='km_driven',color='km_driven')


# Однозначно нельзя утверждать, что чем больше год, тем меньше пробег, так же еще могут повлиять частота использования автомобиля владельцем

# Посмотрим отдельно пробег и цену

# In[50]:


px.scatter(cars,x = 'mileage',y='selling_price',color='selling_price')


# Посмотрим отдельно объем двигателя и количество сидений

# In[51]:


px.scatter(cars,x = 'seats',y='engine',color='engine')


# Однозначной зависимости между количеством сидений и объемом двигателя нет. Наибольшее скопление наблюдается у 5 и 7 местных авто. 

# Посмотрим связь количества владельцев и пробега автомобиля

# In[52]:


plt.figure(figsize=(10,6))

sns.barplot(x='owner', y='km_driven', data = cars, palette='summer')
plt.title('owner - km_driven')
plt.show()


# Что и видим, чем больше владельцев, тем больше пробег автомобиля

# Посмотрим связь топлива и пробега

# In[53]:


plt.figure(figsize=(10,6))

sns.barplot(x='fuel', y='mileage', data = cars, palette='summer')
plt.title('fuel - km_driven')
plt.show()


# In[54]:


cars[cars.fuel == 'Diesel']['mileage'].mean(), cars[cars.fuel == 'Petrol']['mileage'].mean()


# In[55]:


cars[cars.fuel == 'LPG']['mileage'].mean(), cars[cars.fuel == 'CNG']['mileage'].mean()


# Ожидаемо, что средний пробег больше у автомобилей на природном газе

# Посмотрим зависимость между ценой и продавцом

# In[56]:


plt.figure(figsize=(10,6))

sns.barplot(x='seller_type', y='selling_price', data = cars, palette='summer')
plt.title('seller_type - selling_price')
plt.show()


# In[57]:


cars[cars.seller_type == 'Individual']['selling_price'].mean(), cars[cars.seller_type == 'Dealer']['selling_price'].mean()


# Дилер ставит в среднем цены почти в 2 раза больше, чем индивидуальный покупатель. Может дилер продает более мощные и более новые машины?

# Посмотрим на связь между продавцом и годом

# In[58]:


plt.figure(figsize=(10,6))

sns.barplot(x='seller_type', y='year', data = cars, palette='summer')
plt.title('seller_type - year')
plt.show()


# In[59]:


cars[cars.seller_type == 'Individual']['year'].mean(), cars[cars.seller_type == 'Dealer']['year'].mean()


# Разница минимальна

# Посмотрим на связь между продавцом и объемом двигателя

# In[60]:


plt.figure(figsize=(10,6))

sns.barplot(x='seller_type', y='engine', data = cars, palette='summer')
plt.title('seller_type - engine')
plt.show()


# In[61]:


cars[cars.seller_type == 'Individual']['engine'].mean(), cars[cars.seller_type == 'Dealer']['engine'].mean()


# Разница в обьеме двигателя между автомобилями, которые продает дилер и индивидуальный продавец - минимальна

# Посмотрим на связь между продавцом и мощностью двигателя

# In[62]:


plt.figure(figsize=(10,6))

sns.barplot(x='seller_type', y='max_power', data = cars, palette='summer')
plt.title('seller_type - max_power')
plt.show()


# In[63]:


cars[cars.seller_type == 'Individual']['max_power'].mean(), cars[cars.seller_type == 'Dealer']['max_power'].mean()


# В среднем мощность двигателя у дилера больше, чем у индивидуального продавца. Получается, что дилер направлен на продажу более мощных автомобилей.

# Посмотрим на связь между типом трансмиссии и ценой

# In[64]:


plt.figure(figsize=(10,6))

sns.barplot(x='transmission', y='selling_price', data = cars, palette='summer')
plt.title('transmission - selling_price')
plt.show()


# In[65]:


cars[cars.transmission == 'Manual']['selling_price'].mean(), cars[cars.transmission == 'Automatic']['selling_price'].mean()


# В среднем у автомата цена выше, чем у механики

# Посмотрим как влияет связка факторов на цену

# In[66]:


plt.figure(figsize=(13,10))

sns.barplot(x='owner', y='selling_price', hue='seller_type', data = cars, palette='summer')
plt.title('owner&seller_type - selling_price')
plt.show()


# Дилер ставит цену выше для машин с количеством владельцев до 3 (включительно) и машины с тест-драйва

# In[67]:


plt.figure(figsize=(10,8))

sns.barplot(x='transmission', y='selling_price', hue='fuel', data = cars, palette='summer')
plt.title('transmission&fuel - selling_price')
plt.show()


# Связка автомат - дизель дает высокую цену, хотя и в механике дизель тоже оказывается дороже

# ## Преобразование категориальных переменных

# Чтобы применить модель машинного обучения, необходимо перевести категориальные признаки в числовые.

# Начнем с бинарных признаков seller_type и transmission  - переводим их в числа 0 и 1.

# In[68]:


cars['seller_type'].value_counts()


# In[69]:


cars['transmission'].value_counts()


# In[70]:


cars['seller_type'] = cars['seller_type'].map({'Individual' : 1, 'Dealer' : 0})
cars['transmission'] = cars['transmission'].map({'Manual' : 1, 'Automatic' : 0})


# Применим One-Hot Encoding для преобразования остальных категориальных признаков (name, fuel, owner) в числовые значения

# In[71]:


encoder = OneHotEncoder(handle_unknown='ignore')
encoder_cars = pd.DataFrame(encoder.fit_transform(cars[['name', 'fuel', 'owner']]).toarray())


# In[72]:


final_cars = cars.join(encoder_cars)


# In[73]:


final_cars.drop(['name', 'fuel', 'owner'], axis= 1 , inplace= True ) 


# In[74]:


final_cars.info()


# In[75]:


cars.info()


# ##  Построение модели прогнозирования 

# Разобьем данные на обучающую и валидационную выборку функцией train_test_split()

# In[76]:


X = final_cars.drop('selling_price', axis = 1) # признаки

y = final_cars['selling_price']  # целевая переменная
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[77]:


X_train.shape, X_test.shape


# Прогнозирование цены относится к задачам регрессии. Выбираем алгоритм случайных деревьев.

# In[78]:


rf_model = RandomForestRegressor(max_depth = 13, random_state = 0)
rf_model.fit(X_train, y_train)


# In[79]:


y_pred = rf_model.predict(X_test)


# Оценим модель - найдем метрики: R2, MAE, MSLE

# In[80]:


metrics.r2_score(y_test, y_pred)


# In[81]:


median_absolute_error(y_test, y_pred)


# Судя по метрикам наша модель уловила достаточно большую долю переменчивости и прогноз хорошо соотносится с реальными значениями целевой переменной. Однако, модель прогнозирует цены на 48840 меньше реальных.

# ## Сохранение модели

# In[82]:


import pickle

with open('rf_model.pickle', 'wb') as cars_model:
    pickle.dump(rf_model, cars_model)


# In[ ]:




