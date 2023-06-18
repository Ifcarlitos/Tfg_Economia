import numpy as np
import pandas as pd
import os
import DatosMeteo 
import DatosDemanda
import DatosEstructura


#Analisis de datos 2019

# Importar todos los datos de diferentes archivos en una sola tabla 

ruta = 'C:/Users/crome/Desktop/TFG_ECO/Precios_Electricidad/2019/'

#numero total de archivos dentro de la carpeta
num_archivos = len(os.listdir(ruta))

#lista de todos los archivos dentro de la carpeta
lista_archivos = os.listdir(ruta)

#recorrer todos los archivos y guardarlos en una lista
lista_datos = []
for i in range(num_archivos):
    archivo = lista_archivos[i]

    #leer el archivo y guardar los datos en un dataframe
    datos = pd.read_csv(ruta + '/' + archivo, sep = ';', decimal = '.')

    lista_datos.append(datos)

#concatenar todos los archivos en un solo dataframe
datos = pd.concat(lista_datos)

#Eliminar filas con nada en la primera columna con indice 0
datos = datos.dropna(subset = [datos.columns[0]])

#Eliminar ultima columna
datos = datos.drop(datos.columns[-1], axis = 1)

#pasar a csv
datos.to_csv('C:/Users/crome/Desktop/TFG_ECO/2019.csv', sep = ';', decimal = '.')

# Importar datos de 2019
datos = pd.read_csv('C:/Users/crome/Desktop/TFG_ECO/2019.csv', sep = ';', decimal = '.')

#renombrar columnas
datos = datos.rename(columns = {'Unnamed: 0': 'Año', 'Unnamed: 1': 'Mes', 'Unnamed: 2': 'Dia', 'Unnamed: 3': 'Hora', 'Unnamed: 4': 'Precio'})

#eliminar ultima columna
datos = datos.drop(datos.columns[-1], axis = 1)

#datos de dolumas año, mes, dia y hora de decimal a entero
datos['Año'] = datos['Año'].astype(int)
datos['Mes'] = datos['Mes'].astype(int)
datos['Dia'] = datos['Dia'].astype(int)
datos['Hora'] = datos['Hora'].astype(int)

#eliminar columnas que hora
datos = datos.drop(datos.columns[3], axis = 1)

#agrupar por año, mes y dia
datos = datos.groupby(['Año', 'Mes', 'Dia']).mean()

#pasar a csv pepe.csv
datos.to_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario.csv', sep = ';', decimal = '.')

datos = pd.read_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario.csv', sep = ';', decimal = '.')


#Unir los datos de los precios con los datos de la meteorologia
datos = pd.merge(datos, DatosMeteo.DatosMeteorologicos, on = ['Año', 'Mes', 'Dia'])

#poner columna precio como ultima columna
precio = datos['Precio']
datos = datos.drop(['Precio'], axis = 1)
datos['Precio'] = precio

#pasar a csv
datos.to_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario_Climatico.csv', sep = ';', decimal = '.')

#leer datos
datos = pd.read_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario_Climatico.csv', sep = ';', decimal = '.')

#Unir los datros de los precios con los datos de la demanda
datos = pd.merge(datos, DatosDemanda.DatosDemanda, on = ['Año', 'Mes', 'Dia'])

#poner columna precio como ultima columna
precio = datos['Precio']
datos = datos.drop(['Precio'], axis = 1)
datos['Precio'] = precio

#pasar a csv
datos.to_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario_Climatico_Demanda.csv', sep = ';', decimal = '.')
datos = pd.read_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario_Climatico_Demanda.csv', sep = ';', decimal = '.')

#Unir los datos de los precios con los datos de la estructura
datos = pd.merge(datos, DatosEstructura.DatosEstructura, on = ['Año', 'Mes', 'Dia'])

#poner columna precio como ultima columna
precio = datos['Precio']
datos = datos.drop(['Precio'], axis = 1)
datos['Precio'] = precio

#pasar a csv
datos.to_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario_Climatico_Demanda_Estructura.csv', sep = ';', decimal = '.')
datos = pd.read_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario_Climatico_Demanda_Estructura.csv', sep = ';', decimal = '.')

#ELIMINAR COLUMNAS QUE NO SE VAN A USAR, las tres primeras
datos = datos.drop(datos.columns[0], axis = 1)
datos = datos.drop(datos.columns[0], axis = 1)
datos = datos.drop(datos.columns[0], axis = 1)

#pasar a csv
datos.to_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario_Climatico_Demanda_Estructura.csv', sep = ';', decimal = ',')
datos = pd.read_csv('C:/Users/crome/Desktop/TFG_ECO/datos_Media_Diario_Climatico_Demanda_Estructura.csv', sep = ';', decimal = ',')

#Diccionario de errores cuadraticos medios
diccionario_errores = {}
diccionario_r2 = {} 

#---------------------------------------------------------------------------------------------Modelo de regresion lineal

#Modelo de regresion lineal

print('Modelo de regresion lineal')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#separar datos de entrenamiento y de prueba por el 80% y 20%
X = datos.iloc[:, :-1].values

#datos de la columna precio es la utlima columna
y = datos.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#entrenar el modelo
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predecir los resultados
y_pred = regressor.predict(X_test)

#comparar los resultados
df = pd.DataFrame({'Real': y_test, 'Prediccion': y_pred})
print(df)

print('Resultado del modelo de regresion lineal')
#Error cuadratico medio
from sklearn import metrics
print('Error cuadratico medio:', metrics.mean_squared_error(y_test, y_pred))

#R2
print('R2:', metrics.r2_score(y_test, y_pred))

#agregar el error cuadratico medio al diccionario
diccionario_errores['Regresion Lineal'] = metrics.mean_squared_error(y_test, y_pred)

#agregar el r2 al diccionario
diccionario_r2['Regresion Lineal'] = metrics.r2_score(y_test, y_pred)

#Imprimir coeficientes
print('Coeficientes: ', regressor.coef_)
#Imprimir intercepto
print('Intercepto: ', regressor.intercept_)
#Imprimir ecuacion de la recta
print('Ecuacion de la recta en forma matricial: y= ', regressor.coef_, 'x +', regressor.intercept_)


#---------------------------------------------------------------------------------------------Modelo de regresion polinomial

#Modelo de regresion polinomial

print('Modelo de regresion polinomial')

from sklearn.preprocessing import PolynomialFeatures

#separar datos de entrenamiento y de prueba por el 80% y 20%
X = datos.iloc[:, :-1].values

#datos de la columna precio es la utlima columna
y = datos.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#entrenar el modelo
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

#predecir los resultados
y_pred = regressor.predict(poly_reg.fit_transform(X_test))

#comparar los resultados
df = pd.DataFrame({'Real': y_test, 'Prediccion': y_pred})
print(df)

print('Resultados de la regresion polinomial')
#Error cuadratico medio
from sklearn import metrics
print('Error cuadratico medio:', metrics.mean_squared_error(y_test, y_pred))

#R2
print('R2:', metrics.r2_score(y_test, y_pred))

#agregar el error cuadratico medio al diccionario
diccionario_errores['Regresion Polinomial'] = metrics.mean_squared_error(y_test, y_pred)

#agregar el r2 al diccionario
diccionario_r2['Regresion Polinomial'] = metrics.r2_score(y_test, y_pred)

#Imprimir coeficientes
print('Coeficientes: ', regressor.coef_)
#Imprimir intercepto
print('Intercepto: ', regressor.intercept_)
#Imprimir ecuacion de la recta
print('Ecuacion de la recta en forma matricial: y= ', regressor.coef_, 'x +', regressor.intercept_)

#---------------------------------------------------------------------------------------------Modelo de regresion de Ridge

# Regresión de Ridge

print('Modelo de regresion de Ridge')

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

#separar datos de entrenamiento y de prueba por el 80% y 20%
X = datos.iloc[:, :-1].values

#datos de la columna precio es la utlima columna
y = datos.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

#entrenar el modelo
ridgeReg = Ridge(alpha=0.05)

ridgeReg.fit(X_train,y_train)

#predecir los resultados
y_pred = ridgeReg.predict(X_test)

#comparar los resultados
df = pd.DataFrame({'Real': y_test, 'Prediccion': y_pred})

print(df)

#error cuadratico medio
from sklearn import metrics
print('Error cuadratico medio:', metrics.mean_squared_error(y_test, y_pred))

#R2
print('R2:', metrics.r2_score(y_test, y_pred))

#agregar el error cuadratico medio al diccionario
diccionario_errores['Regresion Ridge'] = metrics.mean_squared_error(y_test, y_pred)

#agregar el r2 al diccionario
diccionario_r2['Regresion Ridge'] = metrics.r2_score(y_test, y_pred)

#---------------------------------------------------------------------------------------------

#Regresión Bayesiana

print('Modelo de regresion Bayesiana')

from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split

#separar datos de entrenamiento y de prueba por el 80% y 20%
X = datos.iloc[:, :-1].values

#datos de la columna precio es la utlima columna
y = datos.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

#entrenar el modelo
bayesianRidge = BayesianRidge()

bayesianRidge.fit(X_train,y_train)

#predecir los resultados
y_pred = bayesianRidge.predict(X_test)

#comparar los resultados
df = pd.DataFrame({'Real': y_test, 'Prediccion': y_pred})

print(df)

#error cuadratico medio
from sklearn import metrics

#pasar y_pred de arraylike a numpy array
y_pred = np.asarray(y_pred)

print('Error cuadratico medio:', metrics.mean_squared_error(y_test, y_pred))

#R2
print('R2:', metrics.r2_score(y_test, y_pred))

#agregar el error cuadratico medio al diccionario
diccionario_errores['Regresion Bayesiana'] = metrics.mean_squared_error(y_test, y_pred)

#agregar el r2 al diccionario
diccionario_r2['Regresion Bayesiana'] = metrics.r2_score(y_test, y_pred)


#---------------------------------------------------------------------------------------------

# Proceso Gaussiano

print('Modelo de proceso Gaussiano')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split

#separar datos de entrenamiento y de prueba por el 80% y 20%
X = datos.iloc[:, :-1].values

#datos de la columna precio es la utlima columna
y = datos.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

#entrenar el modelo
kernel = DotProduct() + WhiteKernel()
gaussianProcess = GaussianProcessRegressor(kernel=kernel, random_state=0)

gaussianProcess.fit(X_train,y_train)

#predecir los resultados
y_pred = gaussianProcess.predict(X_test)

#comparar los resultados
df = pd.DataFrame({'Real': y_test, 'Prediccion': y_pred})

print(df)

#Error cuadratico medio
from sklearn import metrics
print('Error cuadratico medio:', metrics.mean_squared_error(y_test, y_pred))

#R2
print('R2:', metrics.r2_score(y_test, y_pred))

#agregar el error cuadratico medio al diccionario
diccionario_errores['Proceso Gaussiano'] = metrics.mean_squared_error(y_test, y_pred)

#agregar el r2 al diccionario
diccionario_r2['Proceso Gaussiano'] = metrics.r2_score(y_test, y_pred)

#---------------------------------------------------------------------------------------------

#Imprimir el diccionario de errores
print(diccionario_errores)

#Imprimir el diccionario de r2
print(diccionario_r2)

#Buscar el modelo con el menor error cuadratico medio
# minimo = min(diccionario_errores, key=diccionario_errores.get)

# print('El modelo con el menor error cuadratico medio es: ', minimo)


#---------------------------------------------------------------------------------------------Modelo de redes neuronales

#Modelo de redes neuronales con funcion de activacion Sigmoide
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# crear modelo de red neuronal

# Datos x como variables independientes y y como variable dependiente
#y es la ultima columna
X = datos.iloc[:, :-1].values
#x es todo menos la ultima columna
y = datos.iloc[:, -1].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear la red neuronal
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=500, batch_size=32)

# Predecir los precios en el conjunto de prueba
y_pred = model.predict(X_test)

# Comparar los resultados
df = pd.DataFrame({'Real': y_test, 'Prediccion': y_pred.flatten()})
print(df)

# Error cuadratico medio
from sklearn import metrics
print('Error cuadratico medio:', metrics.mean_squared_error(y_test, y_pred))

# R2
print('R2:', metrics.r2_score(y_test, y_pred))

#agregar el error cuadratico medio al diccionario
diccionario_errores['Redes Neuronales'] = metrics.mean_squared_error(y_test, y_pred)

#agregar el r2 al diccionario
diccionario_r2['Redes Neuronales'] = metrics.r2_score(y_test, y_pred)

#agrergar el modelo al diccionario
diccionario_errores['Redes Neuronales'] = metrics.mean_squared_error(y_test, y_pred)

#agregar el r2 al diccionario
diccionario_r2['Redes Neuronales'] = metrics.r2_score(y_test, y_pred)

#---------------------------------------------------------------------------------------------

print(diccionario_errores)

print(diccionario_r2)


#Red Neuronal:
#---------------------------------------------------------------------------------------------Modelo de redes neuronales

#Modelo de redes neuronales con funcion de activacion Sigmoide
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# crear modelo de red neuronal

# Datos x como variables independientes y y como variable dependiente
#y es la ultima columna
X = datos.iloc[:, :-1].values
#x es todo menos la ultima columna
y = datos.iloc[:, -1].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear la red neuronal
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=500, batch_size=32)

# Predecir los precios en el conjunto de prueba
y_pred = model.predict(X_test)

# Comparar los resultados
df = pd.DataFrame({'Real': y_test, 'Prediccion': y_pred.flatten()})
print(df)

# Error cuadratico medio
from sklearn import metrics
print('Error cuadratico medio:', metrics.mean_squared_error(y_test, y_pred))

# R2
print('R2:', metrics.r2_score(y_test, y_pred))

#Grafica de predicciones
import matplotlib.pyplot as plt

# Crear una figura
plt.figure(figsize=(10,6))

# Crear un gráfico de línea con los datos reales
plt.plot(range(len(y_test)), y_test, color = 'red', label = 'Real')

# Crear un gráfico de línea con las predicciones
plt.plot(range(len(y_pred)), y_pred, color = 'blue', label = 'Predicción')

# Añadir títulos y etiquetas
plt.title('Comparación de los valores reales y las predicciones')
plt.xlabel('Índice')
plt.ylabel('Precio')
plt.legend()

# Mostrar la gráfica
plt.show()


#------------------------------