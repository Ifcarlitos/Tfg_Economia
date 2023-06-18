import requests
import json
import pandas as pd
import numpy as np

#Generar link para descargar los datos de la API de AEMET
#https://opendata.aemet.es/centrodedescargas/productosAEMET?

Actualizar_datos = False

if Actualizar_datos:
    querystring = {"api_key":"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjcm9tZXJvbWF0YXJpbkBnbWFpbC5jb20iLCJqdGkiOiIwMTlkZDU0ZC1mOTA2LTQ1ZDUtYmQyMC0xOTUyNmE4ODYxODEiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTY3Nzc5MjcwNiwidXNlcklkIjoiMDE5ZGQ1NGQtZjkwNi00NWQ1LWJkMjAtMTk1MjZhODg2MTgxIiwicm9sZSI6IiJ9.CLKSDvMa5IZVpa1-mZzTOAubglMfFtlXrgG1bKW2uf8"}

    headers = {
        'cache-control': "no-cache"
        }

    #DATOS DE BARCELONA

    url = "https://opendata.aemet.es/opendata/sh/8e275d34"

    response = requests.request("GET", url, headers=headers, params=querystring)

    print(response.text)

    datosJsonBarcelona = response.json()

    #DATOS DE MADRID

    url2 = "https://opendata.aemet.es/opendata/sh/34e5f627"

    response2 = requests.request("GET", url2, headers=headers, params=querystring)

    print(response2.text)

    datosJsonMadrid = response2.json()

    #DATOS DE SEVILLA

    url3 = "https://opendata.aemet.es/opendata/sh/736a3fdd"

    response3 = requests.request("GET", url3, headers=headers, params=querystring)

    print(response3.text)

    datosJsonSevilla = response3.json()

    #DATOS DE a coruña

    url4 = "https://opendata.aemet.es/opendata/sh/450b2a78"

    response4 = requests.request("GET", url4, headers=headers, params=querystring)

    print(response4.text)

    datosJsonCoruna = response4.json()

    #DATOS DE a Navarra

    url5 = "https://opendata.aemet.es/opendata/sh/d0f9e9a9"

    response5 = requests.request("GET", url5, headers=headers, params=querystring)

    print(response5.text)

    datosJsonNavarra = response5.json()

    #crear archivo json en la carpeta de datos_climaticos

    with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosBarcelona.json', 'w') as file:
        json.dump(datosJsonBarcelona, file)

    with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosMadrid.json', 'w') as file:
        json.dump(datosJsonMadrid, file)

    with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosSevilla.json', 'w') as file:
        json.dump(datosJsonSevilla, file)

    with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosCoruna.json', 'w') as file:
        json.dump(datosJsonCoruna, file)

    with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosNavarra.json', 'w') as file:
        json.dump(datosJsonNavarra, file)

#Pasar datos de array json a dataframe
import pandas as pd

#datos de barcelona
with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosBarcelona.json') as file:
    datosJsonBarcelona = json.load(file)

#convertir array json a dataframe
#{'fecha': '2019-01-01', 'indicativo': '0201D', 'nombre': 'BARCELONA', 
# 'provincia': 'BARCELONA', 'altitud': '6', 'tmed': '11,4', 
# 'prec': '0,0', 'tmin': '6,6', 'horatmin': '03:40', 'tmax': '16,2', 
# 'horatmax': '12:30', 'dir': '01', 'velmedia': '3,1', 'racha': '6,4', 
# 'horaracha': '07:30'}
dfBarcelona = pd.DataFrame(datosJsonBarcelona)

#datos de madrid
with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosMadrid.json') as file:
    datosJsonMadrid = json.load(file)

#convertir array json a dataframe
dfMadrid = pd.DataFrame(datosJsonMadrid)

#datos de sevilla
with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosSevilla.json') as file:
    datosJsonSevilla = json.load(file)

#convertir array json a dataframe
dfSevilla = pd.DataFrame(datosJsonSevilla)

#datos de a coruña
with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosCoruna.json') as file:
    datosJsonCoruna = json.load(file)

#convertir array json a dataframe
dfCoruna = pd.DataFrame(datosJsonCoruna)

#datos de navarra
with open('C:/Users/crome/Desktop/TFG_ECO/datos_climaticos/datosNavarra.json') as file:
    datosJsonNavarra = json.load(file)

#convertir array json a dataframe
dfNavarra = pd.DataFrame(datosJsonNavarra)

#unir los dataframes en uno solo
df = pd.concat([dfBarcelona, dfMadrid, dfSevilla, dfCoruna, dfNavarra], ignore_index=True)

#eliminar columnas que no se van a utilizar
df = df.drop(['indicativo', 'altitud', 'horatmin', 'horatmax', 'dir', 'velmedia', 'racha', 'horaracha','presMax', 'horaPresMax','presMin','horaPresMin','sol'], axis=1)


#diferentes valores que puede tomar la columna prec
#print(df['prec'].unique())
# ['Ip' 'Acum' '9,9' '9,8' '9,6' '9,5' '9,4' '9,3' '9,1' '9,0' '80,9' '8,8'
#  '8,6' '8,5' '8,4' '8,2' '8,0' '7,8' '7,7' '7,6' '7,4' '7,3' '7,2' '7,0'
#  '6,8' '6,7' '6,6' '6,4' '6,3' '6,2' '6,1' '6,0' '58,3' '53,2' '50,0'
#  '5,9' '5,8' '5,7' '5,6' '5,4' '5,3' '5,2' '5,1' '48,6' '47,2' '42,2'
#  '42,0' '4,8' '4,7' '4,6' '4,4' '4,3' '4,2' '4,1' '4,0' '39,1' '38,5'
#  '38,4' '36,7' '36,6' '34,4' '32,4' '3,9' '3,8' '3,6' '3,5' '3,4' '3,3'
#  '3,2' '3,1' '3,0' '29,7' '29,6' '28,5' '28,2' '27,6' '27,3' '27,2' '26,9'
#  '26,8' '26,4' '25,6' '25,2' '24,9' '24,8' '24,0' '23,8' '23,2' '23,0'
#  '22,8' '22,6' '21,8' '21,6' '20,6' '20,0' '2,9' '2,8' '2,7' '2,6' '2,5'
#  '2,4' '2,3' '2,2' '2,1' '2,0' '18,6' '18,4' '18,3' '18,2' '18,1' '18,0'
#  '17,9' '17,8' '17,4' '17,3' '17,2' '17,0' '16,9' '16,8' '16,2' '15,8'
#  '15,6' '15,2' '15,0' '14,9' '14,6' '14,4' '13,9' '13,7' '13,4' '13,2'
#  '12,9' '12,8' '12,6' '12,4' '12,1' '12,0' '11,8' '11,6' '11,2' '11,1'
#  '10,9' '10,8' '10,6' '10,4' '10,2' '10,1' '10,0' '1,9' '1,8' '1,7' '1,6'
#  '1,5' '1,4' '1,3' '1,2' '1,1' '1,0' '0,9' '0,8' '0,7' '0,6' '0,5' '0,4'
#  '0,3' '0,2' '0,1' '0,0' nan]

#Convertir datos de la columna prec (ip, acum) a nan
df['prec'] = df['prec'].replace(['Ip', 'Acum'], np.nan)

#convertir , por . en los datos de la columna prec
df['prec'] = df['prec'].str.replace(',','.')

#convertir datos de la columna prec a float
df['prec'] = df['prec'].astype(float)

#convertir , por . en los datos de la columna tmin
df['tmin'] = df['tmin'].str.replace(',','.')

#convertir datos de la columna tmin a float
df['tmin'] = df['tmin'].astype(float)

#convertir , por . en los datos de la columna tmax
df['tmax'] = df['tmax'].str.replace(',','.')

#convertir datos de la columna tmax a float
df['tmax'] = df['tmax'].astype(float)

#convertir , por . en los datos de la columna tmed
df['tmed'] = df['tmed'].str.replace(',','.')

#convertir datos de la columna tmed a float
df['tmed'] = df['tmed'].astype(float)

#valores nan de la columna prec a la media de la columna
df['prec'] = df['prec'].fillna(df['prec'].mean())

#valores nan de la columna tmin a la media de la columna
df['tmin'] = df['tmin'].fillna(df['tmin'].mean())

#valores nan de la columna tmax a la media de la columna
df['tmax'] = df['tmax'].fillna(df['tmax'].mean())

#valores nan de la columna tmed a la media de la columna
df['tmed'] = df['tmed'].fillna(df['tmed'].mean())

#convertir datos de la columna fecha a datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')

#convertir datos de la columna fecha a datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')

#agrupar por fecha y calcular la media de los datos
df = df.groupby('fecha').mean()

#separar la columna fecha en dia, mes y año
df['Dia'] = df.index.day
df['Mes'] = df.index.month
df['Año'] = df.index.year

#convertir datos de la columna dia a int
df['Dia'] = df['Dia'].astype(int)

#convertir datos de la columna mes a int
df['Mes'] = df['Mes'].astype(int)

#convertir datos de la columna año a int
df['Año'] = df['Año'].astype(int)

DatosMeteorologicos = df
